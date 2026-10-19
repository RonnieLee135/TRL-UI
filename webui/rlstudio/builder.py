"""Turn a WebUI form submission into (a) a validated parameter set, (b) a real TRL command,
and (c) for the experimental algorithms, a generated training script.

The rules here mirror TRL itself:

* parameters are passed as `--name value`, exactly like `HfArgumentParser` expects;
* `bool` flags are only emitted when True (so TRL's own defaults survive);
* nothing the user left empty is passed — "unset" and "explicitly the default" differ;
* the CLI algorithms (`trl <algo>`) get `--accelerate_config`/`--num_processes`, which
  `trl/cli/accelerate_config.py` forwards to `accelerate launch`;
* experimental algorithms have no CLI subcommand, so we generate a script modelled on
  `trl/scripts/grpo.py` and launch it with `accelerate launch`.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

ACCELERATE_CONFIGS = ["none", "single_gpu", "multi_gpu", "zero1", "zero2", "zero3", "fsdp1", "fsdp2"]

BUILTIN_REWARDS = [
    "accuracy_reward",
    "reasoning_accuracy_reward",
    "think_format_reward",
    "get_soft_overlong_punishment",
    "get_repetition_penalty_reward",
    "get_cosine_scaled_reward",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item for item in re.split(r"[,\s]+", value.strip()) if item]
    return [value]


def _jsonable(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


class Builder:
    def __init__(self, catalog, trl_path: Path, runs_dir: Path):
        self.catalog = catalog
        self.trl_path = Path(trl_path)
        self.runs_dir = Path(runs_dir)

    # ------------------------------------------------------------------ schema
    def _schema(self, algorithm_id: str) -> dict[str, dict]:
        schema = self.catalog.schema_for(algorithm_id)
        return {field["name"]: field for field in schema["fields"]}

    # -------------------------------------------------------------- normalize
    def normalize(self, algorithm_id: str, params: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
        """Coerce form values to the types TRL expects; report what could not be coerced."""

        schema = self._schema(algorithm_id)
        out: dict[str, Any] = {}
        errors: list[dict] = []
        for name, raw in params.items():
            if raw is None or raw == "" or raw == []:
                continue
            if name not in schema:
                errors.append({"level": "error", "field": name, "message": f"未知参数 {name}（算法 {algorithm_id} 不支持）"})
                continue
            if name == "datasets":
                out[name] = _jsonable(raw)
                continue
            field = schema[name]
            ui = field["ui"]
            try:
                if ui == "toggle":
                    value: Any = _as_bool(raw)
                elif ui == "int":
                    value = int(float(raw))
                elif ui == "number":
                    value = float(raw)
                elif ui == "list":
                    value = [int(v) if str(v).lstrip("-").isdigit() else (float(v) if _is_float(v) else str(v)) for v in _as_list(raw)]
                elif ui == "json":
                    value = _jsonable(raw)
                    if isinstance(value, str):
                        raise ValueError("不是合法 JSON")
                else:
                    value = str(raw).strip()
            except (TypeError, ValueError) as exc:
                errors.append({"level": "error", "field": name, "message": f"{name} 取值不合法：{raw!r}（{exc}）"})
                continue
            out[name] = value
        return out, errors

    # -------------------------------------------------------------- validate
    def validate(self, algorithm_id: str, params: dict[str, Any], *, num_processes: int = 1, env: dict | None = None) -> dict:
        if self.catalog.algorithm(algorithm_id) is None:
            return {"errors": [{"level": "error", "field": "algorithm", "message": f"未知算法 {algorithm_id}"}], "warnings": [], "normalized": {}}
        algo = self.catalog.algorithm(algorithm_id)
        schema = self._schema(algorithm_id)
        normalized, errors = self.normalize(algorithm_id, params)
        warnings: list[dict] = []

        def warn(field: str, message: str) -> None:
            warnings.append({"level": "warn", "field": field, "message": message})

        def error(field: str, message: str) -> None:
            errors.append({"level": "error", "field": field, "message": message})

        requires = algo["requires"]
        if not normalized.get("model_name_or_path"):
            error("model_name_or_path", "必须指定策略模型（model_name_or_path）")
        if not normalized.get("dataset_name") and not normalized.get("datasets"):
            error("dataset_name", "必须指定数据集（dataset_name，或提供 datasets 混合配置）")
        if requires.get("reward_funcs") and not (normalized.get("reward_funcs") or normalized.get("reward_model_name_or_path")):
            error("reward_funcs", f"{algo['label']} 需要至少一个奖励函数或奖励模型")
        if requires.get("teacher") in {True, "remote"} and not (
            normalized.get("teacher_model_name_or_path") or normalized.get("teacher_model_server_url") or normalized.get("teacher_server_urls")
        ):
            error("teacher_model_name_or_path", f"{algo['label']} 需要教师模型（teacher_model_name_or_path 或远程教师地址）")
        if requires.get("vllm") == "required" and not normalized.get("vllm_server_base_url") and not normalized.get("teacher_model_server_url") and not normalized.get("teacher_server_urls"):
            warn("vllm_server_base_url", "该算法要求外置 vLLM server，请填好 server 地址并先启动服务")

        for name, value in normalized.items():
            field = schema.get(name)
            if field and field.get("choices") and value not in field["choices"]:
                # list-valued fields (e.g. DPO loss_type) are validated element-wise
                if isinstance(value, list):
                    bad = [v for v in value if v not in field["choices"]]
                    if bad:
                        error(name, f"{name} 含有非法取值 {bad}，可选：{field['choices']}")
                else:
                    error(name, f"{name}={value!r} 不在可选值中：{field['choices']}")

        # Algorithm specific rules, all taken from the TRL docstrings.
        loss_type = normalized.get("loss_type")
        if algorithm_id in {"grpo", "gmpo"}:
            generations = normalized.get("num_generations", 8)
            batch = (
                int(normalized.get("per_device_train_batch_size", 8) or 8)
                * max(1, int(num_processes))
                * int(normalized.get("gradient_accumulation_steps", 1) or 1)
            )
            if isinstance(generations, int) and generations > 1 and batch % generations:
                warn(
                    "num_generations",
                    f"有效 batch（per_device_train_batch_size × 进程数 × 梯度累积 = {batch}）不能被 num_generations={generations} 整除，GRPO 要求整除",
                )
            if loss_type == "luspo" and normalized.get("importance_sampling_level") != "sequence":
                error("importance_sampling_level", "loss_type=luspo 要求 importance_sampling_level='sequence'")
            if normalized.get("importance_sampling_level") == "sequence" and normalized.get("beta", 0):
                warn("beta", "序列级重要性比（GSPO）通常配合 beta=0.0 使用")
        if algorithm_id == "kto" and normalized.get("loss_type") != "apo_zero_unpaired" and normalized.get("train_sampling_strategy") not in {None, "sequential"}:
            error("train_sampling_strategy", "KTO 的 KL 项依赖同 batch 邻居配对，train_sampling_strategy 必须是 'sequential'（或改用 apo_zero_unpaired）")
        if algorithm_id == "dpo":
            values = normalized.get("loss_type")
            weights = normalized.get("loss_weights")
            if isinstance(values, list) and len(values) > 1 and not weights:
                warn("loss_weights", "组合多种 loss_type 时应给出等长的 loss_weights")
            if isinstance(values, list) and isinstance(weights, list) and len(values) != len(weights):
                error("loss_weights", f"loss_type（{len(values)} 项）与 loss_weights（{len(weights)} 项）长度不一致")
        if "num_generations" in normalized and int(normalized["num_generations"]) < 2 and requires.get("generation"):
            warn("num_generations", "num_generations<2 时组内没有对比基线，优势会退化为 0")
        if normalized.get("use_vllm") and env and "vllm" not in (env.get("installed") or []):
            warn("use_vllm", "当前环境没有安装 vllm，real 模式会直接报错")
        if normalized.get("load_in_4bit") and algorithm_id in {"dpo", "kto"}:
            warn("load_in_4bit", "离线偏好族用 4bit 量化会显著拖慢并影响参考模型对数概率的精度")

        return {
            "algorithm": algorithm_id,
            "errors": errors,
            "warnings": warnings,
            "normalized": normalized,
            "ok": not errors,
        }

    # ------------------------------------------------------------------ build
    def build(self, *, job_id: str, algorithm_id: str, params: dict[str, Any], resource: dict, env: dict | None = None) -> dict:
        """Validate, then produce the launch command and the files that go with it."""

        num_processes = max(1, int(resource.get("num_processes") or 1))
        result = self.validate(algorithm_id, params, num_processes=num_processes, env=env)
        if not result["ok"]:
            return {"ok": False, "errors": result["errors"], "warnings": result["warnings"]}
        algo = self.catalog.algorithm(algorithm_id)
        values = result["normalized"]

        job_dir = self.runs_dir / job_id
        output_dir = values.get("output_dir") or str(job_dir / "output")
        values["output_dir"] = output_dir

        accel = resource.get("accelerate_config") or ("multi_gpu" if num_processes > 1 else "single_gpu")
        if accel not in ACCELERATE_CONFIGS:
            accel = "single_gpu"

        files: dict[str, str] = {}
        if algo["launch"].get("kind") == "cli":
            command = self._cli_command(algorithm_id, values, accel, num_processes)
            entry = f"trl {algorithm_id}"
        else:
            script = self._train_script(job_id, algo, values, env)
            files[f"train_{algorithm_id}.py"] = script
            entry = f"accelerate launch train_{algorithm_id}.py"
            command = self._python_command(algorithm_id, values, accel, num_processes, job_dir)

        return {
            "ok": True,
            "warnings": result["warnings"],
            "errors": [],
            "command": command,
            "entry": entry,
            "output_dir": output_dir,
            "job_dir": str(job_dir),
            "files": files,
            "params": values,
            "accelerate_config": accel,
            "num_processes": num_processes,
        }

    # ------------------------------------------------------------- commands
    def _cli_command(self, algorithm_id: str, values: dict, accel: str, num_processes: int) -> list[str]:
        from shutil import which

        prefix = [which("trl") or "trl", algorithm_id]
        args: list[str] = []
        for name, value in self._ordered(values).items():
            args += self._flag(name, value)
        if accel != "none":
            args += ["--accelerate_config", accel]
        if num_processes > 1:
            args += ["--num_processes", str(num_processes)]
        return [*prefix, *args]

    def _python_command(self, algorithm_id: str, values: dict, accel: str, num_processes: int, job_dir: Path) -> list[str]:
        import sys

        script = str(job_dir / f"train_{algorithm_id}.py")
        args: list[str] = []
        for name, value in self._ordered(values).items():
            args += self._flag(name, value)
        if accel == "none":
            return [sys.executable, script, *args]
        config = self.trl_path / "trl" / "accelerate_configs" / f"{accel}.yaml"
        command = ["accelerate", "launch", "--config_file", str(config)]
        if num_processes > 1:
            command += ["--num_processes", str(num_processes)]
        return [*command, script, *args]

    def _ordered(self, values: dict) -> dict:
        """Put the arguments people read first (model, data, reward) in a stable order."""

        priority = [
            "model_name_or_path",
            "model_revision",
            "dtype",
            "dataset_name",
            "dataset_config",
            "dataset_train_split",
            "dataset_test_split",
            "reward_funcs",
            "reward_model_name_or_path",
            "teacher_model_name_or_path",
            "output_dir",
            "learning_rate",
            "max_steps",
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
        ]
        ordered = {name: values[name] for name in priority if name in values}
        for name, value in values.items():
            if name not in ordered:
                ordered[name] = value
        return ordered

    @staticmethod
    def _flag(name: str, value: Any) -> list[str]:
        if isinstance(value, bool):
            return [f"--{name}"] if value else []
        if isinstance(value, (list, tuple)):
            return [f"--{name}", *[str(v) for v in value]]
        if isinstance(value, dict):
            return [f"--{name}", json.dumps(value, ensure_ascii=False)]
        return [f"--{name}", str(value)]

    # ------------------------------------------------------- generated script
    def _train_script(self, job_id: str, algo: dict, values: dict, env: dict | None) -> str:
        """Generate a runnable script for algorithms that have no `trl <algo>` subcommand."""

        module = algo["config_module"].replace("/", ".").removesuffix(".py")
        package = module.rsplit(".", 1)[0]
        config_cls, trainer_cls = algo["config_class"], algo["trainer_class"]
        init_args = algo.get("trainer_init_args", [])
        needs_reward = "reward_funcs" in init_args
        needs_teacher = "teacher_model" in init_args
        roles = algo["data"]["roles"]

        reward_block = ""
        reward_kwarg = ""
        if needs_reward:
            reward_block = f'''
@dataclass
class StudioScriptArguments(ScriptArguments):
    """Extra script arguments: the reward functions to score completions with."""

    reward_funcs: list[str] | None = field(
        default=None,
        metadata={{"help": "内置奖励函数名（{"、".join(BUILTIN_REWARDS)}）或点分导入路径"}},
    )
    reward_model_name_or_path: str | None = field(
        default=None, metadata={{"help": "奖励模型的模型 id 或路径"}}
    )
'''
            reward_kwarg = "    reward_funcs=reward_funcs,\n"
        else:
            reward_block = ""

        script_args_cls = "StudioScriptArguments" if needs_reward else "ScriptArguments"

        reward_build = ""
        if needs_reward:
            reward_build = '''
    # ---- 监督信号（S3）：内置奖励 + 点分路径自定义奖励 + 可选奖励模型
    reward_funcs = []
    if script_args.reward_model_name_or_path:
        reward_funcs.append(script_args.reward_model_name_or_path)
    for name in script_args.reward_funcs or []:
        if name in REWARD_REGISTRY:
            reward_funcs.append(REWARD_REGISTRY[name])
        elif "." in name:
            module_path, attr = name.rsplit(".", 1)
            sys.path.insert(0, os.getcwd())
            reward_funcs.append(getattr(importlib.import_module(module_path), attr))
        else:
            raise ValueError(f"无法加载奖励函数 {name!r}：既不是内置名也不是合法的导入路径")
'''
        teacher_kwarg = ""
        teacher_note = ""
        if needs_teacher:
            teacher_kwarg = "    teacher_model=teacher_model,\n"
            teacher_note = """
    # ---- 教师模型（S1）：可以与学生共用 tokenizer，也可以是远程 server
    teacher_model = getattr(training_args, "teacher_model_name_or_path", None) or script_args.teacher_model
    if teacher_model is None:
        raise ValueError("该算法需要教师模型：请填 teacher_model_name_or_path")
"""

        return f'''"""本文件由 TRL-UI 生成（任务 {job_id}），不需要手工维护。

算法：{algo["label"]}（{algo["full_name"]}）
训练器：{trainer_cls} @ {module}
数据角色：{"、".join(roles)}

结构照搬 `trl/scripts/grpo.py`：四个 dataclass 对应四个正交平面
（任务 / 算法 / 模型 / 数据），由 TrlParser 解析成命令行参数。
"""

import importlib
import os
import sys
from dataclasses import dataclass, field

from accelerate.logging import get_logger
from datasets import load_dataset

from trl import DatasetMixtureConfig, ModelConfig, ScriptArguments, TrlParser, get_dataset
from trl.rewards import (
    accuracy_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    get_soft_overlong_punishment,
    reasoning_accuracy_reward,
    think_format_reward,
)

from {package} import {config_cls}, {trainer_cls}

logger = get_logger(__name__)

REWARD_REGISTRY = {{
    "accuracy_reward": accuracy_reward,
    "reasoning_accuracy_reward": reasoning_accuracy_reward,
    "think_format_reward": think_format_reward,
    "get_repetition_penalty_reward": get_repetition_penalty_reward,
    "get_cosine_scaled_reward": get_cosine_scaled_reward,
    "get_soft_overlong_punishment": get_soft_overlong_punishment(max_completion_len=1280, soft_punish_cache=256),
}}
{reward_block}

def main(script_args, training_args, model_args, dataset_args):
    # ---- 数据与提示（S2）
    if dataset_args.datasets:
        dataset = get_dataset(dataset_args)
    elif script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
            streaming=script_args.dataset_streaming,
        )
    else:
        raise ValueError("必须提供 `dataset_name` 或 `datasets` 之一。")
{reward_build}{teacher_note}
    # ---- 组装训练器：模型（S1） + 数据（S2） + 信号（S3）
    trainer = {trainer_cls}(
        model=model_args.model_name_or_path,
{reward_kwarg}{teacher_kwarg}        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"✅ 训练完成，模型已保存到 {{training_args.output_dir}}")


def make_parser(subparsers=None, prog=None):
    dataclass_types = ({script_args_cls}, {config_cls}, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        return subparsers.add_parser("{algo['id']}", help="{algo['label']} training script", dataclass_types=dataclass_types)
    return TrlParser(dataclass_types, prog=prog)


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args, dataset_args)
'''

    # ------------------------------------------------------------- artifacts
    @staticmethod
    def shell_command(command: list[str]) -> str:
        return " \\\n  ".join(shlex.quote(part) for part in command)


def _is_float(value: Any) -> bool:
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False
