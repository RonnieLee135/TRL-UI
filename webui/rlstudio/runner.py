"""Job execution.

Two modes, and the difference is visible everywhere in the UI:

* ``real``   — run the generated command with `subprocess`, stream stdout/stderr into the
               store, parse TRL's `{'loss': ..., 'reward': ...}` log dicts into metrics.
* ``simulate`` — no GPU / no torch on this machine, so a built-in simulator emits the log
               lines and metrics the chosen algorithm would produce, at a realistic pace.
               Simulated jobs are tagged `模拟` in the UI and never presented as results.

The simulator is deliberately explicit about being a simulator: its first line says so,
and the job record carries ``mode="simulate"``.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

from . import envvars
from .builder import Builder
from .store import Store

LOG_DICT_RE = re.compile(r"\{'loss':.*\}")
METRIC_RE = re.compile(r"'([A-Za-z0-9_/\.\-]+)':\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")
STEP_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)")

# Which metrics each family produces, used by the simulator and by the chart defaults.
SIM_METRICS = {
    "online-rl": ["loss", "reward", "reward_std", "kl", "entropy", "completions/mean_length"],
    "preference": ["loss", "rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies"],
    "distillation": ["loss", "distillation_loss", "jsd", "entropy"],
    "reward": ["loss", "accuracy", "margin"],
    "sft": ["loss", "entropy", "mean_token_accuracy"],
}


class Runner:
    def __init__(self, store: Store, builder: Builder, runs_dir: Path, catalog, env_provider=None):
        self.store = store
        self.builder = builder
        self.runs_dir = Path(runs_dir)
        self.catalog = catalog
        # The user-set environment variables of the 「环境」 page (HF_HOME / HF_ENDPOINT /
        # HF_TOKEN / CUDA_VISIBLE_DEVICES), read again at every launch so an edit applies to
        # the next job without restarting the server.
        self.env_provider = env_provider
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def env_vars(self) -> dict[str, str]:
        overrides = self.env_provider() if self.env_provider else {}
        return {name: value for name, value in (overrides or {}).items() if value}

    # ------------------------------------------------------------------ public
    def is_running(self, job_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(job_id)
            return proc is not None and proc.poll() is None

    def start(self, job: dict) -> dict:
        if self.is_running(job["id"]):
            return {"ok": False, "message": "任务已经在运行"}
        if job["status"] in {"running"}:
            return {"ok": False, "message": "任务状态为 running"}
        algo = self.catalog.algorithm(job["algorithm"])
        job_dir = self.runs_dir / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)

        self.store.update_job(job["id"], status="running", started_at=time.time(), finished_at=None, exit_code=None, message=None)
        self.store.append_event(job["id"], f"$ {self.builder.shell_command(job['command'])}", "system")
        if job["mode"] == "simulate":
            thread = threading.Thread(target=self._run_simulated, args=(job, algo), daemon=True)
        else:
            thread = threading.Thread(target=self._run_real, args=(job,), daemon=True)
        thread.start()
        return {"ok": True}

    def stop(self, job: dict) -> dict:
        with self._lock:
            proc = self._procs.get(job["id"])
        if proc is not None and proc.poll() is None:
            proc.terminate()
            self.store.append_event(job["id"], "收到停止请求，正在终止进程…", "system")
            return {"ok": True, "message": "已发送终止信号"}
        if job["status"] == "running":
            self.store.update_job(job["id"], status="stopped", finished_at=time.time(), message="已停止（模拟任务）")
            self.store.append_event(job["id"], "已停止（模拟任务）", "system")
            return {"ok": True, "message": "已停止"}
        return {"ok": False, "message": "任务没有在运行"}

    # --------------------------------------------------------------- real mode
    def _run_real(self, job: dict) -> None:
        job_dir = self.runs_dir / job["id"]
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TRL_STUDIO_JOB", job["id"])
        overrides = self.env_vars()
        env.update(overrides)  # what the user set on the 「环境」 page wins over the server's own env
        command = job["command"]
        self.store.append_event(job["id"], f"工作目录：{job_dir}", "system")
        if overrides:
            self.store.append_event(job["id"], f"注入环境变量：{envvars.summarize(overrides)}", "system")
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(job_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            self._fail(job["id"], f"无法启动命令：{exc}。请确认 trl / accelerate 已安装并在 PATH 中。")
            return
        with self._lock:
            self._procs[job["id"]] = proc
        assert proc.stdout is not None
        step = 0
        for line in proc.stdout:
            line = line.rstrip("\n")
            self.store.append_event(job["id"], line)
            parsed = self._parse_metrics(line)
            if parsed:
                step += 1
                for name, value in parsed.items():
                    self.store.append_metric(job["id"], step, name, value)
        code = proc.wait()
        with self._lock:
            self._procs.pop(job["id"], None)
        if code == 0:
            self.store.append_event(job["id"], "训练进程正常退出（exit code 0）", "system")
            self.store.update_job(job["id"], status="finished", finished_at=time.time(), exit_code=code)
        else:
            self.store.append_event(job["id"], f"训练进程退出，exit code = {code}", "system")
            self.store.update_job(job["id"], status="failed", finished_at=time.time(), exit_code=code, message=f"exit code {code}")

    def _fail(self, job_id: str, message: str) -> None:
        self.store.append_event(job_id, message, "stderr")
        self.store.update_job(job_id, status="failed", finished_at=time.time(), message=message)

    @staticmethod
    def _parse_metrics(line: str) -> dict[str, float]:
        if not LOG_DICT_RE.search(line) and "'loss'" not in line:
            return {}
        metrics: dict[str, float] = {}
        for name, raw in METRIC_RE.findall(line):
            if name in {"epoch", "step"}:
                continue
            try:
                metrics[name] = float(raw)
            except ValueError:
                continue
        return metrics

    # ----------------------------------------------------------- simulate mode
    def _run_simulated(self, job: dict, algo: dict) -> None:
        family = algo["family"]
        total_steps = int(job["params"].get("max_steps") or 60)
        total_steps = max(4, min(total_steps, 200))
        metrics = SIM_METRICS.get(family, ["loss"])
        rng = random.Random(job["id"])
        self.store.append_event(
            job["id"],
            "[模拟模式] 本机没有可用的 torch/加速器，这里不会真的训练。"
            "输出用于演示界面与产物（metrics 曲线、日志解析、checkpoint 路径），不代表任何训练结果。",
            "system",
        )
        self.store.append_event(job["id"], f"[模拟模式] 算法 {algo['label']}（{family}），共 {total_steps} 步", "system")
        self.store.append_event(job["id"], f"[模拟模式] 命令：{self.builder.shell_command(job['command'])}", "system")
        base_loss = 1.4 if family in {"online-rl", "preference"} else 2.2
        for step in range(1, total_steps + 1):
            if self.store.get_job(job["id"])["status"] != "running":
                self.store.append_event(job["id"], "[模拟模式] 已被停止", "system")
                return
            progress = step / total_steps
            loss = base_loss * math.exp(-2.2 * progress) + rng.uniform(-0.02, 0.02)
            values = {"loss": round(loss, 4)}
            if family == "online-rl":
                values["reward"] = round(0.2 + 0.55 * (1 - math.exp(-2.4 * progress)) + rng.uniform(-0.03, 0.03), 4)
                values["reward_std"] = round(max(0.01, 0.4 - 0.2 * progress + rng.uniform(-0.02, 0.02)), 4)
                values["kl"] = round(0.0005 + 0.004 * (1 - progress) + rng.uniform(0, 0.0005), 5)
                values["entropy"] = round(1.9 - 0.6 * progress + rng.uniform(-0.03, 0.03), 4)
                values["completions/mean_length"] = round(320 + 180 * progress + rng.uniform(-10, 10), 1)
            elif family == "preference":
                margin = 0.05 + 0.5 * (1 - math.exp(-2.0 * progress))
                values["rewards/chosen"] = round(0.1 + 0.4 * progress, 4)
                values["rewards/rejected"] = round(-0.1 - 0.3 * progress, 4)
                values["rewards/margins"] = round(margin, 4)
                values["rewards/accuracies"] = round(min(0.95, 0.5 + 0.4 * progress), 4)
            elif family == "distillation":
                values["distillation_loss"] = round(0.9 * math.exp(-2.0 * progress) + rng.uniform(-0.01, 0.01), 4)
                values["jsd"] = round(0.35 * (1 - math.exp(-2.5 * progress)), 4)
                values["entropy"] = round(1.6 - 0.4 * progress, 4)
            elif family == "reward":
                values["accuracy"] = round(min(0.95, 0.5 + 0.4 * progress), 4)
                values["margin"] = round(0.2 + 1.2 * progress, 4)
            else:
                values["entropy"] = round(1.2 - 0.3 * progress, 4)
                values["mean_token_accuracy"] = round(min(0.98, 0.55 + 0.4 * progress), 4)
            for name, value in values.items():
                self.store.append_metric(job["id"], step, name, value)
            self.store.append_event(job["id"], self._format_sim_line(step, total_steps, values))
            time.sleep(0.35)
        artifacts = sorted(
            str(p.relative_to(self.runs_dir / job["id"]))
            for p in (self.runs_dir / job["id"]).rglob("*")
            if p.is_file()
        )
        self.store.append_event(job["id"], f"[模拟模式] 训练完成，产物：{', '.join(artifacts)}", "system")
        self.store.update_job(
            job["id"], status="finished", finished_at=time.time(), exit_code=0, message="模拟运行完成（非真实训练结果）"
        )

    @staticmethod
    def _format_sim_line(step: int, total: int, values: dict) -> str:
        body = ", ".join(f"'{k}': {v}" for k, v in values.items())
        return f"{{'loss': {values.get('loss')}, 'grad_norm': 0.42, 'learning_rate': 1e-06, 'epoch': {step / total:.2f}}}  [{step}/{total}]  {body}"

    # ------------------------------------------------------------- artifacts
    def write_artifacts(self, job: dict, build: dict) -> list[str]:
        job_dir = self.runs_dir / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for name, content in (build.get("files") or {}).items():
            path = job_dir / name
            path.write_text(content, encoding="utf-8")
            written.append(name)
        (job_dir / "command.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n# 由 TRL-UI 生成，可直接复制到任意有 GPU 的机器运行\n"
            + self.builder.shell_command(job["command"])
            + "\n",
            encoding="utf-8",
        )
        written.append("command.sh")
        (job_dir / "params.json").write_text(
            json.dumps(
                {
                    "algorithm": job["algorithm"],
                    "mode": job["mode"],
                    "params": job["params"],
                    "command": job["command"],
                    "accelerate_config": build.get("accelerate_config"),
                    "num_processes": build.get("num_processes"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append("params.json")
        (job_dir / "README.md").write_text(self._job_readme(job, build), encoding="utf-8")
        written.append("README.md")
        return written

    def _job_readme(self, job: dict, build: dict) -> str:
        algo = self.catalog.algorithm(job["algorithm"])
        lines = [
            f"# {job['name']}",
            "",
            f"- 任务 id：`{job['id']}`",
            f"- 算法：**{algo['label']}**（{algo['full_name']}，{algo['status']}）",
            f"- 执行模式：{'模拟（未真实训练）' if job['mode'] == 'simulate' else '真实训练'}",
            f"- 训练器：`{algo['trainer_class']}` @ `{algo['trainer_module']}`",
            f"- 输出目录：`{build['output_dir']}`",
            "",
            "## 命令",
            "",
            "```bash",
            self.builder.shell_command(job["command"]),
            "```",
            "",
            "## 关键参数",
            "",
            "| 参数 | 值 |",
            "| --- | --- |",
        ]
        schema = {f["name"]: f for f in algo["fields"]} | {c["name"]: c for c in self.catalog.common_training_args}
        for name, value in sorted(build["params"].items()):
            field = schema.get(name, {})
            mark = "★ " if field.get("tier") == "key" else ""
            lines.append(f"| {mark}`{name}` | `{value}` |")
        env_vars = self.env_vars()
        if env_vars:
            lines += ["", "## 环境变量", "", "启动时注入任务进程（在「环境」页面设置）：", ""]
            for name, value in sorted(env_vars.items()):
                shown = "（已设置，不写入文件）" if name in envvars.SECRET_NAMES else f"`{value}`"
                lines.append(f"- `{name}` = {shown}")
            lines += ["", "换机器复现时请自行 `export` 上面这些变量（尤其是令牌，不会落在产物里）。"]
        lines += ["", "## 复现", "", "```bash", "cd $(dirname $0) && bash command.sh", "```", ""]
        return "\n".join(lines)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def file_list(job_dir: Path) -> list[dict]:
        if not job_dir.is_dir():
            return []
        out = []
        for path in sorted(job_dir.rglob("*")):
            if path.is_file():
                out.append(
                    {
                        "name": str(path.relative_to(job_dir)),
                        "size": path.stat().st_size,
                        "mtime": path.stat().st_mtime,
                    }
                )
        return out
