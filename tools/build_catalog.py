#!/usr/bin/env python3
"""Merge the extracted TRL fields with the hand-curated metadata into catalog/catalog.json.

Everything the WebUI shows comes from this file, and every entry is traceable:

  * parameter names / types / defaults / help  -> parsed from the TRL source (source_catalog.json)
  * parameter groups and their workflow stage -> docstring section headers + catalog/param_notes.json
  * algorithm metadata (family, launch, data contract, pitfalls) -> catalog/algorithms.json

The catalogue is organised by the same four planes TRL's training scripts use
(`make_parser()` takes `(ScriptArguments, <Algo>Config, ModelConfig, DatasetMixtureConfig)`):

    plane=script     dataset_* + the reward-function arguments (the algorithm's inputs)
    plane=algorithm  <Algo>Config — the algorithm's own hyper-parameters
    plane=model      ModelConfig — model id, dtype, LoRA, quantization
    plane=runtime    TrainingArguments + accelerate — parallelisation, memory, logging

Validation is strict on purpose: a typo in a curated file fails the build instead of
silently producing a UI field TRL would reject.

    python3 tools/build_catalog.py [--trl ../trl] [--check]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

STAGES = ["env", "model", "data", "signal", "rollout", "estimate", "update", "output", "deprecated"]

CHOICE_PATTERNS = [
    re.compile(r"(?:Possible|Supported) values are:?\s*(.+?)(?:\.\s|\.$)", re.S),
    re.compile(r"(?:Possible|Supported) values are\s+(.+?)(?:\.\s|\.$)", re.S),
]
TOKEN_RE = re.compile(r"[`'\"]([A-Za-z0-9_\-\.]+)[`'\"]")

TYPE_MAP = [
    (re.compile(r"^bool\b"), "toggle"),
    (re.compile(r"^int\b"), "int"),
    (re.compile(r"^float\b"), "number"),
    (re.compile(r"^list\[|^tuple"), "list"),
    (re.compile(r"^dict|^Any"), "json"),
]

# Extra fields that live outside the four config classes but are part of the launch command.
VIRTUAL_FIELDS = {
    "reward_funcs": {
        "type": "list[str]",
        "default": None,
        "help": "奖励函数列表：内置名（accuracy_reward、reasoning_accuracy_reward、think_format_reward、"
        "get_repetition_penalty_reward、get_cosine_scaled_reward、get_soft_overlong_punishment）"
        "或点分导入路径（如 my_lib.rewards.code_reward）。可以一次挂多个，按 reward_weights 加权。",
        "group": "奖励函数",
        "stage": "signal",
        "tier": "key",
        "plane": "script",
    },
    "reward_model_name_or_path": {
        "type": "str | None",
        "default": None,
        "help": "奖励模型的模型 id 或本地路径；会被当作一个额外的奖励函数挂上去。",
        "group": "奖励函数",
        "stage": "signal",
        "tier": "key",
        "plane": "script",
    },
}

SCRIPT_FIELD_META = {
    "dataset_name": {"group": "数据集", "stage": "data", "tier": "key"},
    "dataset_config": {"group": "数据集", "stage": "data", "tier": "advanced"},
    "dataset_train_split": {"group": "数据集", "stage": "data", "tier": "key"},
    "dataset_test_split": {"group": "数据集", "stage": "data", "tier": "advanced"},
    "dataset_streaming": {"group": "数据集", "stage": "data", "tier": "advanced"},
    "datasets": {"group": "数据集混合", "stage": "data", "tier": "advanced"},
    "streaming": {"group": "数据集混合", "stage": "data", "tier": "advanced"},
    "test_split_size": {"group": "数据集混合", "stage": "data", "tier": "advanced"},
}

MODEL_FIELD_META = {
    "model_name_or_path": {"group": "模型加载", "stage": "model", "tier": "key"},
    "model_revision": {"group": "模型加载", "stage": "model", "tier": "advanced"},
    "dtype": {"group": "模型加载", "stage": "model", "tier": "key"},
    "attn_implementation": {"group": "模型加载", "stage": "model", "tier": "advanced"},
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def class_index(source: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for module, mod in source["modules"].items():
        for cls in mod["classes"]:
            index.setdefault(cls["name"], {**cls, "module": module})
    return index


def resolve_fields(cls_name: str, index: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Return (fields, chain) with base-class fields first and subclass overrides replacing them."""

    chain: list[str] = []
    seen: set[str] = set()
    cursor = cls_name
    while cursor and cursor in index and cursor not in seen:
        seen.add(cursor)
        chain.append(cursor)
        bases = [b for b in index[cursor]["bases"] if b in index]
        cursor = bases[0] if bases else None
    order: list[str] = []
    by_name: dict[str, dict] = {}
    for name in reversed(chain):  # base first → subclass overrides
        for field in index[name]["fields"]:
            if field["name"] not in by_name:
                order.append(field["name"])
            by_name[field["name"]] = {**field, "declared_in": name}
    return [by_name[n] for n in order], chain


def extract_choices(help_text: str) -> list[str] | None:
    """Pull enum-ish values out of help strings such as 'Supported values are 'grpo', 'dapo''."""

    for pattern in CHOICE_PATTERNS:
        match = pattern.search(help_text)
        if not match:
            continue
        tokens = [t for t in TOKEN_RE.findall(match.group(1)) if not t.replace(".", "").isdigit()]
        tokens = [t for t in tokens if len(t) < 40]
        unique: list[str] = []
        for token in tokens:
            if token not in unique:
                unique.append(token)
        if 1 < len(unique) <= 24:
            return unique
    return None


def widget(annotation: str, choices: list[str] | None) -> str:
    base = annotation.replace(" | None", "").replace("None | ", "").strip()
    if choices:
        return "select"
    for pattern, kind in TYPE_MAP:
        if pattern.search(base):
            return kind
    return "text"


class Renderer:
    def __init__(self, index: dict[str, dict], notes: dict, key_overrides: dict[str, list[str]]):
        self.index = index
        self.notes = notes
        self.key_overrides = key_overrides

    def render(self, cls_name: str, *, plane: str, algorithm=None) -> list[dict]:
        fields, _chain = resolve_fields(cls_name, self.index)
        key_names = set(self.notes["key_field_overrides"].get(cls_name, []))
        if algorithm is not None:
            key_names |= set(algorithm.get("key_params", []))
        rendered = []
        for field in fields:
            rendered.append(self.render_field(field, cls_name=cls_name, plane=plane, key_names=key_names))
        return rendered

    def render_field(self, field: dict, *, cls_name: str, plane: str, key_names: set[str]) -> dict:
        name = field["name"]
        hidden = name in self.notes["hidden_fields"] or f"{cls_name}.{name}" in self.notes["hidden_fields"]
        override = self.notes["field_group_overrides"].get(f"{cls_name}.{name}")
        group, stage = field.get("group"), None
        if override:
            group, stage = override["group"], override["stage"]
        elif group and group in self.notes["group_stage_map"]:
            stage = self.notes["group_stage_map"][group]
        if stage is None or (stage == "deprecated" and not hidden):
            if name in self.notes["fallback_groups"]:
                fallback = self.notes["fallback_groups"][name]
                group, stage = fallback["group"], fallback["stage"]
            else:
                for rule in self.notes["fallback_group_rules"]:
                    if re.search(rule["match"], name):
                        group, stage = rule["group"], rule["stage"]
                        break
        if stage is None:
            group, stage = group or "算法其他参数", "algorithm"
        if group in {None, "", "[!NOTE]"}:
            group = "算法其他参数"
        if hidden:
            stage = "deprecated"

        choices = (
            self.notes["choices"].get(f"{cls_name}.{name}")
            or self.notes["choices"].get(f"*.{name}")
            or field.get("choices")
            or extract_choices(field["help"])
            or None
        )
        default = field["default"]
        if f"{cls_name}.{name}" in self.notes["default_overrides"]:
            default = self.notes["default_overrides"][f"{cls_name}.{name}"]
        return {
            "name": name,
            "type": field["type"],
            "default": default,
            "help": field["help"],
            "group": group,
            "stage": stage,
            "plane": plane,
            "tier": "key" if name in key_names else ("deprecated" if stage == "deprecated" else "advanced"),
            "ui": widget(field["type"], choices),
            "choices": choices,
            "declared_in": field["declared_in"],
        }


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trl", type=Path, default=root.parent / "trl")
    parser.add_argument("--source", type=Path, default=root / "catalog" / "source_catalog.json")
    parser.add_argument("--out", type=Path, default=root / "catalog" / "catalog.json")
    parser.add_argument("--check", action="store_true", help="validate only, do not write")
    args = parser.parse_args()

    source = load(args.source)
    algorithms_doc = load(root / "catalog" / "algorithms.json")
    notes = load(root / "catalog" / "param_notes.json")
    index = class_index(source)
    repo: Path = args.trl.resolve()
    renderer = Renderer(index, notes, notes["key_field_overrides"])

    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------- model plane
    model_fields = renderer.render("ModelConfig", plane="model")
    for field in model_fields:
        override = MODEL_FIELD_META.get(field["name"])
        if override:
            field.update({k: v for k, v in override.items()})
        field["stage"] = "model"  # ModelConfig only ever feeds the model stage
    for name in ("model_name_or_path", "use_peft", "lora_r", "lora_alpha", "lora_target_modules", "load_in_4bit"):
        for field in model_fields:
            if field["name"] == name and name in notes["key_field_overrides"]["ModelConfig"]:
                field["tier"] = "key"

    # ------------------------------------------------------------ script plane
    script_fields = []
    script_args_cls = next(
        (c for c in index if c == "ScriptArguments"),
        None,
    )
    if script_args_cls is None:
        errors.append("ScriptArguments 没在 TRL 源码里找到")
    else:
        for field in renderer.render(script_args_cls, plane="script"):
            meta = SCRIPT_FIELD_META.get(field["name"])
            if meta:
                field.update(meta)
            elif field["stage"] == "algorithm":
                field["stage"] = "data"
                field["group"] = "数据集"
            script_fields.append(field)
    for field_name in ("datasets", "streaming", "test_split_size"):
        for field in renderer.render("DatasetMixtureConfig", plane="script"):
            if field["name"] == field_name:
                field.update(SCRIPT_FIELD_META[field_name])
                script_fields.append(field)
    for name, spec in VIRTUAL_FIELDS.items():
        script_fields.append(
            {
                "name": name,
                "declared_in": "virtual",
                "ui": widget(spec["type"], None),
                "choices": None,
                **spec,
            }
        )

    # -------------------------------------------------------------- algorithms
    algorithms = []
    for algo in algorithms_doc["algorithms"]:
        cls_name = algo["config_class"]
        if cls_name not in index:
            errors.append(f"{algo['id']}: config class {cls_name} not found in the TRL source")
            continue
        if algo["family"] not in {f["id"] for f in algorithms_doc["families"]}:
            errors.append(f"{algo['id']}: unknown family {algo['family']}")
        for stage in algo["stages"]:
            if stage not in STAGES:
                errors.append(f"{algo['id']}: unknown stage '{stage}' in stage notes")
        if algo["trainer_class"] not in source["trainers"]:
            errors.append(f"{algo['id']}: trainer class {algo['trainer_class']} not found in the TRL source")
        if not (repo / algo["doc"]).is_file():
            errors.append(f"{algo['id']}: doc {algo['doc']} does not exist")
        launch = algo["launch"]
        if launch.get("kind") == "cli":
            if launch["cli"] not in source["cli"]["launchable"]:
                errors.append(f"{algo['id']}: '{launch['cli']}' is not a launchable TRL CLI command")
            elif not (repo / launch["script"]).is_file():
                errors.append(f"{algo['id']}: launch script {launch['script']} does not exist")

        fields = renderer.render(cls_name, plane="algorithm", algorithm=algo)
        field_names = {f["name"] for f in fields}
        for name in algo.get("key_params", []):
            if name not in field_names:
                errors.append(f"{algo['id']}: key param '{name}' is not a field of {cls_name}")

        # Which script-plane fields this algorithm actually consumes.
        own_script = [f for f in script_fields if f["name"] not in {"reward_funcs", "reward_model_name_or_path"} or algo["requires"].get("reward_funcs")]
        own = {f["name"] for f in fields} | {f["name"] for f in own_script} | {f["name"] for f in model_fields}

        algo_out = dict(algo)
        algo_out["config_chain"] = [c for c in resolve_fields(cls_name, index)[1]]
        algo_out["trainer_init_args"] = source["trainers"][algo["trainer_class"]].get("init_args", [])
        algo_out["fields"] = fields
        algo_out["script_fields"] = [f["name"] for f in own_script]
        algo_out["model_fields"] = [f["name"] for f in model_fields]
        algo_out["groups"] = [
            {"group": g, "stage": s, "fields": [f["name"] for f in fields if f["group"] == g and f["stage"] == s]}
            for g, s in dict.fromkeys((f["group"], f["stage"]) for f in fields)
        ]
        algo_out["counts"] = {
            "fields": len(fields) + len(own_script) + len(model_fields),
            "algorithm_fields": len(fields),
            "key": sum(1 for f in [*fields, *own_script, *model_fields] if f["tier"] == "key"),
            "deprecated": sum(1 for f in fields if f["tier"] == "deprecated"),
        }
        algo_out["_own"] = own
        algorithms.append(algo_out)

    # ------------------------------------------------- common training arguments
    declared = {f["name"] for cls in index.values() for f in cls["fields"]}
    common = []
    for item in notes["common_training_args"]:
        evidence = repo / item["evidence"]
        if not evidence.is_file():
            errors.append(f"common arg {item['name']}: evidence file {item['evidence']} missing")
        elif item["name"] not in evidence.read_text(encoding="utf-8", errors="ignore"):
            errors.append(f"common arg {item['name']}: not mentioned in {item['evidence']}")
        if item["name"] in declared:
            warnings.append(f"common arg {item['name']}: also declared by a TRL config (deduped per algorithm)")
        if item["stage"] not in STAGES:
            errors.append(f"common arg {item['name']}: bad stage {item['stage']}")
        common.append({**item, "tier": "key", "ui": widget(item["type"], None), "choices": None, "plane": "runtime"})

    # --------------------------------------------------------- per algorithm wiring
    for algo in algorithms:
        own = algo.pop("_own")
        algo["common_training_args"] = [c["name"] for c in common if c["name"] not in own]

    # --------------------------------------------------------------------- presets
    presets = {}
    for algo_id, preset in notes["presets"].items():
        algo = next((a for a in algorithms if a["id"] == algo_id), None)
        if algo is None:
            errors.append(f"preset {algo_id}: no such algorithm")
            continue
        known = (
            {f["name"] for f in algo["fields"]}
            | set(algo["script_fields"])
            | set(algo["model_fields"])
            | {c["name"] for c in common}
        )
        params = {}
        for section in ("training", "algorithm", "model", "script"):
            for name, value in preset.get(section, {}).items():
                if name not in known:
                    errors.append(f"preset {algo_id}.{section}: unknown parameter '{name}'")
                params[name] = value
        presets[algo_id] = {"note": preset.get("note", ""), "params": params}

    # Every suggested dataset must actually be mentioned in that algorithm's doc page.
    dataset_examples: dict[str, list[str]] = {}
    for algo_id, examples in notes.get("dataset_examples", {}).items():
        algo = next((a for a in algorithms if a["id"] == algo_id), None)
        if algo is None:
            errors.append(f"dataset_examples {algo_id}: no such algorithm")
            continue
        doc_text = (repo / algo["doc"]).read_text(encoding="utf-8", errors="ignore")
        kept = []
        for name in examples:
            if name not in doc_text:
                errors.append(f"dataset_examples {algo_id}: '{name}' 不在 {algo['doc']} 里")
                continue
            kept.append(name)
        if kept:
            dataset_examples[algo_id] = kept

    if errors:
        print("catalog build FAILED:", file=sys.stderr)
        for err in errors:
            print("  -", err, file=sys.stderr)
        return 1

    catalog = {
        "meta": {
            "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trl_version": source["source"]["version"],
            "trl_commit": source["source"]["commit"],
            "trl_path": source["source"]["path"],
            "built_by": "tools/build_catalog.py",
            "source_files": [
                "catalog/source_catalog.json",
                "catalog/algorithms.json",
                "catalog/param_notes.json",
            ],
        },
        "stages": algorithms_doc["stages"],
        "families": algorithms_doc["families"],
        "planes": [
            {"id": "script", "label": "任务平面", "summary": "ScriptArguments + DatasetMixtureConfig：数据来源与切分、奖励函数。"},
            {"id": "algorithm", "label": "算法平面", "summary": "<Algo>Config：采样、优势、损失等算法超参。"},
            {"id": "model", "label": "模型平面", "summary": "ModelConfig：模型 id、精度、LoRA 与量化。"},
            {"id": "runtime", "label": "运行平面", "summary": "TrainingArguments + accelerate：并行、显存、日志、checkpoint。"},
        ],
        "algorithms": algorithms,
        "dataset_examples": dataset_examples,
        "script_fields": script_fields,
        "model_fields": model_fields,
        "common_training_args": common,
        "presets": presets,
        "modules": {
            name: {"path": mod["path"], "classes": [c["name"] for c in mod["classes"]]}
            for name, mod in source["modules"].items()
        },
    }

    uniq_warnings = sorted(set(warnings))
    if args.check:
        print(f"catalog check OK ({len(algorithms)} algorithms, {len(common)} common args)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = sum(a["counts"]["fields"] for a in algorithms)
    print(f"catalog: {len(algorithms)} algorithms, {total} fields per-algorithm (with planes), {len(common)} common args -> {args.out}")
    if uniq_warnings:
        print(f"{len(uniq_warnings)} notes:")
        for warning in uniq_warnings[:12]:
            print("   -", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
