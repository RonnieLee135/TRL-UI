#!/usr/bin/env python3
"""Render docs/02-algorithm-params.md from catalog/catalog.json.

The document is generated, never hand-edited: it is the readable view of the same
catalog the WebUI serves, so the two can never drift apart.

    python3 tools/gen_params_doc.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGE_LABEL = {
    "env": "环境与资源",
    "model": "模型与参考/教师",
    "data": "数据与提示",
    "signal": "监督信号",
    "rollout": "采样",
    "estimate": "优势/估计",
    "update": "损失与更新",
    "output": "记录与产物",
    "deprecated": "已弃用",
}


def short(text: str, limit: int = 240) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=root / "catalog" / "catalog.json")
    parser.add_argument("--out", type=Path, default=root / "docs" / "02-algorithm-params.md")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    meta = catalog["meta"]
    families = {f["id"]: f for f in catalog["families"]}

    out: list[str] = []
    out.append("# TRL 算法入参总表（自动生成）\n")
    out.append(
        "> 本文件由 `tools/gen_params_doc.py` 从 `catalog/catalog.json` 生成，"
        "后者又由 `tools/extract_source_catalog.py` 直接解析 TRL 源码得到，**请勿手工编辑**。\n"
    )
    out.append(
        f"- TRL 版本：`{meta['trl_version']}`（commit `{meta['trl_commit']['commit']}`，{meta['trl_commit']['date']}）\n"
        f"- 生成时间：{meta['generated_at']}\n"
        f"- 算法数：{len(catalog['algorithms'])}，参数条目：{sum(a['counts']['fields'] for a in catalog['algorithms'])}，"
        f"通用训练参数：{len(catalog['common_training_args'])}\n"
    )
    out.append(
        "\n每个参数的取值来源：类型/默认值来自 `field(...)` 声明，说明来自 `metadata['help']` 或类 docstring 的 "
        "`Parameters:` 段落，分组标题直接采用 docstring 里的 `> Parameters that control ...`。"
        "`stage` 列是这些分组映射到[训练流程抽象](03-training-workflow.md)的阶段。\n"
    )

    out.append("\n## 算法总览\n")
    out.append("| 算法 | 全称 | 家族 | 状态 | 配置类 | 参数数（关键/弃用） | 启动方式 |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    for algo in catalog["algorithms"]:
        launch = algo["launch"]
        how = f"`trl {launch['cli']}`" if launch.get("kind") == "cli" else "Python API（实验）"
        out.append(
            f"| [{algo['label']}](#{algo['id']}) | {algo['full_name']} | {families[algo['family']]['label']} | "
            f"{'稳定' if algo['status'] == 'stable' else '实验'} | `{algo['config_class']}` | "
            f"{algo['counts']['fields']}（{algo['counts']['key']}/{algo['counts']['deprecated']}） | {how} |"
        )

    out.append("\n## 通用训练参数\n")
    out.append(
        "这些是 `transformers.TrainingArguments` 的字段，TRL 自己不重新声明，但对任何算法都生效；"
        "WebUI 把它们归到「训练资源」面板。留空即不传给训练脚本，使用框架默认值。\n"
    )
    out.append("| 参数 | 类型 | 阶段 | 默认 | 证据 | 说明 |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for item in catalog["common_training_args"]:
        default = "—" if item["default"] is None else f"`{item['default']}`"
        out.append(
            f"| `{item['name']}` | {item['type']} | {STAGE_LABEL[item['stage']]} | {default} | "
            f"`{item['evidence']}` | {short(item['help'])} |"
        )

    for algo in catalog["algorithms"]:
        out.append(f"\n## <a id=\"{algo['id']}\"></a>{algo['label']}（`{algo['id']}`）\n")
        family = families[algo["family"]]
        out.append(f"- **全称**：{algo['full_name']}（{family['label']}，{'稳定' if algo['status'] == 'stable' else '实验'}）")
        out.append(f"- **配置类**：`{algo['config_class']}` @ `{algo['config_module']}`（继承链：{' → '.join('`' + c + '`' for c in algo['config_chain'])}）")
        out.append(f"- **训练器**：`{algo['trainer_class']}` @ `{algo['trainer_module']}`")
        out.append(f"- **文档**：`{algo['doc']}`" + (f" · 论文：<https://huggingface.co/papers/{algo['paper']}>" if algo.get("paper") else ""))
        out.append(
            f"- **启动**：`{algo['launch'].get('cli') and 'trl ' + algo['launch']['cli'] or algo['launch'].get('example', 'Python API')}`"
        )
        if algo.get("script_args"):
            out.append(f"- **脚本参数**：{', '.join('`' + s + '`' for s in algo['script_args'])}")
        out.append(f"- **数据契约**：{'、'.join(algo['data']['roles'])}；{algo['data']['format']}")
        out.append(f"\n> {algo['summary']}\n")

        out.append("\n### 训练流程映射\n")
        out.append("| 阶段 | 该算法在此阶段做什么 |")
        out.append("| --- | --- |")
        for stage in catalog["stages"]:
            if stage["id"] in algo["stages"]:
                out.append(f"| {stage['label']} | {short(algo['stages'][stage['id']], 400)} |")

        out.append("\n### 参数\n")
        last = None
        for field in algo["fields"]:
            key = (field["stage"], field["group"])
            if key != last:
                last = key
                out.append(f"\n**{STAGE_LABEL.get(field['stage'], field['stage'])} · {field['group']}**\n")
                out.append("| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |")
                out.append("| --- | --- | --- | --- | --- | --- |")
            default = "—" if field["default"] is None else f"`{field['default']}`"
            choices = ", ".join(f"`{c}`" for c in field["choices"]) if field["choices"] else "—"
            tier = "★" if field["tier"] == "key" else ("弃用" if field["tier"] == "deprecated" else "")
            out.append(
                f"| `{field['name']}` | {field['type']} | {default} | {tier} | {choices} | {short(field['help'])} |"
            )

        if algo.get("variants"):
            out.append("\n### 同一配置内的算法变体\n")
            for param, values in algo["variants"].items():
                out.append(f"- `{param}`：")
                for value in values:
                    out.append(f"  - {value}")
        if algo.get("pros") or algo.get("cons") or algo.get("pitfalls"):
            out.append("\n### 取舍与坑\n")
            for item in algo.get("pros", []):
                out.append(f"- ✅ {item}")
            for item in algo.get("cons", []):
                out.append(f"- ⚠️ {item}")
            for item in algo.get("pitfalls", []):
                out.append(f"- 🔻 {item}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"docs: {args.out} ({len('\n'.join(out).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
