#!/usr/bin/env python3
"""Extract the TRL trainer catalogue straight out of the TRL source tree.

The script never imports TRL (it must run on a machine without torch/transformers):
it parses the trainer config modules with `ast` and reads

  * every dataclass field  -> name, annotation, default, metadata["help"]
  * the class docstring    -> per-field description and the `> Parameters that ...`
                              section headers, which become the UI groups.

Usage:
    python3 tools/extract_source_catalog.py [--trl ../trl] [--out catalog/source_catalog.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

# Modules that carry trainer configurations (trainer configs, script arguments, model config).
CONFIG_GLOBS = [
    "trl/trainer/*_config.py",
    "trl/trainer/model_config.py",
    "trl/experimental/*/*_config.py",
    "trl/scripts/*.py",
    "trl/scripts/utils.py",
]


def _unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - very defensive
        return None


def _literal(node: ast.AST | None, default: str | None = None) -> tuple[str | None, bool]:
    """Return (repr_of_value, is_literal)."""

    if node is None:
        return default, True
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None, False
    if isinstance(value, str):
        return value, True
    return repr(value), True


def parse_docstring(doc: str | None) -> tuple[dict[str, dict], list[str], str]:
    """Parse a numpydoc-ish TRL docstring.

    Returns (fields, group_order, summary) where `fields[name] = {"desc":..., "group":...}`.
    Section headers look like `> Parameters that control generation`; entries look like
    `name (`type`, *optional*, defaults to `x`):` followed by an indented description.
    """

    if not doc:
        return {}, [], ""

    lines = doc.expandtabs(4).splitlines()
    # The summary is everything before the first empty line after stripping indentation.
    summary_lines: list[str] = []
    for line in lines:
        if line.strip() == "":
            if summary_lines:
                break
            continue
        summary_lines.append(line.strip())
    summary = " ".join(summary_lines)

    # Locate the parameter section ("Parameters:" or "Args:").
    start = None
    for i, line in enumerate(lines):
        if line.strip() in {"Parameters:", "Args:", "Arguments:", "Keyword Arguments:"}:
            start = i + 1
            break
    if start is None:
        return {}, [], summary

    body = lines[start:]
    # Drop trailing sections (Returns:, Examples: ...) that are not indented deeper.
    base_indent = min((len(l) - len(l.lstrip()) for l in body if l.strip()), default=0)
    fields: dict[str, dict] = {}
    groups: list[str] = []
    current_group: str | None = None
    current_name: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current_name, buf
        if current_name is not None:
            desc = " ".join(" ".join(buf).split())
            fields[current_name] = {"desc": desc, "group": current_group}
        current_name, buf = None, []

    entry_re = re.compile(r"^(\w+)\s*\((.+?)\)\s*:?\s*$")
    for raw in body:
        line = raw.rstrip()
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if not stripped:
            if current_name is not None:
                buf.append("")
            continue
        if indent <= base_indent and stripped.endswith(":") and not entry_re.match(stripped):
            # A new top-level section (Returns:, Examples:, ...) ends the parameter block.
            if stripped not in {"Parameters:", "Args:"}:
                break
        if stripped.startswith(">"):
            flush()
            current_group = stripped.lstrip(">").strip().rstrip(":")
            if current_group not in groups:
                groups.append(current_group)
            continue
        m = entry_re.match(stripped)
        if m and indent <= base_indent + 4:
            flush()
            current_name = m.group(1)
            continue
        if current_name is not None:
            buf.append(stripped)
    flush()
    return fields, groups, summary


def _field_default(call: ast.Call) -> tuple[str | None, str | None, list[str] | None]:
    """Read `field(default=..., default_factory=..., metadata={"help": ..., "choices": [...]})`."""

    default: str | None = None
    help_text: str | None = None
    choices: list[str] | None = None
    for kw in call.keywords:
        if kw.arg == "default":
            default, ok = _literal(kw.value)
            if not ok:
                default = _unparse(kw.value)
        elif kw.arg == "default_factory":
            default = f"<{_unparse(kw.value)}>"
        elif kw.arg == "metadata" and isinstance(kw.value, ast.Dict):
            for k, v in zip(kw.value.keys, kw.value.values):
                if not isinstance(k, ast.Constant):
                    continue
                if k.value == "help":
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        help_text = " ".join(v.value.split())
                    else:
                        help_text = _unparse(v)
                elif k.value == "choices":
                    try:
                        value = ast.literal_eval(v)
                    except Exception:
                        continue
                    if isinstance(value, list):
                        choices = [str(item) for item in value]
    return default, help_text, choices


def parse_module(path: Path, repo: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = {_unparse(d) for d in node.decorator_list}
        is_dataclass = any(d and d.startswith("dataclass") for d in decorators)
        if not is_dataclass and not node.name.endswith(("Config", "Arguments")):
            continue
        doc_fields, groups, summary = parse_docstring(ast.get_docstring(node, clean=False))
        fields = []
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            name = stmt.target.id
            annotation = _unparse(stmt.annotation) or "Any"
            help_text = None
            choices = None
            if stmt.value is None:
                default = None
            elif isinstance(stmt.value, ast.Call) and getattr(stmt.value.func, "id", "") == "field":
                default, help_text, choices = _field_default(stmt.value)
            else:
                default, ok = _literal(stmt.value)
                if not ok:
                    default = _unparse(stmt.value)
            doc = doc_fields.get(name, {})
            fields.append(
                {
                    "name": name,
                    "type": annotation,
                    "default": default,
                    "help": help_text or doc.get("desc") or "",
                    "help_source": "metadata" if help_text else ("docstring" if doc.get("desc") else "none"),
                    "choices": choices,
                    "group": doc.get("group"),
                }
            )
        classes.append(
            {
                "name": node.name,
                "bases": [_unparse(b) for b in node.bases],
                "summary": summary,
                "groups": groups,
                "fields": fields,
                "lineno": node.lineno,
            }
        )
    return {"path": str(path.relative_to(repo)), "classes": classes}


def parse_trainer_meta(path: Path) -> dict:
    """Pull the trainer class name + a few structural facts (methods, constructor args, docstring summary)."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("Trainer"):
            methods = [
                s.name
                for s in node.body
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and not s.name.startswith("__")
            ]
            init = next((s for s in node.body if isinstance(s, ast.FunctionDef) and s.name == "__init__"), None)
            init_args: list[str] = []
            if init is not None:
                init_args = [a.arg for a in init.args.args if a.arg != "self"]
                init_args += [f"*{init.args.vararg.arg}"] if init.args.vararg else []
                init_args += [a.arg for a in init.args.kwonlyargs]
            return {
                "name": node.name,
                "file": path.name,
                "methods": methods,
                "init_args": init_args,
                "bases": [_unparse(b) for b in node.bases if b],
                "summary": " ".join((ast.get_docstring(node) or "").split())[:600],
            }
    return {}


def git_info(repo: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=20, check=False
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "date": run("log", "-1", "--format=%ad"),
        "subject": run("log", "-1", "--format=%s"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--trl", type=Path, default=here.parent.parent / "trl", help="path to the TRL checkout")
    parser.add_argument("--out", type=Path, default=here.parent / "catalog" / "source_catalog.json")
    args = parser.parse_args()

    repo: Path = args.trl.resolve()
    if not (repo / "trl").is_dir():
        raise SystemExit(f"not a TRL checkout: {repo}")

    modules = {}
    for pattern in CONFIG_GLOBS:
        for path in sorted(repo.glob(pattern)):
            parsed = parse_module(path, repo)
            if parsed["classes"]:
                modules[parsed["path"]] = parsed

    trainers = {}
    for path in sorted(list(repo.glob("trl/trainer/*_trainer.py")) + list(repo.glob("trl/experimental/*/*_trainer.py"))):
        meta = parse_trainer_meta(path)
        if meta:
            trainers[meta["name"]] = {**meta, "path": str(path.relative_to(repo))}

    scripts = {}
    for path in sorted(repo.glob("trl/scripts/*.py")):
        text = path.read_text(encoding="utf-8")
        if "def make_parser(" not in text:
            continue
        m = re.search(r"dataclass_types\s*=\s*\(([^)]*)\)", text, re.S)
        scripts[path.stem] = {
            "path": str(path.relative_to(repo)),
            "dataclass_types": [t.strip() for t in m.group(1).split(",")] if m else [],
        }

    cli = {}
    cli_init = repo / "trl/cli/commands/__init__.py"
    if cli_init.is_file():
        cli_source = cli_init.read_text(encoding="utf-8")
        cli["launchable"] = re.findall(r'TrainingCommand\("([a-z_]+)"\)', cli_source)
        cli["entry_point"] = "trl = \"trl.cli:main\"" in (repo / "pyproject.toml").read_text(encoding="utf-8")

    version = (repo / "VERSION").read_text(encoding="utf-8").strip() if (repo / "VERSION").is_file() else ""

    out = {
        "source": {
            "path": str(repo),
            "version": version,
            "commit": git_info(repo),
            "extractor": "tools/extract_source_catalog.py",
        },
        "cli": cli,
        "scripts": scripts,
        "modules": modules,
        "trainers": trainers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_fields = sum(len(c["fields"]) for m in modules.values() for c in m["classes"])
    print(
        f"TRL {version} @{out['source']['commit']['commit']}: "
        f"{len(modules)} modules, {n_fields} fields, {len(trainers)} trainers, "
        f"{len(scripts)} scripts -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
