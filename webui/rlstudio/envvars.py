"""The four environment variables every Hugging Face / GPU workflow ends up setting, as data.

They are edited on the 「环境」 page and injected into every **real** training subprocess
(see `runner.Runner._run_real`) and into the probe subprocess of `doctor`, so three rules
build the whole module:

* the names are a **whitelist** — a request may only ever name a variable listed in
  `FIELDS`, otherwise "set an env var" would be a way to put `LD_PRELOAD` on the training
  process;
* each name has its own validation, run before anything is stored;
* `HF_TOKEN` is a secret: stored and injected like the others, but never sent back to the
  browser — the UI only sees `preview()`, and logs/artifacts only ever say "已设置".

Storage is the ``settings`` table of the job database (`store.Store`), so the values
survive a restart without a second config file to keep in sync.
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping

FIELDS: tuple[dict[str, Any], ...] = (
    {
        "name": "HF_HOME",
        "label": "模型缓存目录",
        "purpose": "transformers / datasets 下载的权重与数据都落在这里。指向大盘或共享目录，多台机器就能复用同一份缓存。",
        "placeholder": "/data/hf-cache 或 D:\\hf-cache",
        "secret": False,
    },
    {
        "name": "HF_ENDPOINT",
        "label": "Hugging Face 镜像",
        "purpose": "连不上 huggingface.co 或下载太慢时换成镜像，例如 https://hf-mirror.com；留空表示用官方站点。",
        "placeholder": "https://hf-mirror.com",
        "secret": False,
    },
    {
        "name": "HF_TOKEN",
        "label": "访问令牌",
        "purpose": "下载 gated 模型（Llama 等）或私有数据集时使用。只注入训练进程，界面回显、任务日志与产物里都不会出现它的值。",
        "placeholder": "hf_…",
        "secret": True,
    },
    {
        "name": "CUDA_VISIBLE_DEVICES",
        "label": "可见显卡",
        "purpose": "限定探测与训练能看到的 GPU，例如 0 或 0,1；留空表示全部可见。改完重新探测可看到上方设备列表随之变化。",
        "placeholder": "0,1",
        "secret": False,
    },
)

NAMES: tuple[str, ...] = tuple(field["name"] for field in FIELDS)
SECRET_NAMES = frozenset(field["name"] for field in FIELDS if field["secret"])
MAX_LENGTH = 1024

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_URL_RE = re.compile(r"^https?://[^\s/]+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s")
# CUDA_VISIBLE_DEVICES accepts indices ("0,1"), MIG names and GPU UUIDs, never arbitrary text.
_DEVICE_LIST_RE = re.compile(r"^[A-Za-z0-9_.\-]+(\s*,\s*[A-Za-z0-9_.\-]+)*$")


def validate(name: str, value: str) -> str | None:
    """Return the reason a value is unusable, or None."""

    if _CONTROL_RE.search(value):
        return "不能包含换行或控制字符"
    if len(value) > MAX_LENGTH:
        return f"长度不能超过 {MAX_LENGTH} 个字符"
    if name == "HF_ENDPOINT" and not _URL_RE.match(value):
        return "必须是 http:// 或 https:// 开头的地址"
    if name == "CUDA_VISIBLE_DEVICES" and not _DEVICE_LIST_RE.match(value):
        return "只能是逗号分隔的 GPU 序号或 UUID，例如 0 或 0,1"
    if name == "HF_TOKEN" and _WHITESPACE_RE.search(value):
        return "令牌不能包含空格"
    return None


def preview(value: str) -> str:
    """A reminder of *which* secret is stored, never enough to use it."""

    return f"{value[:4]}…{value[-4:]}" if len(value) >= 12 else "•" * 8


def load(store) -> dict[str, str]:
    """The variables the user has set, ignoring anything else in the settings table."""

    stored = store.get_settings()
    return {name: stored[name] for name in NAMES if stored.get(name)}


def save(store, values: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Persist a submitted subset of the four variables.

    A key with an empty (or null) value removes that variable — that is how the UI clears a
    field, and how the token is deleted without ever echoing it back. Returns the stored
    value set plus per-field errors; nothing is written when any value is rejected.
    """

    accepted: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name, raw in values.items():
        if name not in NAMES:
            raise ValueError(f"不支持的环境变量 {name!r}，只能设置 {'、'.join(NAMES)}")
        text = "" if raw is None else str(raw).strip()
        if text:
            problem = validate(name, text)
            if problem:
                errors[name] = problem
                continue
        accepted[name] = text
    if errors:
        return load(store), errors
    store.set_settings(accepted)
    return load(store), {}


def describe(stored: Mapping[str, str]) -> list[dict]:
    """Everything the UI needs per field: current value, preview, and what the process had."""

    fields = []
    for spec in FIELDS:
        name = spec["name"]
        value = stored.get(name) or ""
        inherited = os.environ.get(name) or ""
        fields.append(
            {
                "name": name,
                "label": spec["label"],
                "purpose": spec["purpose"],
                "placeholder": spec["placeholder"],
                "secret": spec["secret"],
                "value": "" if spec["secret"] else value,
                "preview": preview(value) if spec["secret"] and value else None,
                "set": bool(value),
                "source": "user" if value else ("process" if inherited else None),
                "process_value": preview(inherited) if (spec["secret"] and inherited) else inherited,
            }
        )
    return fields


def summarize(values: Mapping[str, str]) -> str:
    """One line for a console banner or a job log line: values, secrets redacted."""

    if not values:
        return "（未设置）"
    return ", ".join(
        f"{name}={'(已设置)' if name in SECRET_NAMES else value}" for name, value in sorted(values.items())
    )
