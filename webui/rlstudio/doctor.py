"""Environment probe: what can this machine actually run?

TRL needs torch + transformers + datasets (and accelerate for the CLI launcher), so the
probe answers three questions the UI cares about:

  1. can we launch a real training run here?  -> `can_train`
  2. how many accelerators and how much memory? -> `devices`
  3. which TRL checkout / CLI do we point at?  -> `trl`

The probing of torch runs in a subprocess (a broken CUDA install must not take the web
server down with it) with a hard timeout.

The user's environment variables (the 「环境」 page: HF_HOME / HF_ENDPOINT / HF_TOKEN /
CUDA_VISIBLE_DEVICES) are part of the probe: they are handed to the subprocess, so
CUDA_VISIBLE_DEVICES changes the device list this report shows — the report has to describe
the machine the *jobs* will actually see, not the one the server inherited.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROBE = r"""
import json, sys
info = {"devices": [], "dtype_hints": []}
try:
    import torch
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        info["backend"] = "cuda"
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i, "name": props.name,
                "memory_gb": round(props.total_memory / 1024**3, 1),
                "capability": f"{props.major}.{props.minor}",
            })
        info["dtype_hints"] = ["bfloat16"] if torch.cuda.get_device_capability(0)[0] >= 8 else ["float16"]
    elif hasattr(torch, "npu") and torch.npu.is_available():
        info["backend"] = "npu"
        for i in range(torch.npu.device_count()):
            info["devices"].append({"index": i, "name": torch.npu.get_device_name(i), "memory_gb": None,
                                    "capability": None})
        info["dtype_hints"] = ["bfloat16"]
    else:
        info["backend"] = "cpu"
except Exception as exc:  # torch missing or broken
    info["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(info))
"""

PACKAGES = [
    ("torch", "训练框架", True),
    ("transformers", "模型与 Trainer 基类", True),
    ("datasets", "数据集加载", True),
    ("accelerate", "trl CLI 的启动器", True),
    ("peft", "LoRA/QLoRA", False),
    ("trl", "本项目的算法来源", True),
    ("vllm", "生成加速（use_vllm）", False),
    ("deepspeed", "ZeRO 分片", False),
    ("bitsandbytes", "4/8bit 量化", False),
    ("liger_kernel", "Liger 内核", False),
    ("trackio", "训练日志后端", False),
    ("wandb", "训练日志后端", False),
]


def _probe_torch(env: dict[str, str], timeout: int = 25) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=timeout, check=False, env=env
        )
    except subprocess.TimeoutExpired:
        return {"error": f"probe timed out after {timeout}s"}
    if proc.returncode != 0:
        return {"error": (proc.stderr or "probe failed").strip().splitlines()[-1][:300]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "probe produced no JSON"}


def _package_version(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    try:  # importlib.metadata is cheap; no import of the package itself
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except PackageNotFoundError:
        return "已安装"
    except Exception:
        return "已安装"


def run_doctor(trl_path: Path, runs_dir: Path, env_vars: dict[str, str] | None = None) -> dict:
    """Probe the machine as the training jobs will see it.

    `env_vars` are the user-set variables from the 「环境」 page (see `envvars`); they win
    over whatever the server process inherited, exactly like they do for a real job.
    """

    env_vars = {name: value for name, value in (env_vars or {}).items() if value}
    process_env = {**os.environ, **env_vars}

    packages = []
    for name, purpose, essential in PACKAGES:
        version = _package_version(name)
        packages.append(
            {"name": name, "version": version, "installed": version is not None, "essential": essential, "purpose": purpose}
        )
    installed = {p["name"] for p in packages if p["installed"]}

    torch_info = _probe_torch(process_env)
    devices = torch_info.get("devices", [])
    backend = torch_info.get("backend", "unknown")

    checks: list[dict] = []

    def check(level: str, title: str, detail: str, hint: str = "") -> None:
        checks.append({"level": level, "title": title, "detail": detail, "hint": hint})

    missing_essential = [p["name"] for p in packages if p["essential"] and not p["installed"]]
    if missing_essential:
        check(
            "fail",
            "缺少训练所需的核心依赖",
            "、".join(missing_essential),
            "pip install trl 会带上 torch/transformers/datasets/accelerate；缺 vLLM 时把 use_vllm 关掉。",
        )
    else:
        check("ok", "训练依赖齐备", f"trl {_package_version('trl')} / transformers {_package_version('transformers')}")

    if torch_info.get("error"):
        check("fail", "torch 探测失败", torch_info["error"], "先在命令行确认 `python -c \"import torch\"` 能跑通。")
    elif backend == "cpu":
        check("warn", "没有可用的加速器", "torch 只看到 CPU", "CPU 上可以跑通小模型冒烟，但正式 RL 训练需要 GPU/NPU。")
    elif backend == "unknown":
        check("fail", "无法探测加速器", torch_info.get("torch_error", "torch 不可用"), "装好与你的驱动匹配的 torch 后再刷新。")
    else:
        total = sum(d["memory_gb"] or 0 for d in devices)
        detail = f"合计显存约 {total:.1f} GB" if total else "显存未知"
        visible = process_env.get("CUDA_VISIBLE_DEVICES")
        check(
            "ok",
            f"检测到 {len(devices)} 张 {backend.upper()} 设备",
            f"{detail}（CUDA_VISIBLE_DEVICES={visible}）" if visible and backend == "cuda" else detail,
        )

    cli = shutil.which("trl")
    if cli:
        check("ok", "找到 trl 命令行", cli)
    else:
        check(
            "warn",
            "PATH 里没有 trl 命令",
            "会退化为 `python -m trl.cli.main`",
            "`pip install trl` 或把 trl 的 bin 目录加进 PATH。",
        )

    if (trl_path / "trl" / "trainer" / "grpo_config.py").is_file():
        check("ok", "TRL 源码目录可用", str(trl_path))
    else:
        check("fail", "没有找到 TRL 源码目录", str(trl_path), "用 --trl 指向 TRL checkout。")

    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        probe_file = runs_dir / ".write-probe"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink()
        usage = shutil.disk_usage(runs_dir)
        check("ok", "运行目录可写", f"{runs_dir}（剩余 {usage.free / 1024**3:.1f} GB）")
        if usage.free < 20 * 1024**3:
            check("warn", "磁盘空间偏紧", f"剩余 {usage.free / 1024**3:.1f} GB", "checkpoint 会很快吃满磁盘，建议设 save_total_limit。")
    except OSError as exc:
        check("fail", "运行目录不可写", f"{runs_dir}: {exc}", "换一个可写路径或修正权限。")

    # The user-set variables (「环境」页) that change how models and datasets get downloaded.
    hf_home = process_env.get("HF_HOME")
    if hf_home:
        home = Path(hf_home).expanduser()
        if home.is_dir():
            check("ok", "HF_HOME 已生效", f"模型与数据集缓存在 {home}")
        else:
            check(
                "warn",
                "HF_HOME 指向的目录还不存在",
                str(home),
                "首次下载时 transformers 会自动创建，确认父目录存在且可写即可。",
            )
    hf_endpoint = process_env.get("HF_ENDPOINT")
    if hf_endpoint:
        check("ok", "HF 镜像已启用", f"下载走 {hf_endpoint}（HF_ENDPOINT）")
    if process_env.get("HF_TOKEN"):
        check("ok", "HF_TOKEN 已设置", "gated 模型与私有数据集可以用它下载（值不会显示在界面与日志里）")

    can_train = not missing_essential and not torch_info.get("error") and backend != "cpu" and bool(devices)
    mode = "real" if can_train else "simulate"
    if mode == "simulate":
        check(
            "warn",
            "当前环境只能演示，不能真训",
            "默认以「模拟模式」提交任务：命令与产物照常生成，日志与曲线由内置模拟器产生",
            "把任务模式改成 real 可以看到真实的报错；真训请在有 GPU 且装好 trl 的机器上运行。",
        )

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
        "backend": backend,
        "devices": devices,
        "dtype_hints": torch_info.get("dtype_hints", []),
        "packages": packages,
        "installed": sorted(installed),
        "trl": {"path": str(trl_path), "cli": cli},
        "runs_dir": str(runs_dir),
        "env_vars": sorted(env_vars),  # names only — the values live behind /api/env-vars
        "can_train": can_train,
        "recommended_mode": mode,
        "checks": checks,
    }
