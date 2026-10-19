#!/usr/bin/env python3
"""TRL-UI — a WebUI for training RL models with TRL.

Standard library only: no pip install, no npm, no CDN, no build step.

    python3 webui/server.py                      # http://127.0.0.1:8770
    python3 webui/server.py --port 8899 --trl ../trl

The app reads the analysis catalogue produced by tools/build_catalog.py, lets you fill a
form per algorithm, writes the equivalent `trl <algo> …` command (plus a runnable script
for the experimental algorithms) into webui/runs/<job>/ and can launch or simulate it.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from rlstudio import envvars  # noqa: E402
from rlstudio.api import App, serve  # noqa: E402


def free_port(host: str, preferred: int) -> int:
    """Never fight over a port: if it is taken, walk up until one is free."""

    for candidate in range(preferred, preferred + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise SystemExit(f"{host} 上 {preferred}-{preferred + 39} 都被占用，请用 --port 指定其它端口")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770, help="起始端口（被占用时自动后移）")
    parser.add_argument("--trl", type=Path, default=ROOT.parent / "trl", help="TRL 源码目录")
    parser.add_argument("--catalog", type=Path, default=ROOT / "catalog" / "catalog.json")
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--db", type=Path, default=HERE / "runs" / "studio.db")
    parser.add_argument("--no-port-shift", action="store_true", help="端口被占用时直接失败")
    args = parser.parse_args()

    if not args.catalog.is_file():
        print(f"找不到算法目录 {args.catalog}，先运行：python3 tools/build_catalog.py --trl {args.trl}", file=sys.stderr)
        return 2

    port = args.port if args.no_port_shift else free_port(args.host, args.port)
    app = App(
        catalog_path=args.catalog,
        trl_path=args.trl,
        runs_dir=args.runs,
        db_path=args.db,
        static_dir=HERE / "static",
    )
    doctor = app.doctor()
    httpd = serve(app, args.host, port)
    meta = app.catalog.meta
    print(f"TRL-UI  →  http://{args.host}:{port}/")
    print(f"  TRL        : {meta['trl_version']} @{meta['trl_commit']['commit']}  ({args.trl})")
    print(f"  算法目录   : {args.catalog}（{len(app.catalog.algorithms)} 个算法）")
    print(f"  任务目录   : {app.runs_dir}")
    print(f"  运行模式   : 推荐 {doctor['recommended_mode']}（可训练={doctor['can_train']}，后端={doctor['backend']}，设备={len(doctor['devices'])}）")
    print(f"  环境变量   : {envvars.summarize(app.env_vars())}（在「环境」页面修改）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
