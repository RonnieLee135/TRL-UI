"""HTTP API + static file serving, standard library only (`http.server`).

Routes
------
GET  /                              the single page app
GET  /static/<file>                 app assets (no CDN, no build step)
GET  /api/catalog                   algorithms / families / stages / presets (slim)
GET  /api/algorithms/<id>           parameter schema for one algorithm (the wizard's form)
GET  /api/doctor                    environment report
GET  /api/env-vars                  user-set environment variables (HF_HOME / HF_ENDPOINT / HF_TOKEN / CUDA_VISIBLE_DEVICES)
POST /api/env-vars                  save them (an empty value removes a variable)
POST /api/validate                  validate a parameter set without creating a job
GET  /api/jobs                      job list
POST /api/jobs                      create a job (writes runs/<id>/… and returns the command)
GET  /api/jobs/<id>                 job detail + artifacts
POST /api/jobs/<id>/start|stop      control
POST /api/jobs/<id>/delete          remove the record (never touches a running job)
GET  /api/jobs/<id>/logs?since=N    JSON log tail (fallback for the SSE stream)
GET  /api/jobs/<id>/metrics         metric series
GET  /api/jobs/<id>/files/<name>    download a generated artifact
GET  /api/jobs/<id>/events          Server-Sent Events: log lines, metrics, status
"""

from __future__ import annotations

import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import envvars
from .builder import Builder
from .catalog import Catalog
from .doctor import run_doctor
from .runner import Runner
from .store import Store

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".md": "text/markdown; charset=utf-8",
    ".py": "text/plain; charset=utf-8",
    ".sh": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


class App:
    """Holds the state shared by all request threads."""

    def __init__(self, catalog_path: Path, trl_path: Path, runs_dir: Path, db_path: Path, static_dir: Path):
        self.catalog = Catalog(catalog_path)
        self.trl_path = Path(trl_path)
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir = Path(static_dir)
        self.store = Store(db_path)
        self.builder = Builder(self.catalog, self.trl_path, self.runs_dir)
        self.runner = Runner(self.store, self.builder, self.runs_dir, self.catalog, env_provider=self.env_vars)
        self._doctor: dict | None = None
        self._doctor_at: float = 0.0
        self._env_vars: dict[str, str] | None = None

    def env_vars(self) -> dict[str, str]:
        """The environment variables the user set on the 「环境」 page (cached per request)."""

        if self._env_vars is None:
            self._env_vars = envvars.load(self.store)
        return self._env_vars

    def reload_env_vars(self) -> dict[str, str]:
        self._env_vars = envvars.load(self.store)
        self._doctor = None  # the probe depends on CUDA_VISIBLE_DEVICES / HF_HOME / HF_ENDPOINT
        self._doctor_at = 0.0
        return self._env_vars

    def doctor(self, refresh: bool = False) -> dict:
        if refresh or self._doctor is None or time.time() - self._doctor_at > 30:
            self._doctor = run_doctor(self.trl_path, self.runs_dir, self.env_vars())
            self._doctor_at = time.time()
        return self._doctor


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "trl-ui"

    app: App  # injected in make_handler()

    # ------------------------------------------------------------------ plumbing
    def log_message(self, fmt: str, *args) -> None:  # keep the console readable
        if "/events" in (self.path or ""):
            return
        if args and str(args[0]).startswith(("GET /api/jobs", "POST /api/jobs", "GET /api/log")):
            return
        super().log_message(fmt, *args)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message, "status": status}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def _static(self, rel: str) -> None:
        target = (self.app.static_dir / rel.lstrip("/")).resolve()
        if not target.is_relative_to(self.app.static_dir.resolve()) or not target.is_file():
            self._error(404, "not found")
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # --------------------------------------------------------------------- verbs
    def do_GET(self) -> None:  # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")

    def _route(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path == "/" or path == "/index.html":
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/") :])
            if path.startswith("/api/"):
                return self._api(method, path[len("/api/") :], query)
            return self._error(404, "not found")
        except ValueError as exc:
            return self._error(400, str(exc))
        except BrokenPipeError:
            return
        except Exception as exc:  # never take the server down for one bad request
            import traceback

            traceback.print_exc()
            return self._error(500, f"{type(exc).__name__}: {exc}")

    # ----------------------------------------------------------------------- api
    def _api(self, method: str, path: str, query: dict) -> None:
        app = self.app
        parts = [p for p in path.split("/") if p]

        if parts == ["catalog"]:
            payload = app.catalog.summary()
            payload["doctor"] = {
                k: app.doctor().get(k) for k in ("can_train", "recommended_mode", "backend", "devices")
            }
            return self._json(payload)

        if len(parts) == 2 and parts[0] == "algorithms":
            schema = app.catalog.algorithm(parts[1])
            if schema is None:
                return self._error(404, f"未知算法 {parts[1]}")
            schema["schema_fields"] = app.catalog.schema_for(parts[1])["fields"]
            schema["preset"] = app.catalog.presets.get(parts[1], {})
            schema["stages"] = app.catalog.stages
            return self._json(schema)

        if parts == ["doctor"]:
            return self._json(app.doctor(refresh="refresh" in query))

        if parts == ["env-vars"]:
            if method == "GET":
                return self._json({"fields": envvars.describe(app.env_vars())})
            if method == "POST":
                return self._save_env_vars(self._body())

        if parts == ["validate"] and method == "POST":
            body = self._body()
            result = app.builder.validate(
                body.get("algorithm", ""),
                body.get("params") or {},
                num_processes=int(body.get("num_processes") or 1),
                env=app.doctor(),
            )
            return self._json(result)

        if parts == ["preview"] and method == "POST":
            body = self._body()
            job_id = body.get("job_id") or f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            build = app.builder.build(
                job_id=job_id,
                algorithm_id=body.get("algorithm", ""),
                params=body.get("params") or {},
                resource=body.get("resource") or {},
                env=app.doctor(),
            )
            if not build["ok"]:
                return self._json({"ok": False, "errors": build["errors"], "warnings": build["warnings"], "job_id": job_id})
            return self._json(
                {
                    "ok": True,
                    "job_id": job_id,
                    "errors": [],
                    "warnings": build["warnings"],
                    "command": app.builder.shell_command(build["command"]),
                    "entry": build["entry"],
                    "output_dir": build["output_dir"],
                    "files": sorted(build["files"]),
                    "params": build["params"],
                }
            )

        if parts == ["jobs"]:
            if method == "GET":
                jobs = app.store.list_jobs()
                for job in jobs:
                    job["running"] = app.runner.is_running(job["id"])
                return self._json({"jobs": jobs})
            if method == "POST":
                return self._create_job(self._body())

        if len(parts) >= 2 and parts[0] == "jobs":
            job_id = parts[1]
            job = app.store.get_job(job_id)
            if job is None:
                return self._error(404, f"任务 {job_id} 不存在")
            job_dir = app.runs_dir / job_id

            if len(parts) == 2 and method == "GET":
                return self._json(
                    {
                        "job": job,
                        "command": app.builder.shell_command(job["command"]),
                        "algorithm": app.catalog.algorithm(job["algorithm"]),
                        "files": app.runner.file_list(job_dir),
                        "running": app.runner.is_running(job_id),
                    }
                )
            if len(parts) == 3 and parts[2] in {"start", "stop"} and method == "POST":
                result = app.runner.start(job) if parts[2] == "start" else app.runner.stop(job)
                return self._json({**result, "job": app.store.get_job(job_id)})
            if len(parts) == 3 and parts[2] == "delete" and method == "POST":
                ok = app.store.delete_job(job_id)
                return self._json(
                    {"ok": ok, "message": "已删除任务记录" if ok else "任务正在运行或状态不允许删除"}
                )
            if len(parts) == 3 and parts[2] == "logs" and method == "GET":
                since = int((query.get("since") or ["0"])[0])
                return self._json(
                    {
                        "events": app.store.get_events(job_id, since),
                        "last_id": app.store.last_event_id(job_id),
                        "status": job["status"],
                        "running": app.runner.is_running(job_id),
                    }
                )
            if len(parts) == 3 and parts[2] == "metrics" and method == "GET":
                return self._json({"metrics": app.store.get_metrics(job_id), "job": job})
            if len(parts) == 3 and parts[2] == "events" and method == "GET":
                return self._sse(job_id, query)
            if len(parts) == 4 and parts[2] == "files" and method == "GET":
                return self._download(job_dir, parts[3])

        return self._error(404, f"未知接口 {self.path}")

    # ------------------------------------------------------------------ actions
    def _save_env_vars(self, body: dict) -> None:
        """Persist the environment variables of the 「环境」 page.

        Only the four whitelisted names are accepted (`envvars.save` rejects the rest), and an
        empty value removes a variable — that is how a field is cleared, and how HF_TOKEN is
        deleted without the browser ever holding its value.
        """

        app = self.app
        values = body.get("values")
        if not isinstance(values, dict):
            return self._error(400, "请求体需要 values 对象")
        try:
            stored, errors = envvars.save(app.store, values)
        except ValueError as exc:
            return self._error(400, str(exc))
        if errors:
            # `field_errors` is a name → message map: the form shows them inline, next to the
            # box the user has to fix (the API's `errors` key elsewhere is a list of entries).
            return self._json(
                {
                    "ok": False,
                    "error": "、".join(f"{name}：{message}" for name, message in errors.items()),
                    "field_errors": errors,
                    "fields": envvars.describe(app.env_vars()),
                },
                400,
            )
        app.reload_env_vars()
        removed = [name for name, value in values.items() if not (value or "").strip() and name not in stored]
        return self._json({"ok": True, "errors": {}, "removed": removed, "fields": envvars.describe(stored)})

    def _create_job(self, body: dict) -> None:
        app = self.app
        algorithm = (body.get("algorithm") or "").strip()
        if app.catalog.algorithm(algorithm) is None:
            return self._error(400, f"未知算法 {algorithm!r}")
        mode = body.get("mode") or app.doctor()["recommended_mode"]
        if mode not in {"real", "simulate"}:
            return self._error(400, "mode 只能是 real 或 simulate")
        resource = body.get("resource") or {}

        requested_id = (body.get("job_id") or "").strip()
        job_id = (
            requested_id
            if re.fullmatch(r"job-[0-9a-z-]{6,60}", requested_id)
            else f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        build = app.builder.build(
            job_id=job_id,
            algorithm_id=algorithm,
            params=body.get("params") or {},
            resource=resource,
            env=app.doctor(),
        )
        if not build["ok"]:
            return self._json({"ok": False, "errors": build["errors"], "warnings": build["warnings"]}, 400)

        name = (body.get("name") or f"{algorithm}-{job_id[-6:]}").strip()
        job = app.store.create_job(
            job_id=job_id,
            name=name,
            algorithm=algorithm,
            mode=mode,
            params=build["params"],
            command=build["command"],
            output_dir=build["output_dir"],
        )
        written = app.runner.write_artifacts(job, build)
        app.store.append_event(job["id"], f"任务已创建，产物目录 runs/{job['id']}/：{', '.join(written)}", "system")
        if mode == "simulate":
            app.store.append_event(
                job["id"], "提示：当前机器无法真实训练，任务将以模拟模式运行（可在任务页切换成 real 看真实报错）", "system"
            )
        return self._json(
            {
                "ok": True,
                "job": app.store.get_job(job["id"]),
                "command": app.builder.shell_command(build["command"]),
                "warnings": build["warnings"],
                "files": written,
            },
            201,
        )

    def _download(self, job_dir: Path, name: str) -> None:
        target = (job_dir / name).resolve()
        if not target.is_relative_to(job_dir.resolve()) or not target.is_file():
            return self._error(404, "文件不存在")
        body = target.read_bytes()
        if len(body) > 512 * 1024:
            return self._error(413, "文件过大，请直接在磁盘上查看")
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(target.suffix, "text/plain; charset=utf-8"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'inline; filename="{target.name}"')
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------------------- SSE
    def _sse(self, job_id: str, query: dict) -> None:
        app = self.app
        header_id = self.headers.get("Last-Event-ID")
        since = int((query.get("since") or ["0"])[0])
        if header_id and header_id.isdigit():
            since = max(since, int(header_id))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(event: str, data, event_id: int | None = None) -> None:
            chunk = ""
            if event_id is not None:
                chunk += f"id: {event_id}\n"
            chunk += f"event: {event}\n"
            chunk += "data: " + json.dumps(data, ensure_ascii=False, default=str) + "\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

        last_id = since
        idle = 0.0
        try:
            emit("hello", {"job_id": job_id, "since": since})
            while True:
                events = app.store.get_events(job_id, last_id, limit=500)
                for event in events:
                    last_id = max(last_id, event["id"])
                    emit("log", event, event["id"])
                job = app.store.get_job(job_id)
                if job is None:
                    emit("gone", {"job_id": job_id})
                    return
                metrics = app.store.get_metrics(job_id)
                emit(
                    "status",
                    {
                        "status": job["status"],
                        "running": app.runner.is_running(job_id),
                        "message": job.get("message"),
                        "exit_code": job.get("exit_code"),
                        "finished_at": job.get("finished_at"),
                        "metrics": metrics,
                        "last_id": last_id,
                    },
                )
                if events:
                    idle = 0.0
                else:
                    idle += 0.5
                    if idle >= 15:
                        idle = 0.0
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                if not events and job["status"] in {"finished", "failed", "stopped"}:
                    time.sleep(0.4)
                    emit("done", {"status": job["status"], "message": job.get("message")})
                    return
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):
            return


def make_handler(app: App):
    class BoundHandler(Handler):
        pass

    BoundHandler.app = app
    return BoundHandler


def serve(app: App, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    return httpd
