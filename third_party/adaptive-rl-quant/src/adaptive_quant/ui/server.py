from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from adaptive_quant.security_bypass import enforce_security_bypass_policy
from adaptive_quant.ui.catalog import launcher_catalog
from adaptive_quant.ui.commands import build_job_env, build_workflow_command, format_command
from adaptive_quant.ui.jobs import JobManager, action_catalog, start_configured_workflow
from adaptive_quant.ui.security import (
    audit_log,
    generate_launcher_token,
    launcher_token_header,
    read_api_json_body,
    validate_run_options,
    verify_launcher_token,
)
from adaptive_quant.ui.status import dashboard_status

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def build_catalog_response(*, repo: Path, launcher_token: str | None = None) -> dict[str, Any]:
    """JSON payload for GET /api/catalog (testable without a live HTTP server)."""
    status = dashboard_status(repo=repo)
    catalog = launcher_catalog(repo=repo, status=status)
    catalog["status"] = status
    if launcher_token:
        catalog["launcher_token"] = launcher_token
    return catalog


def build_preview_response(
    *,
    workflow: str,
    options: dict[str, Any] | None,
    repo: Path,
    python_bin: str,
) -> dict[str, Any]:
    """JSON payload for POST /api/preview (testable without a live HTTP server)."""
    opts = dict(options or {})
    validate_run_options(opts)
    label, command = build_workflow_command(
        workflow=workflow,
        options=opts,
        repo=repo,
        python_bin=python_bin,
    )
    return {
        "workflow": workflow,
        "label": label,
        "command": command,
        "command_preview": format_command(command),
        "env_keys": sorted(build_job_env(opts, repo=repo).keys()),
    }


class LauncherHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        repo: Path,
        python_bin: str,
        jobs: JobManager,
        launcher_token: str,
    ) -> None:
        self.repo = repo
        self.python_bin = python_bin
        self.jobs = jobs
        self.launcher_token = launcher_token
        from adaptive_quant.ui.chat import ChatSessionManager

        self.chat_session = ChatSessionManager(repo=repo)
        super().__init__(server_address, handler_cls)


class LauncherRequestHandler(BaseHTTPRequestHandler):
    server: LauncherHTTPServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any] | list[Any], *, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _parse_run_body(self) -> tuple[str, dict[str, Any]]:
        body = read_api_json_body(self.headers, self.rfile)
        workflow = str(body.get("workflow") or body.get("action", "")).strip()
        if not workflow:
            raise ValueError("workflow is required")
        options = body.get("options")
        if options is not None and not isinstance(options, dict):
            raise ValueError("options must be an object")
        return workflow, options if isinstance(options, dict) else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route in {"/", "/index.html"}:
            index = _STATIC_DIR / "index.html"
            self._send_bytes(index.read_bytes(), "text/html; charset=utf-8")
            return

        if route.startswith("/static/"):
            rel = route.removeprefix("/static/")
            target = (_STATIC_DIR / rel).resolve()
            if not str(target).startswith(str(_STATIC_DIR.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mime, _ = mimetypes.guess_type(str(target))
            self._send_bytes(target.read_bytes(), mime or "application/octet-stream")
            return

        if route == "/api/status":
            status = dashboard_status(repo=self.server.repo)
            status["actions"] = action_catalog(status)
            self._send_json(status)
            return

        if route == "/api/catalog":
            self._send_json(
                build_catalog_response(
                    repo=self.server.repo,
                    launcher_token=self.server.launcher_token,
                )
            )
            return

        if route == "/api/jobs":
            self._send_json(self.server.jobs.list_jobs())
            return

        if route.startswith("/api/jobs/"):
            job_id = route.removeprefix("/api/jobs/").strip("/")
            record = self.server.jobs.get(job_id)
            if record is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_json(record.to_dict())
            return

        if route == "/api/chat/config":
            from adaptive_quant.ui.chat import build_chat_config

            self._send_json(build_chat_config(repo=self.server.repo))
            return

        if route == "/api/models":
            from adaptive_quant.ui.chat import build_models_response

            self._send_json(build_models_response(repo=self.server.repo))
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            verify_launcher_token(self.headers, self.server.launcher_token)
        except PermissionError as exc:
            self._send_json({"error": str(exc)}, status=403)
            return

        parsed = urlparse(self.path)
        if parsed.path == "/api/preview":
            self._handle_preview()
            return
        if parsed.path == "/api/chat":
            self._handle_chat()
            return
        if parsed.path == "/api/models":
            self._handle_models()
            return
        if parsed.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._handle_run()

    def _handle_preview(self) -> None:
        try:
            workflow, options = self._parse_run_body()
            self._send_json(
                build_preview_response(
                    workflow=workflow,
                    options=options,
                    repo=self.server.repo,
                    python_bin=self.server.python_bin,
                )
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)

    def _handle_chat(self) -> None:
        try:
            body = read_api_json_body(self.headers, self.rfile)
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            from adaptive_quant.ui.chat import build_chat_response

            self._send_json(
                build_chat_response(
                    repo=self.server.repo,
                    body=body,
                    session=self.server.chat_session,
                )
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
        except (FileNotFoundError, OSError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_models(self) -> None:
        try:
            body = read_api_json_body(self.headers, self.rfile)
            if body is not None and not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            from adaptive_quant.ui.chat import build_models_response

            self._send_json(build_models_response(repo=self.server.repo, body=body or {}))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
        except (FileNotFoundError, TimeoutError, RuntimeError, OSError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_run(self) -> None:
        try:
            workflow, options = self._parse_run_body()
            audit_log(
                f"run workflow={workflow!r} option_keys={sorted(options.keys())} "
                f"client={self.client_address[0]!r}"
            )
            record = start_configured_workflow(
                jobs=self.server.jobs,
                repo=self.server.repo,
                python_bin=self.server.python_bin,
                workflow=workflow,
                options=options,
            )
            self._send_json(record.to_dict(), status=202)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)


def serve_launcher(
    *,
    repo: Path,
    python_bin: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    stop_event: threading.Event | None = None,
    ready_event: threading.Event | None = None,
    bound_port: list[int] | None = None,
) -> None:
    jobs = JobManager()
    launcher_token = generate_launcher_token()
    enforce_security_bypass_policy(context="launcher")
    httpd = LauncherHTTPServer(
        (host, port),
        LauncherRequestHandler,
        repo=repo,
        python_bin=python_bin,
        jobs=jobs,
        launcher_token=launcher_token,
    )
    if bound_port is not None:
        bound_port.append(int(httpd.server_address[1]))
    if ready_event is not None:
        ready_event.set()
    if stop_event is None:
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
        return

    def _watch_stop() -> None:
        stop_event.wait()
        httpd.shutdown()

    threading.Thread(target=_watch_stop, daemon=True).start()
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


__all__ = [
    "build_catalog_response",
    "build_preview_response",
    "launcher_token_header",
    "serve_launcher",
]
