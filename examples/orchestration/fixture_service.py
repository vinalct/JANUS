"""Deterministic credential-free HTTP fixture for the orchestration example."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_READY_FILE = PROJECT_ROOT / "runtime" / "fixture" / "ready.json"


class ExampleFixtureServer(ThreadingHTTPServer):
    """HTTP server carrying immutable payloads and the deliberate A-failure switch."""

    fail_a: bool
    details: dict[str, list[dict[str, Any]]]
    reference: list[dict[str, Any]]
    ready_file: Path


class ExampleFixtureHandler(BaseHTTPRequestHandler):
    server: ExampleFixtureServer

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/health":
            self._send(200, {"status": "ready"})
            return
        if parsed.path == "/reference":
            if self.server.fail_a:
                self._send(503, {"error": "deliberate A failure"})
                return
            self._send(200, self.server.reference if _first_page(query) else [])
            return
        if parsed.path == "/details":
            reference_id = _required_query(query, "reference_id")
            payload = self.server.details.get(reference_id)
            if payload is None:
                self._send(404, {"error": f"unknown reference_id {reference_id!r}"})
                return
            self._send(200, payload if _first_page(query) else [])
            return
        if parsed.path == "/independent":
            window_start = _required_query(query, "window_start")
            window_end = _required_query(query, "window_end")
            payload = [
                {
                    "event_id": f"independent-{window_start}-{window_end}",
                    "window_start": window_start,
                    "window_end": window_end,
                }
            ]
            self._send(200, payload if _first_page(query) else [])
            return
        self._send(404, {"error": f"unknown fixture path {parsed.path!r}"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"fixture {self.address_string()} {format % args}")

    def _send(self, status: int, payload: object) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    fail_a: bool = False,
    ready_file: Path = DEFAULT_READY_FILE,
) -> ExampleFixtureServer:
    """Build, but do not start, the deterministic fixture server."""
    payload_root = PROJECT_ROOT / "payloads"
    server = ExampleFixtureServer((host, port), ExampleFixtureHandler)
    server.fail_a = fail_a
    server.ready_file = ready_file
    server.reference = _load_json(payload_root / "reference.json", list)
    server.details = _load_json(payload_root / "details.json", dict)
    return server


def _load_json(path: Path, expected_type: type[Any]) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, expected_type):
        raise TypeError(f"Fixture payload {path} must contain {expected_type.__name__}")
    return payload


def _required_query(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name, [])
    return values[0] if values and values[0] else ""


def _first_page(query: dict[str, list[str]]) -> bool:
    return query.get("page", ["1"])[0] == "1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--fail-a",
        action="store_true",
        help="Return HTTP 503 for A while keeping C healthy.",
    )
    parser.add_argument("--ready-file", type=Path, default=DEFAULT_READY_FILE)
    args = parser.parse_args()

    server = build_server(
        host=args.host,
        port=args.port,
        fail_a=args.fail_a,
        ready_file=args.ready_file,
    )
    address, port = server.server_address
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.write_text(
        json.dumps({"host": address, "port": port, "fail_a": args.fail_a}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"JANUS example fixture listening at http://{address}:{port}; fail_a={args.fail_a}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        args.ready_file.unlink(missing_ok=True)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
