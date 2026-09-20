"""Peak-RSS probe for the file family's download path."""

from __future__ import annotations

import gc
import resource
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from janus.models import ExecutionPlan, RunContext, SourceConfig

_CHUNK = b"janus-memory-probe-payload-" + bytes(range(256)) * 16


@dataclass(frozen=True, slots=True)
class MemoryProbeResult:
    """What one probe run observed."""

    payload_bytes: int
    baseline_rss_bytes: int
    peak_rss_bytes: int
    artifact_count: int
    artifact_bytes: int

    @property
    def delta_bytes(self) -> int:
        """Peak RSS minus the baseline taken just before extraction started."""
        return self.peak_rss_bytes - self.baseline_rss_bytes

    @property
    def delta_ratio(self) -> float:
        """Peak-RSS growth as a multiple of the payload."""
        return self.delta_bytes / self.payload_bytes if self.payload_bytes else 0.0

    def render(self) -> str:
        mib = 2**20
        return (
            f"payload           : {self.payload_bytes:,} bytes "
            f"({self.payload_bytes / mib:.1f} MiB)\n"
            f"baseline RSS      : {self.baseline_rss_bytes:,} bytes "
            f"({self.baseline_rss_bytes / mib:.1f} MiB)\n"
            f"peak RSS          : {self.peak_rss_bytes:,} bytes "
            f"({self.peak_rss_bytes / mib:.1f} MiB)\n"
            f"peak - baseline   : {self.delta_bytes:,} bytes "
            f"({self.delta_bytes / mib:.1f} MiB)\n"
            f"delta / payload   : {self.delta_ratio:.2f}x\n"
            f"artifacts written : {self.artifact_count} "
            f"({self.artifact_bytes:,} bytes)"
        )


def max_rss_bytes() -> int:
    """This process's peak RSS in bytes (``ru_maxrss`` is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class _PayloadHandler(BaseHTTPRequestHandler):
    """Serves the generated payload with a declared ``Content-Length``, and nothing else."""

    payload_bytes = 0
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # BaseHTTPRequestHandler dispatches on this exact name
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Length", str(self.payload_bytes))
        self.end_headers()
        remaining = self.payload_bytes
        while remaining > 0:
            block = _CHUNK[: min(len(_CHUNK), remaining)]
            self.wfile.write(block)
            remaining -= len(block)

    def do_HEAD(self) -> None:  # BaseHTTPRequestHandler dispatches on this exact name
        self.send_response(200)
        self.send_header("Content-Length", str(self.payload_bytes))
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # stdlib signature
        """Silence the default stderr access log."""


@contextmanager
def serve_payload(payload_bytes: int) -> Iterator[str]:
    """Serve ``payload_bytes`` of filler on loopback; yields the URL of the one artifact."""
    handler = type("_SizedPayloadHandler", (_PayloadHandler,), {"payload_bytes": payload_bytes})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}/probe.csv"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


#: The smallest ``local`` profile ``_default_storage_layout`` accepts. The probe runs against a
#: throwaway project root, so it seeds this rather than reading the repository's own profile —
#: a measurement that wrote into ``data/raw`` would not be repeatable.
_PROBE_ENVIRONMENT_PROFILE = """\
name: local

runtime:
  log_level: WARNING

spark:
  app_name: janus-memory-probe
  master: local[1]
  warehouse_dir: data/metadata/spark-warehouse

storage:
  root_dir: data
  raw_dir: data/raw
  bronze_dir: data/bronze
  metadata_dir: data/metadata
"""


def seed_environment_profile(project_root: Path) -> Path:
    """Write the minimal ``local`` profile into ``project_root`` unless one is already there."""
    config_path = project_root / "conf" / "environments" / "local.yaml"
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_PROBE_ENVIRONMENT_PROFILE, encoding="utf-8")
    return config_path


def build_probe_plan(
    project_root: Path, url: str, *, source_id: str = "memory_probe"
) -> ExecutionPlan:
    """A minimal enabled ``static_file`` plan pointed at ``url``.

    Mirrors the shape ``tests/unit/strategies/http/conftest.py::build_plan("file", …)``
    produces, built here so the probe stays importable outside that suite's conftest.
    """
    source_config = SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "file",
            "strategy": "file",
            "strategy_variant": "static_file",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": {
                "url": url,
                "method": "GET",
                "format": "csv",
                "timeout_seconds": 120,
                "auth": {"type": "none"},
                "pagination": {"type": "none"},
                "rate_limit": {
                    "requests_per_minute": 600,
                    "concurrency": 1,
                    "backoff_seconds": 1,
                },
            },
            "extraction": {
                "mode": "full_refresh",
                "checkpoint_strategy": "none",
                "retry": {
                    "max_attempts": 1,
                    "backoff_strategy": "fixed",
                    "backoff_seconds": 1,
                },
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": "csv", "write_mode": "append"},
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "csv"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        project_root / "conf" / "sources" / f"{source_id}.yaml",
    )
    run_context = RunContext.create(
        run_id=f"run-{source_id}",
        environment="local",
        project_root=project_root,
        started_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def measure_download_peak_rss(project_root: Path, payload_bytes: int) -> MemoryProbeResult:
    """Run one real ``FileStrategy.extract`` over a generated payload and report peak RSS.

    ``ru_maxrss`` is a high-water mark that never falls, so the baseline is taken after the
    server, the plan and the strategy exist — the delta is what *extraction* added.
    """
    from janus.strategies.files.core import FileStrategy

    seed_environment_profile(project_root)
    with serve_payload(payload_bytes) as url:
        plan = build_probe_plan(project_root, url)
        strategy = FileStrategy()
        gc.collect()
        baseline = max_rss_bytes()
        result = strategy.extract(plan)
        peak = max_rss_bytes()

    return MemoryProbeResult(
        payload_bytes=payload_bytes,
        baseline_rss_bytes=baseline,
        peak_rss_bytes=peak,
        artifact_count=len(result.artifacts),
        artifact_bytes=sum(
            Path(artifact.path).stat().st_size
            for artifact in result.artifacts
            if Path(artifact.path).is_file()
        ),
    )


def main() -> int:
    """One-off run: ``python -m tests.support.memory_probe [payload_mib] [project_root]``."""
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload_mib", nargs="?", type=int, default=256)
    parser.add_argument("project_root", nargs="?", default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="janus-memory-probe-") as tmp:
        root = Path(args.project_root) if args.project_root else Path(tmp)
        print(measure_download_peak_rss(root, args.payload_mib * 2**20).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
