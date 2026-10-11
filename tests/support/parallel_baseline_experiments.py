"""Slow, explicit baseline measurements; never collected by pytest.

Spawn targets are module-level functions. Only the child imports PySpark.
Full child logs remain in data/metadata/parallel-experiments for diagnosis.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import median
from unittest.mock import patch

TRIALS = ((2, 20), (4, 10))
TIMEOUT_SECONDS = 180
IVY_ERRORS = re.compile(r"resolve|lock|ivy-report|ConcurrentModificationException", re.IGNORECASE)


def _wait_for_file(path, timeout=TIMEOUT_SECONDS):
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.005)


def _redirect_output(path):
    with path.open("w", encoding="utf-8") as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)


def _local_runtime(root):
    from janus.utils.environment import load_environment_config, prepare_runtime

    # Measure shipped defaults, independent of the caller's credentials/profile overrides.
    environment = {k: v for k, v in os.environ.items() if not k.startswith("JANUS_")}
    with patch.dict(os.environ, environment, clear=True):
        config = load_environment_config("local", root)
    return config, prepare_runtime(config, root)


def _spark_child(root, directory, index, block=False):
    _redirect_output(directory / f"worker-{index}.log")
    result = {"pid": os.getpid(), "success": False}
    spark = None
    started = time.monotonic()
    try:
        from janus.utils.environment import build_spark_session

        config, paths = _local_runtime(root)
        (directory / f"ready-{index}").touch()
        _wait_for_file(directory / "go")
        result["build_started_ns"] = time.monotonic_ns()
        spark = build_spark_session(config, paths)
        result["application_id"] = spark.sparkContext.applicationId
        result["jvm_pid"] = spark.sparkContext._gateway.proc.pid
        result["count"] = spark.range(1).count()
        result["success"] = result["count"] == 1
        if block:
            (directory / "session.json").write_text(json.dumps(result), encoding="utf-8")
            while True:
                time.sleep(0.1)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception as error:
                result["success"] = False
                result["stop_error"] = f"{type(error).__name__}: {error}"
        result["wall_seconds"] = round(time.monotonic() - started, 6)
        (directory / f"worker-{index}.json").write_text(json.dumps(result), encoding="utf-8")


def _terminate(process):
    if process.is_alive():
        process.kill()
    process.join(timeout=5)


def _cache_reports(root):
    ivy = root / "data/metadata/ivy"
    return {
        str(path.relative_to(ivy)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(ivy.rglob("*"))
        if path.is_file()
        and (
            (path.name.startswith("resolved-") and path.suffix == ".xml")
            or "-spark-submit-parent-" in path.name
        )
    }


def _ivy_trial(root, directory, workers):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_spark_child, args=(root, directory, i)) for i in range(workers)
    ]
    before = _cache_reports(root)
    started = time.monotonic()
    try:
        for process in processes:
            process.start()
        for index in range(workers):
            _wait_for_file(directory / f"ready-{index}")
        (directory / "go").touch()
        deadline = time.monotonic() + TIMEOUT_SECONDS
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
    finally:
        for process in processes:
            _terminate(process)
    after = _cache_reports(root)
    results = []
    for index, process in enumerate(processes):
        result_path = directory / f"worker-{index}.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"success": False}
        log_path = directory / f"worker-{index}.log"
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        results.append(
            {
                **result,
                "exit_code": process.exitcode,
                "ivy_error_text": [line for line in log.splitlines() if IVY_ERRORS.search(line)],
                "log": str(log_path.relative_to(root)),
            }
        )
    starts = [item["build_started_ns"] for item in results if "build_started_ns" in item]
    return {
        "workers": workers,
        "success": all(item["success"] and item["exit_code"] == 0 for item in results),
        "wall_seconds": round(time.monotonic() - started, 6),
        "build_start_spread_ms": round((max(starts) - min(starts)) / 1e6, 3) if starts else None,
        "results": results,
        "cache_added": sorted(after.keys() - before.keys()),
        "cache_removed": sorted(before.keys() - after.keys()),
        "cache_changed": sorted(
            key for key in before.keys() & after.keys() if before[key] != after[key]
        ),
    }


def ivy_experiment(root):
    directory = _experiment_directory(root, "ivy")
    trials = []
    for workers, repetitions in TRIALS:
        for _ in range(repetitions):
            trial_dir = directory / f"trial-{len(trials) + 1:02d}"
            trial_dir.mkdir()
            trial = _ivy_trial(root, trial_dir, workers)
            trial["trial"] = len(trials) + 1
            trials.append(trial)
            print(
                f"Ivy trial {trial['trial']}/30 N={workers}: {trial['success']}",
                file=sys.stderr,
                flush=True,
            )
    failures = sum(not item["success"] for item in trials)
    return {
        "trials": trials,
        "failures": failures,
        "decision": (
            "keep spark.jars.packages" if not failures else "TASK-12 uses verified spark.jars"
        ),
    }


def _process_rows():
    """ps -o pid,ppid,comm equivalent when the slim image has no procps binary."""
    rows = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                stat = (entry / "stat").read_text()
                end = stat.rindex(")")
                state, ppid, *_ = stat[end + 2 :].split()
                rows.append(
                    {
                        "pid": int(entry.name),
                        "ppid": int(ppid),
                        "comm": stat[stat.index("(") + 1 : end],
                        "state": state,
                    }
                )
            except (OSError, ValueError):
                continue
    return rows


def orphan_jvm_experiment(root):
    directory = _experiment_directory(root, "orphan-jvm")
    trials = []
    context = multiprocessing.get_context("spawn")
    for signum in (signal.SIGKILL, signal.SIGTERM):
        trial_dir = directory / signum.name
        trial_dir.mkdir()
        process = context.Process(target=_spark_child, args=(root, trial_dir, 0, True))
        process.start()
        jvm_pid = None
        try:
            _wait_for_file(trial_dir / "ready-0")
            (trial_dir / "go").touch()
            _wait_for_file(trial_dir / "session.json")
            session = json.loads((trial_dir / "session.json").read_text())
            jvm_pid = session["jvm_pid"]
            original = next(row for row in _process_rows() if row["pid"] == jvm_pid)
            if original["ppid"] != process.pid or original["comm"] != "java":
                raise AssertionError(f"Unexpected gateway process: {original}")
            started = time.monotonic()
            os.kill(process.pid, signum)
            process.join(timeout=5)
            observations = []
            exited_at = None
            deadline = started + 30
            while True:
                row = next((row for row in _process_rows() if row["pid"] == jvm_pid), None)
                elapsed = round(time.monotonic() - started, 6)
                observations.append({"elapsed_seconds": elapsed, "process": row})
                if exited_at is None and (row is None or row["state"] == "Z"):
                    exited_at = elapsed
                if row is None or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            trials.append(
                {
                    "signal": signum.name,
                    "worker_exit_code": process.exitcode,
                    "original_process": original,
                    "observations": observations,
                    "time_to_exit_seconds": exited_at if exited_at is not None else "never",
                    "time_to_disappear_seconds": elapsed if row is None else "never",
                    "disappeared": row is None,
                    "zombie": row is not None and row["state"] == "Z",
                }
            )
        finally:
            _terminate(process)
            # Cleanup only the gateway owned by this experiment if it stayed alive.
            if jvm_pid is not None and any(
                row["pid"] == jvm_pid and row["state"] != "Z" for row in _process_rows()
            ):
                os.kill(jvm_pid, signal.SIGKILL)
    return {
        "trials": trials,
        "polling": "/proc stat (ps unavailable in pinned slim image)",
        "decision": "TASK-16 records and kills JVM pid"
        if any(item["time_to_exit_seconds"] == "never" for item in trials)
        else "TASK-16 needs no explicit live-JVM kill",
    }


def _spawn_target(mode, root):
    if mode == "noop":
        return
    import janus.runtime  # noqa: F401

    if mode == "import":
        return
    from janus.registry import load_registry

    registry = load_registry(root)
    if mode == "registry":
        return
    from janus.orchestration import BatchPlanner, BatchPlanRequest, BatchSelection

    # Only the example is enabled in the shipped registry. Enable the ten CNPJ
    # configs in this in-memory snapshot, retaining their files and contracts.
    registry = replace(
        registry,
        sources=tuple(
            replace(source, enabled=source.domain == "receita_federal")
            for source in registry.sources
        ),
    )
    request = BatchPlanRequest.create(
        environment="local",
        project_root=root,
        pipeline_run_id="parallel-spawn-cost",
        planned_at=datetime(2026, 1, 1, tzinfo=UTC),
        selection=BatchSelection.create(domains=("receita_federal",)),
    )
    plan = BatchPlanner().plan(request, registry=registry)
    if len(plan.sources) != 10:
        raise AssertionError(f"Expected ten sources, found {len(plan.sources)}")
    if any(source.failure is not None for source in plan.sources):
        raise AssertionError("Spawn-cost batch contains a planning failure")


def spawn_cost_experiment(root):
    context = multiprocessing.get_context("spawn")
    rows = []
    for mode in ("noop", "import", "registry", "plan"):
        samples = []
        for _ in range(20):
            process = context.Process(target=_spawn_target, args=(mode, root))
            started = time.perf_counter()
            process.start()
            process.join(timeout=60)
            elapsed = time.perf_counter() - started
            if process.exitcode != 0:
                _terminate(process)
                raise RuntimeError(f"Spawn {mode} failed: exit {process.exitcode}")
            samples.append(round(elapsed, 6))
        rows.append(
            {
                "mode": mode,
                "samples_seconds": samples,
                "median_seconds": round(median(samples), 6),
                "p95_seconds": sorted(samples)[18],
            }
        )  # Nearest rank, ceil(.95 * 20).
    return {
        "rows": rows,
        "p95_method": "nearest rank",
        "decision": (
            "must explain planning cost above 5 seconds"
            if rows[-1]["p95_seconds"] > 5
            else "Keep whole-batch planning through BatchPlanner"
        ),
    }


def _experiment_directory(root, name):
    directory = root / "data/metadata/parallel-experiments" / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def environment():
    java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    nproc = subprocess.run(["nproc"], capture_output=True, text=True, check=True)
    return {
        "python": platform.python_version(),
        "pyspark": version("pyspark"),
        "java": java.stderr.strip(),
        "nproc": int(nproc.stdout),
        "os.cpu_count": os.cpu_count(),
        "os.process_cpu_count": os.process_cpu_count(),
        "platform": platform.platform(),
        "container": Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(),
    }


def run_experiment(name, root):
    if not (Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()):
        raise RuntimeError("Run the M0 measurements inside the JANUS container")
    drivers = {
        "ivy": ivy_experiment,
        "orphan-jvm": orphan_jvm_experiment,
        "spawn-cost": spawn_cost_experiment,
    }
    result = environment() if name == "environment" else drivers[name](root)
    return {"run_at": datetime.now(tz=UTC).isoformat(), "experiment": name, **result}
