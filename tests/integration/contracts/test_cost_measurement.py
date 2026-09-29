"""Opt-in NFR-2 timing of the shipped local Spark profile."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.support.cnpj_csv_generator import DEFAULT_ROWS, DEFAULT_SEED, write_csv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = PROJECT_ROOT / "data" / "metadata" / "scratch" / "order-19" / "cost"
MODES = ("lenient", "strict")
CNPJ_CSV_BYTES = 224_036_341
CNPJ_CSV_SHA256 = "abe0704474b5d8a738199fa2a352fd7144cf712fca9e91ca7cc14f3930ff152a"


@pytest.mark.skipif(
    os.environ.get("JANUS_COST_MEASUREMENT") != "1",
    reason="NFR-2 benchmark is opt-in: set JANUS_COST_MEASUREMENT=1 in the Spark container",
)
def test_materialization_cost_under_local_profile() -> None:
    pytest.importorskip("pyspark", reason="NFR-2 benchmark requires the Spark container")
    from tests.support.contract_enforcement_cost import cost_cnpj, cost_inep

    scratch = Path(os.environ.get("JANUS_COST_SCRATCH", SCRATCH))
    scratch.mkdir(parents=True, exist_ok=True)
    csv = scratch / f"estabelecimentos_{DEFAULT_ROWS}_seed{DEFAULT_SEED}.csv"
    digest = write_csv(csv)
    assert csv.stat().st_size == CNPJ_CSV_BYTES
    assert digest == CNPJ_CSV_SHA256
    results: dict[str, dict[str, dict[str, Any]]] = {"inep": {}, "cnpj": {}}
    output = scratch / "measurements.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for shape, measure in (
            ("inep", lambda mode: cost_inep(scratch / mode, enforcement=mode)),
            ("cnpj", lambda mode: cost_cnpj(scratch / mode, csv, enforcement=mode)),
        ):
            for mode in MODES:
                result = measure(mode)
                expected_rows = 3 if shape == "inep" else DEFAULT_ROWS
                assert result["enforcement"] == mode
                assert result["rows"] == expected_rows
                assert result["spark.driver.memory"] == "8g"
                record = {
                    "measured_at_utc": datetime.now(UTC).isoformat(),
                    "shape": shape,
                    "csv_sha256": digest if shape == "cnpj" else None,
                    **result,
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                results[shape][mode] = record

    print("\nNFR-2 materialization cost (three timed runs per cell; after warm-up)")
    print(
        "shape | rows | lenient median s / jobs | strict median s / jobs | delta | "
        "peak RSS KiB lenient / strict"
    )
    for shape in ("inep", "cnpj"):
        lenient, strict = (results[shape][mode] for mode in MODES)
        delta = 100 * (strict["median_seconds"] / lenient["median_seconds"] - 1)
        lenient_jobs = ",".join(str(count) for count in lenient["timed_jobs"])
        strict_jobs = ",".join(str(count) for count in strict["timed_jobs"])
        print(
            f"{shape} | {strict['rows']:,} | "
            f"{lenient['median_seconds']:.3f} / {lenient_jobs} | "
            f"{strict['median_seconds']:.3f} / {strict_jobs} | {delta:+.1f}% | "
            f"{lenient['driver_peak_rss_kib']} / {strict['driver_peak_rss_kib']}"
        )
    print(f"JSONL: {output}")
