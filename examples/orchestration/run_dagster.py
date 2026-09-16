"""Execute and collect one local Dagster run of the orchestration example."""

from __future__ import annotations

import argparse
import json

from bootstrap import prepare_example_runtime
from definitions import build_example_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Bounded Dagster-owned whole-source retries; defaults to zero.",
    )
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error("--max-retries must be zero or greater")

    prepare_example_runtime()
    adapter = build_example_adapter(max_retries=args.max_retries)
    result, outcome = adapter.execute_in_process(raise_on_error=False)
    print(json.dumps(outcome.to_summary(), indent=2, sort_keys=True))
    return 0 if result.success and outcome.is_successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
