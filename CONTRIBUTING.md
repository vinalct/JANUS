# Contributing to JANUS

Thanks for contributing. This project targets a pinned, reproducible toolchain
(Python `3.13.12`, PySpark `4.0.1`, OpenJDK `17`, Iceberg runtime `1.10.1`) and
enforces it in CI. The one rule that matters most: **get CI green locally before
you push.**

## How CI works

Every push to `main` and every pull request targeting `main` runs
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), which has two jobs:

- **fast** (no JVM) — `ruff check src tests`, `mypy`, and the unit suite
  (`pytest tests/unit`). Spark-backed tests self-skip here, which is expected.
  This is the quick feedback loop (a few minutes).
- **spark** (full stack, in the container) — builds the JANUS image from
  [`docker/Dockerfile`](docker/Dockerfile) and runs the **entire** suite,
  including `tests/integration`, so PySpark, JDK 17, and the Iceberg runtime are
  present. This is the job that actually exercises the Spark writer and Iceberg
  commit path.

Both jobs are **required** status checks before a change can merge to `main`.

Two guarantees the spark job enforces:

- **Zero PySpark skips.** The job fails if any test is skipped because PySpark
  could not be imported. If you add a Spark-backed test, guard the import with
  `pytest.importorskip("pyspark.sql")` so the fast job still self-skips cleanly
  while the container job runs it for real.
- **Hermeticity.** CI uses no secrets and makes no calls to live federal
  endpoints. All sources stay `enabled: false`; tests rely on fixtures and
  injected transports. Keep new tests offline.

## Reproduce CI locally

The closest reproduction of the pipeline is a single command:

```bash
make ci
```

This runs lint + type check + the full test suite **inside the container**, so
it exercises the same PySpark/JDK/Iceberg stack as the spark CI job. It exits
non-zero on any lint, type, or test failure. `make ci` is kept in lockstep with
`ci.yml`; if you change the workflow, change `make ci` in the same PR.

### Fast local loop (no container)

If you want the sub-minute loop and only need the fast-job checks, install the
pinned fast-job tooling on the host and run them directly:

```bash
python -m pip install \
  ruff==0.11.6 \
  mypy==2.2.0 \
  pytest==9.0.2 \
  PyYAML==6.0.3 \
  certifi==2026.2.25

ruff check src tests && mypy && pytest tests/unit -ra
```

## Licensing and contribution terms

JANUS is released under the [Apache License 2.0](LICENSE). By contributing, you
agree that your contributions are licensed under the same terms.

We use the **Developer Certificate of Origin (DCO)**: you certify that you wrote
the contribution or otherwise have the right to submit it under the project's
license. Sign off each commit with the `-s` flag, which appends a
`Signed-off-by` trailer with your real name and email:

```bash
git commit -s -m "your message"
```

By signing off, you certify the [DCO 1.1](https://developercertificate.org/).

Do not submit code you do not have the right to license (e.g. copied from a
GPL/AGPL project, or work owned by an employer or academic institution under an
IP-assignment clause) — it cannot be accepted. If a project entity is later
formed, contributors may be asked to accept a Contributor License Agreement
(CLA) so the terms can be administered centrally.
