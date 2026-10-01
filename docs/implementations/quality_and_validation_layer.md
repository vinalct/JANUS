# Quality and Validation Layer

The short version is: JANUS now has a shared validation layer.

Before this step, the project could validate source YAML while loading the registry, and it could persist run metadata and outputs, but it still did not have one common place to describe quality checks for runtime datasets or to store validation outcomes as a first-class metadata artifact.

This step fills that gap with a small reusable layer built around three ideas:

- validation rules come from the source contract where that makes sense;
- validation should stay strategy-agnostic;
- validation results should be easy to persist and inspect per run.

The result is not a source-specific rules engine. It is a shared quality gate that can be called by later strategy implementations once they have a DataFrame in hand and explicit outputs to validate.

## What was added

- `src/janus/quality/models.py` now defines the typed validation artifacts used across the quality layer.
- `src/janus/quality/store.py` now provides metadata-zone persistence for validation reports.
- `src/janus/quality/validators.py` now provides the reusable validation checks and the `QualityGate` orchestration entry point.
- `src/janus/quality/__init__.py` now exposes the quality-layer API for downstream imports.
- `tests/unit/quality/test_validators.py` covers successful report persistence, actionable failures, schema handling, and output contract checks.

## The validation model

The shared layer is built around two small record types.

### `ValidationCheck`

This is the unit of validation output.

Each check records:

- a phase: `config`, `data`, or `output`;
- a check name;
- an outcome: `passed`, `failed`, or `skipped`;
- a readable message;
- optional details for debugging and reporting.

The idea is to keep one failed rule easy to understand without forcing callers to parse an unstructured exception string.

### `ValidationReport`

This is the run-scoped wrapper around a set of checks.

It carries the run id, source identity, environment, strategy metadata, an emission timestamp, and the full set of emitted checks. It also computes a small summary of how many checks passed, failed, or were skipped.

That gives the metadata layer a stable artifact shape instead of treating validation as an incidental log line.

## What the quality gate checks

`QualityGate` is the orchestration entry point. It gathers the checks below into one report and can optionally persist that report to the metadata zone.

### Config checks

#### `quality_contract`

This check validates the key rules of the data contract loaded into the plan before looking at any
data. Every `primaryKey` column must also be `required`: a null merge key is a failure. The details
keep their `required_fields` / `unique_fields` names but carry the contract's `required` columns and
`primaryKey`. Duplicate property names never reach this check; the contract loader refuses them.

#### `schema_contract_mode`

This check reports the contract loaded into the plan, including its `compatibility` and
`enforcement`. Every source must declare `schema.contract`; contract loading and active-status
errors are collected by the registry before the quality gate runs. A plan without a contract is
reported as `skipped` here; the materializer is what refuses to write it.

## The pre-write pass (before implementation)

The structural decision no longer happens in the gate. `BronzeMaterializer` runs
`run_pre_write_pass` on every batch **before** the bronze write: the pure structural check from
`quality/contract_checks.py`, and for `strict` contracts one aggregation that counts null or blank
required values and malformed JSON/CSV rows. A refused batch raises a `ContractEnforcementError`
subclass carrying `failure_stage`, and nothing is written for it. The gate then *reports* the same
evidence in the validation JSON rather than deciding a second time. See
[Data contracts](../data-contracts.md#enforcement-modes) for the modes and failure stages.

## Data checks

These checks run against a Spark DataFrame, or are rendered from the pre-write evidence.

#### `required_fields`

This check verifies that the contract's `required` columns exist and are populated. Under
`strict` it is rendered from the pre-write count, so a failure means the batch was refused before
the write. Under `lenient` it runs after the write on the normalized DataFrame, as before.

It fails when:

- a required column is missing from the DataFrame;
- a required field contains null values;
- a required string field contains blank values.

#### `unique_fields`

This check verifies that the contract's `primaryKey` is actually unique in the provided DataFrame.

It fails when:

- one or more key columns are missing;
- duplicate key groups are found.

When duplicates exist, the report includes the duplicate-group count and a small sample of the repeated keys.

#### `schema_expectations`

This check reports the structural comparison of the batch with its data contract: names and types
in the contract's type vocabulary. With pre-write evidence it is rendered from what the
materializer already decided, merged across batches (`details.batches`; a failed batch's message
is prefixed `batch i/n:`). Without evidence, for a caller that hands the gate a DataFrame directly,
it runs the same structural check on that frame and ignores the undeclared normalization columns.

It fails when a declared column is missing, an undeclared column is present, or a type differs.
A nullable column the contract marks `required` is listed under `nullability_relaxed` and is not a
failure; the required-value count enforces it. There is no tolerance flag: an undeclared column is
admitted by adding it to the contract.

#### `malformed_rows`

For JSON, JSONL and CSV handoffs this check reports the rows Spark could not parse into the
contract's types, with `count`, `threshold`, `batch_index` and up to five scrubbed samples of at
most 500 characters. It is `failed` when a `strict` batch exceeds `janus.maxMalformedRows`,
`passed` with `severity: warning` under `lenient`, and `skipped` for Parquet handoffs.

## Output checks

#### `output_columns`

This check validates the expected output-column contract on the DataFrame that is about to be treated as a structured output.

By default, it checks for the normalization metadata columns introduced by the shared normalization base, such as:

- `janus_run_id`;
- `janus_source_id`;
- `janus_environment`;
- `janus_strategy_family`;
- `janus_strategy_variant`;
- `ingestion_timestamp`;
- `ingestion_date`.

This is a lightweight sanity check for the bronze-side shared contract rather than a business-schema validator.

#### `materialized_outputs`

This check validates the `WriteResult` objects produced by writers.

It currently checks that:

- the written path stays under the configured root for its zone;
- `records_written` is not negative;
- non-raw outputs use the configured output format.

That means this check can validate raw, bronze, and metadata write contracts at the path and metadata level.

## Raw versus bronze

This step is mostly about DataFrame-level validation, which means the most meaningful checks are intended to run after raw artifacts have been read into Spark and before or around bronze persistence.

In practice, that means:

- config checks are independent of any zone and can run as soon as a plan exists;
- required-field, unique-field, schema, and output-column checks are bronze-oriented because they operate on DataFrames;
- output-contract validation can also look at raw and metadata writes through their `WriteResult` objects.

What this step does **not** add is raw-payload semantic validation.

There is still no generic check yet for questions such as:

- whether a raw JSON response contains expected top-level keys;
- whether a downloaded file matches an upstream checksum contract;
- whether one raw payload shape is valid before Spark reads it.

That boundary is intentional. The shared quality layer now covers the common reusable checks without pretending every raw source artifact can be judged the same way.

## Validation report persistence

Validation reports can now be written to the metadata zone under:

- `validations/<run_id>.json`

The report persistence is intentionally simple. One run gets one validation artifact.

That keeps validation outcomes close to the rest of the run metadata while avoiding a more complicated storage design before the strategy runtime starts using the layer more heavily.

## What the tests lock down

The new unit tests cover the parts of the layer that later strategy work will depend on.

They cover:

- building and persisting a successful validation report;
- surfacing clear failure messages for missing, blank, and duplicate values;
- rejecting a `primaryKey` column that the contract does not mark `required`;
- reporting structural mismatches against the data contract;
- rejecting outputs that land outside the configured zone root.

The pre-write pass has its own suites: `tests/unit/quality/test_contract_checks.py`,
`tests/unit/quality/test_pre_write_pass.py` and `tests/unit/quality/test_malformed_rows_check.py`.

The focused verification for this step passed with:

- `python -m ruff check src/janus/quality tests/unit/quality`
- `python -m pytest tests/unit/quality/test_validators.py`

In the current shell, the Spark-backed tests are skipped when `pyspark` is not installed. The non-Spark checks still run and protect the config, schema-loading, and output-contract behavior.

