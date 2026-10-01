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

This check validates the quality rules themselves before looking at any data.

It currently catches:

- duplicate entries in `required_fields`;
- duplicate entries in `unique_fields`;
- `unique_fields` that are not also listed in `required_fields`.

That last rule is deliberate. If JANUS is going to enforce uniqueness on a key, that key should also be treated as mandatory.

#### `schema_contract_mode`

This check validates whether the schema settings are coherent.

It reports the fields from the contract loaded into the plan. Every source must declare
`schema.contract`; contract loading and active-status errors are collected by the registry
before the quality gate runs. Schema evolution follows the contract's
`janus.compatibility` declaration.

## Data checks

These checks run against a Spark DataFrame.

#### `required_fields`

This check verifies that all configured required fields exist and are populated.

It fails when:

- a required column is missing from the DataFrame;
- a required field contains null values;
- a required string field contains blank values.

#### `unique_fields`

This check verifies that the configured uniqueness key is actually unique in the provided DataFrame.

It fails when:

- one or more unique key columns are missing;
- duplicate key groups are found.

When duplicates exist, the report includes the duplicate-group count and a small sample of the repeated keys.

#### `schema_expectations`

This check compares the observed DataFrame columns with an expected schema contract.

That expected schema can come from either:

- an explicit list of fields passed by the caller; or
- a JSON schema file declared in the source config.

It fails when:

- expected fields are missing from the observed DataFrame;
- unexpected fields appear while schema evolution is disabled.

If schema evolution is allowed, extra fields are tolerated and the check still passes.

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
- rejecting inconsistent quality-rule configuration;
- rejecting unexpected columns when schema evolution is disabled;
- rejecting outputs that land outside the configured zone root;
- loading expected field names from a JSON schema document.

The focused verification for this step passed with:

- `python -m ruff check src/janus/quality tests/unit/quality`
- `python -m pytest tests/unit/quality/test_validators.py`

In the current shell, the Spark-backed tests are skipped when `pyspark` is not installed. The non-Spark checks still run and protect the config, schema-loading, and output-contract behavior.

