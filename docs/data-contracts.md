# Data contracts

## What a contract is and where it lives

Each bronze table has one ODCS v3.2.0 YAML contract at `conf/contracts/<domain>/<table>.yaml`. Co-writers share that file. Source YAML points to it with `schema.contract`. Git records declared intent; the Iceberg catalog records actual table schema and history.

The contract is binding. Every bronze batch is checked against it before the write, the live table is checked against it before extraction, and it decides how a table may change (see [Enforcement modes](#enforcement-modes) and [The evolution matrix](#the-evolution-matrix)).

## Format

JANUS reads the subset below, copied from the [pinned ODCS schema guide](schemas/odcs/README.md). Every checked-in contract validates against the pinned v3.2.0 JSON Schema in CI.

| ODCS key | JANUS use | Required |
|---|---|---|
| `apiVersion` | Must equal `v3.2.0`, the pinned version string. | yes |
| `kind` | Must equal `DataContract`. | yes |
| `id` | Stable contract id; convention: `<domain>.<bronze table>`. | yes |
| `name` | Human-readable contract name. | yes |
| `version` | Contract version as semver `MAJOR.MINOR.PATCH`. | yes |
| `status` | One of `draft`, `active`, or `deprecated`; this is a JANUS restriction of the ODCS free string. | yes |
| `domain` | Must equal the referencing source entry's `domain`. | yes |
| `description.purpose` | One paragraph used by the OpenLineage documentation facet. | yes |
| `tags` | List of strings. | no |
| `team` | List of `{username, role}` members with at least one `owner` role. | yes |
| `schema` | Exactly one element whose `name` is the Bronze table name, whose `physicalType` is `table`, and which contains `properties`. | yes |
| `schema[0].properties[]` | Bronze field metadata: `name`, `businessName`, `description`, derived `logicalType`, JANUS `physicalType`, `required`, `unique`, `primaryKey`, `classification`, and `customProperties` entries for `sourceField` and `sourceFormat`. A `struct` may contain nested `properties`; an `array` may contain `items`. The name `_janus_corrupt_record` is reserved. | `name` and `physicalType`; other values have JANUS defaults |
| `customProperties` | Pairs for `janus.compatibility` (`additive`, `backward`, or `frozen`), `janus.enforcement` (`strict` or `lenient`), optionally `janus.maxMalformedRows` (a non-negative integer string, default `0`), and `janus.draftedFrom` (free text on drafts only). | compatibility and enforcement |

ODCS uses arrays for `schema`, `team` and `customProperties`. For example:

```yaml
apiVersion: v3.2.0
kind: DataContract
id: example.federal_open_data_example
name: Federal open data example
version: '1.0.0'
status: active
domain: example
description:
  purpose: Example bronze table contract.
team:
  - username: janus
    role: owner
schema:
  - name: federal_open_data_example
    physicalType: table
    properties:
      - name: id
        businessName: Record ID
        description: Upstream identifier.
        logicalType: string
        physicalType: string
        required: true
        classification: public
        customProperties:
          - property: sourceField
            value: id
          - property: sourceFormat
            value: json
customProperties:
  - property: janus.compatibility
    value: additive
  - property: janus.enforcement
    value: strict
```

Quote `version`: YAML can parse other numeric-looking values unexpectedly. Keep properties and nested items in read order. The loader ignores allowed ODCS keys outside its subset and rejects misspelled `janus.` properties.

## Type vocabulary

| Contract | ODCS `logicalType` | Iceberg | Spark JSON |
|---|---|---|---|
| `boolean` | `boolean` | `boolean` | `boolean` |
| `integer` | `integer` | `int` | `integer` |
| `long` | `integer` | `long` | `long` |
| `float` | `number` | `float` | `float` |
| `double` | `number` | `double` | `double` |
| `decimal(p,s)` | `number` | `decimal(p,s)` | `decimal(p,s)` |
| `string` | `string` | `string` | `string` |
| `binary` | `string` | `binary` | `binary` |
| `date` | `date` | `date` | `date` |
| `timestamp` | `date` | `timestamp` | `timestamp_ntz` |
| `timestamptz` | `date` | `timestamptz` | `timestamp` |
| `struct` | `object` | `struct<...>` | `struct` |
| `array` | `array` | `list<...>` | `array` |
| `map` | `object` | `map<...>` | `map` |

`timestamp` has no time zone; `timestamptz` is Spark's zone-aware timestamp. Decimal requires precision and scale (`1 ≤ p ≤ 38`, `0 ≤ s ≤ p`). Nested child `required` flags preserve Spark nullability. `short` and `byte` are excluded because silently widening one would change a bronze type.

## Bronze naming rule

Keep upstream column names verbatim in `name`, including case and spelling. `businessName` is the readable label. Rename or normalize fields in silver.

## Draft, review, active

With Spark available, draft from saved raw data or a local fixture:

```sh
janus contract draft --source-id <id> --from-raw <run-id> --include-disabled --out conf/contracts/<domain>/<table>.yaml
janus contract draft --source-id <id> --from-fixture <path> --include-disabled --out conf/contracts/<domain>/<table>.yaml
```

The CLI writes `status: draft` and `janus.draftedFrom`; it never runs inside `run` or `run-all`. Review actual samples, column names and order, nested types, nullability, descriptions, classification, ownership, and agreement with `quality.required_fields` / `unique_fields`. Inference can mistake all-null fields, numeric-looking text, dates and sparse data. Keep inferred order (alphabetical for JSON handoffs, source order for CSV). Set `required: true` only when observed null count is zero and a reviewer confirms the rule. After review, set `status: active` and run contract and bronze differential gates.

An enabled source must reference an `active` contract; the registry refuses to load it otherwise. A `draft` is unreviewed inferred output and may only back a disabled source.

## Versioning and compatibility

Use semver for `version`. `schema_version` changes on any contract byte edit, even if semver does not. Only the MAJOR component is read by code: it authorizes a declared breaking change (see [The breaking-change workflow](#the-breaking-change-workflow)). The rest is the review convention:

- **MAJOR**: a change the table cannot absorb in place: a dropped, renamed or narrowed column, a type change outside the promotions, or a newly required column.
- **MINOR**: a new nullable column, a promotion allowed by `backward`, or a reordering of properties. Writes address columns by name, so a reordering issues no DDL.
- **PATCH**: metadata only, for example descriptions or `janus.enforcement`. Order-19 switched every checked-in contract to `strict` as a PATCH: the eighteen active contracts went from `1.0.0` to `1.0.1` and the twelve drafts from `0.1.0` to `0.1.1`.

`janus.compatibility` (`additive`, `backward`, `frozen`) decides what may change in place; `janus.enforcement` (`strict`, `lenient`) decides how data checks and the preflight fail. Both keys are required, so every contract states its modes. `janus contract draft` writes `additive` and `lenient`: upstream APIs add fields far more often than they change them, and `frozen` is one line to opt into. Every checked-in contract is reviewed to `strict` before it ships.

## Identity in every run record

`schema_version` is SHA-256 of contract bytes, computed once at registry load. `contract_id` and `contract_version` come from YAML. The fields travel through run-metadata JSON, lineage JSON and `metadata.runs`; JSON omits absent values and the table columns are nullable. Bronze output datasets carry the standard OpenLineage `SchemaDatasetFacet` from the plan. The [drift query](queries/observability/config-version-drift.sql) shows changes by source; see the [runs-table guide](queryable-observability.md) for v1 rows.

## Enforcement modes

`janus.enforcement` sets how a contract's data checks and its preflight fail. It never turns off the structural check, which runs for every batch in both modes. Every checked-in contract is `strict`.

| | `strict` | `lenient` |
|---|---|---|
| Structural check (names and types) | Before the write, every batch; a mismatch refuses the batch. | Same. |
| Null or blank values in a `required` column | Counted before the write; any count refuses the batch (`failure_stage: contract_check`). | Checked after the write by `data.required_fields`. |
| Malformed JSON, JSONL and CSV rows | Counted before the write; a count above `janus.maxMalformedRows` refuses the batch (`failure_stage: malformed_rows`). | Counted before the write, written anyway, reported as a warning check. |
| Batch persistence | Persisted `MEMORY_AND_DISK` when the contract has required columns or the handoff is JSON/CSV; unpersisted after the write. | Persisted for JSON/CSV handoffs only, to count malformed rows. |
| Preflight `refused` or `catalog_unavailable` | Fails the run before extraction (`failure_stage: contract_preflight`). | Logs `contract_preflight_warning` and continues. |

The data checks share one pass over the persisted batch: one aggregation sums the null indicators of every required column and the corrupt-record indicator together. Only when that count is non-zero does a second query collect up to five samples. `janus.maxMalformedRows` is the strict threshold, a non-negative integer string in `customProperties`, default `0`.


### Malformed JSON and CSV rows

JANUS reads JSON, JSONL and CSV handoffs in Spark `PERMISSIVE` mode with the reserved corrupt-record column `_janus_corrupt_record` appended to the generated schema. The validation report's `data.malformed_rows` check carries the count and up to five samples. Samples are scrubbed with the log sanitizer and limited to 500 characters. The column is removed before normalization and never reaches bronze. Parquet handoffs are typed by construction; the check reports `skipped` and nothing is persisted for it.

Spark's default `multiLine: true` for JSON treats a page as one document. A single drifted record can mark every row in that page as corrupt while preserving good parsed values; the count is rows Spark could not vouch for. JSONL and CSV count individual records. In CSV, a bad header can mark every subsequent data row corrupt.

## What the pre-write check compares

`check_frame_against_contract` in `quality/contract_checks.py` is pure. The batch's Spark schema arrives as JSON types through `schema_contracts.frame_columns_from_spark_schema`, the contract becomes the same JSON through the type vocabulary, and both sides are compared in vocabulary spellings.

- **Names.** Every declared top-level property must be in the batch (`missing_column`). Every batch column must be declared (`unexpected_column`), in every compatibility mode. A new upstream field is admitted by adding it to the contract, never by a batch carrying it. Struct children are compared by name too, at their dotted path.
- **Types.** Vocabulary spellings must be equal (`type_mismatch`, rendered `amount: type mismatch (contract long, frame string)`). Containers compare their element, key and value types.
- **Nullability is enforced on data, not on the schema.** Spark's JSON and CSV readers make every applied column nullable, so a nullable batch column the contract marks `required` is recorded as `nullability_relaxed` and is not a violation. `required` is enforced by the strict pre-write count or the lenient post-write check.
- **Column order** is not compared.
- **Normalization columns.** The check runs on the batch as read, before normalization adds the eight JANUS metadata columns; the post-write report ignores those columns.
- **`_janus_corrupt_record`** is reserved. A contract property with that name is a loader error. The column is allowed only while the pre-write pass runs; after it is dropped the check runs again, and a leaked column would be `corrupt_column_leaked`.

A refused batch commits nothing. In a multi-batch file handoff, batches before it stay committed; JANUS does not roll them back. The next successful full refresh replaces them.

Every enforcement failure names its stage:

| `failure_stage` | Error type | Raised when | What is committed |
|---|---|---|---|
| `contract_check` | `ContractViolationError` (`MissingContractError` for a plan with no contract) | The structural check fails, or a strict batch has null/blank required values. | Nothing for the refused batch. |
| `malformed_rows` | `MalformedRowsError` | A strict batch has more malformed rows than `janus.maxMalformedRows`. | Nothing for the refused batch. |
| `schema_evolution` | `SchemaEvolutionRefusedError` | The live table cannot become what the contract declares under its compatibility mode (checked by the writer, before any DDL). | Nothing for that batch. |
| `contract_preflight` | `ContractPreflightError` | A strict preflight is `refused` or `catalog_unavailable`. | Nothing: no request, no raw artifact, no Spark session. |

`failure_stage` is written to run-metadata JSON, lineage JSON, the CLI's executed-run summary and the `janusRun` OpenLineage facet. It is not a `metadata.runs` column; query `error_type` instead.

## The evolution matrix

`plan_schema_evolution` in `writers/evolution.py` is the one decision about what a bronze table may become. It is pure and takes the contract, the live columns in vocabulary spellings, the table's recorded contract version, the batch's write strategy and the batch index. The writer calls it for append, merge and full refresh alike, and the preflight calls it before extraction.

| Difference (contract vs live) | `additive` | `backward` | `frozen` |
|---|---|---|---|
| identical | noop | noop | noop |
| + nullable column | evolve (add) | evolve (add) | refused (frozen) |
| + required column | refused (newly_required) | refused | refused |
| `integer → long` (contract wider) | refused (retyped) | evolve (promote) | refused |
| `float → double` | refused | evolve | refused |
| `decimal(10,2) → decimal(18,2)` | refused | evolve | refused |
| `decimal(18,2) → decimal(18,4)` | refused (retyped) | refused (retyped) | refused |
| `long → integer` (narrowing) | refused (narrowed) | refused | refused |
| `string → long` | refused (retyped) | refused | refused |
| − column | refused (dropped) | refused | refused |
| rename (`label → title`) | refused (renamed_or_dropped) | refused | refused |
| live has an undeclared column | refused (undeclared_live_column) | refused | refused |
| any refusal + MAJOR bump + `replace_table` batch 1 | breaking_replace | breaking_replace | breaking_replace |
| any refusal + MAJOR bump + `insert` | refused | refused | refused |
| any refusal + MAJOR bump + `replace_table` batch 2 | refused | refused | refused |
| any refusal + no bump (`1.0.0 → 1.1.0`) | refused | refused | refused |

**Promotions.** The allowed promotions are exactly the ones verified on the pinned Spark 4.0.1 / Iceberg 1.10.1 pair: `integer → long`, `float → double`, and `decimal(p,s) → decimal(p',s)` with `p' > p` and the same scale. The engine refused narrowing, a scale change and `string → int`. Anything outside this set is a refusal, never a guess.

**What the writer does.**

- `noop`: writes the batch.
- `evolve`: runs `ALTER TABLE … ADD COLUMNS` for the nullable additions and `ALTER TABLE … ALTER COLUMN … TYPE` for each promotion, then writes. The `ALTER`s write metadata versions, not snapshots; the table UUID and history are retained, and the write adds one snapshot.
- `breaking_replace`: `REPLACE TABLE … AS SELECT` with `history_reset_reason` beginning `contract major version <recorded> -> <declared>`.
- `refused`: raises `SchemaEvolutionRefusedError` before any statement.

After every successful bronze statement the writer stamps the contract identity into the table properties, but only when the stamp differs. A changed `spark.partition_by` on a full refresh still takes `REPLACE TABLE` with `history_reset_reason` `partition spec changed: …`, independently of the contract.

**Limits**

- Nullability is not evolved. JANUS never issues `ALTER COLUMN … SET NOT NULL` or `DROP NOT NULL`; flipping `required` on an existing column changes no DDL, and the next batch's pre-write pass enforces it.
- Column order is not compared.
- Nested evolution is a refusal in this order. Iceberg can add a field inside a struct, but that is outside the verified set, so any nested difference is refused as `retyped` at the container's path. Verifying nested evolution is future work.
- A live column the contract does not declare is labelled `renamed_or_dropped` when a missing contract column has the same type, `undeclared_live_column` when it sits after the last declared column, and `dropped` otherwise. The labels are diagnostic; all three are refused alike.
- A table stamp that is not valid semver adds an `invalid_recorded_version` refusal and never authorizes a replacement.

**`schema_evolution` vocabulary.** The plan renders one string, written into the bronze write metadata and projected into the `metadata.runs` column `schema_evolution`:

| Rendered | Meaning |
|---|---|
| `none` | The table already matched; also every first write. |
| `added:a,b` | Nullable columns added in place. |
| `promoted:c(integer->long)` | Columns promoted in place, in vocabulary spellings. |
| `added:a;promoted:c(integer->long)` | Both. |
| `breaking_replace` | A declared MAJOR change replaced the table. |
| `refused:<kind>,<kind>` | The plan refused. A refused batch writes nothing, so this appears in logs and plans but never in a write result; the refusal is in `failure_reason`. |

## The breaking-change workflow

A change the matrix refuses needs a new MAJOR version and a full refresh. There is no override flag; the contract diff in git is the approval.

1. Change the contract and bump MAJOR, for example `1.0.1` to `2.0.0`.
2. Run the source as a full refresh (`extraction.mode: full_refresh`, `spark.write_mode: overwrite`). Before extraction, the preflight reports `will_evolve`; for an append or merge source it reports `refused`, because the bump authorizes only the first batch of a full refresh.
3. The writer runs `REPLACE TABLE … AS SELECT`, records `history_reset_reason: contract major version 1 -> 2 …` in the write metadata and stamps `2.0.0` on the table.

This holds under every compatibility mode, `frozen` included. The mode governs in-place evolution; a MAJOR bump declares a new table.

On the pinned Iceberg 1.10.1, `REPLACE TABLE` is a replace transaction, not drop-and-recreate. The UUID and the snapshot log survive and the old snapshot stays readable with `VERSION AS OF`, but it is no longer an ancestor of the current state: `rollback_to_snapshot` refuses it. What a breaking change loses is lineage.

**The table stamp.** Read it with `SHOW TBLPROPERTIES <table>` in Spark, or `table.properties` in PyIceberg:

| Property | Value |
|---|---|
| `janus.contract_id` | Contract `id` that last wrote the table. |
| `janus.contract_version` | Its `version`; the recorded MAJOR comes from here. |
| `janus.schema_version` | SHA-256 of that contract's bytes. |

**Unstamped tables.** A table written before the implementation of Data Contracts has no stamp and counts as major `0`. Under an active contract (`1.x`), the first full refresh that would otherwise be refused is therefore accepted once as a declared breaking change; after it the stamp is set, and the next breaking change needs another MAJOR bump. A draft (`0.x`) contract never authorizes a replacement, because `0` is not greater than `0`. The previous major has to live with the table: both Spark and PyIceberg can read it, and it does not depend on best-effort run emission.

## Reading a failed preflight or `malformed_rows` check

The preflight runs in both entry points, live `--execute` and `--ingest-raw-to-bronze`, after the run's START is recorded and before extraction or raw rehydration. It reads the live table through PyIceberg with the catalog derivation `metadata.runs` also uses, starts no Spark session, and is bounded by a five-second budget. A bronze target that is not Iceberg is `ok` without a catalog call. The throwaway `local-hadoop` profile cannot be represented in PyIceberg, so strict sources fail their preflight there with `catalog_unavailable`; use `local`.

| Outcome | Meaning | `strict` | `lenient` |
|---|---|---|---|
| `ok` | The live table matches the contract. | Continues. | Continues. |
| `will_evolve` | The writer will add or promote columns, or replace the table on a declared MAJOR full refresh. The reason carries the rendered plan. | Continues. | Continues. |
| `refused` | The table differs outside the compatibility mode. The reason lists `column: kind`. | Fails before extraction. | Warns; the write later fails with `schema_evolution`. |
| `table_missing` | No table yet: the run is a first write. | Continues. | Continues. |
| `catalog_unavailable` | The catalog raised, or did not answer within the budget. The reason is the exception type only, never a URI or credential. | Fails before extraction. | Warns and continues. |

Where each surface shows it:

| Surface | What to read |
|---|---|
| Structured log | `contract_preflight_finished` (`outcome`, `reason`, `duration_seconds`), then `contract_preflight_ok`, `contract_preflight_warning`, or `source_execution_failed` with `failure_stage`. |
| Run-metadata and lineage JSON | `status`, `failure_stage`, `error_type`, `failure_reason`, and `run_attributes.contract_preflight_outcome`. |
| Validation JSON | `data.schema_expectations` (`mismatches`, `nullability_relaxed`, contract identity, `batches`), `data.required_fields`, and `data.malformed_rows` (`count`, `threshold`, `samples`, `truncated`, `batch_index`, plus `severity: warning` under `lenient`). |
| `metadata.runs` | `contract_preflight_outcome`, `schema_evolution`, `malformed_rows`, `error_type`, `failure_reason`, `validation_report_path`. |
| OpenLineage `janusRun` facet | `failure_stage`, `contract_preflight_outcome`, `schema_evolution`, `malformed_rows`. |

**Example 1: a refused preflight.** A strict `additive` contract declares `amount long`; the live table has `amount int`. `additive` does not promote, so the plan refuses. The structured events and the run record carry these fields:

```text
contract_preflight_finished outcome=refused reason="amount: retyped"
source_execution_failed error_type=ContractPreflightError failure_stage=contract_preflight
failure_reason: <contract id> v1.0.1: preflight refused — amount: retyped
```

No request was sent, no raw artifact written and no Spark session started. The run metadata has `run_attributes.contract_preflight_outcome: refused`; the `metadata.runs` row has `contract_preflight_outcome = refused` and `error_type = ContractPreflightError`, with `schema_evolution` and `malformed_rows` `NULL` because nothing was read or written. Fix it in the contract: set `janus.compatibility: backward` to allow the promotion in place, or bump MAJOR and run a full refresh.

**Example 2: a strict malformed batch.** A JSONL page carries `"amount": "abc"` for a `long` column. The batch is refused before the write with `failure_stage: malformed_rows` and `failure_reason: 1 malformed rows exceed max_malformed_rows 0 (batch 1/1)`. The validation JSON's `data.malformed_rows` check is `failed` with `count: 1`, `threshold: 0` and the scrubbed raw record in `samples`; the table's snapshot count is unchanged, and `metadata.runs` shows `malformed_rows = 1`, `error_type = MalformedRowsError`. Fix the upstream data, correct the contract type if the source legitimately changed (a MAJOR bump if the table must change), raise `janus.maxMalformedRows`, or switch to `lenient`, which writes the parsed values with nulls and reports the count as a warning.

**Example 3: a contract-check failure on batch 2 of a file source.** A file handoff with three CSV members is materialized one batch per member. Batch 1 matches and is written; batch 2 lacks a declared column. The run fails with `failure_stage: contract_check` and `failure_reason` beginning `<contract id> v<version>: frame does not match the contract (batch 2/3)`. The validation JSON's `data.schema_expectations` check is `failed`, its message prefixed `batch 2/3:`, and `details.batches` counts the batches checked. Batch 1's write result stays in the run metadata's materialized outputs, because JANUS does not roll a committed batch back; batch 3 was never read. On a full refresh the table holds only batch 1's rows until the next successful run replaces it.

For the history of these signals, run the [schema drift query](queries/observability/schema-drift-by-source.sql). It returns, per source, every run whose preflight found work or trouble, whose table evolved, that counted malformed rows, that changed contract bytes, or that stopped on one of the four enforcement errors.

## Guardrails

| Test | What it refuses |
|---|---|
| `tests/unit/quality/test_contract_checks.py` | A structural check that compares Spark spellings, misses a mismatch kind, treats nullability as a violation, or imports an engine at any depth. |
| `tests/unit/runtime/test_pre_write_gate_wiring.py` | A failed check that still reaches the writer, a check outside the read-to-normalization window, a contract-less plan that writes, and a live or replay failure recorded without `failure_stage`. |
| `tests/integration/contracts/test_pre_write_gate.py` | A structural mismatch that commits a snapshot (AC-1), a failing first batch that creates the table, and a positional append (AC-3). |
| `tests/integration/contracts/test_malformed_rows.py` | Type drift absorbed as nulls under `strict`, a corrupt-record column in a bronze table the suite writes, and a dropped pin on either Spark corrupt-record rule (AC-2). |
| `tests/unit/writers/test_evolution_plan.py` | Any matrix cell drifting, a promotion outside the verified set, and an engine import in the planner. |
| `tests/unit/writers/test_schema_ddl.py` | An unquoted identifier or literal in a DDL builder, and a projection out of target-table order. |
| `tests/integration/contracts/test_evolution_matrix.py` | Any of the 57 cells behaving differently on a real Iceberg table for append, merge or full refresh; a promotion that loses history; a breaking replace that keeps rollback (AC-4). |
| `tests/unit/runtime/test_contract_preflight.py` | A preflight outcome outside the five, a hang past the budget, a catalog URI in a reason, an engine loaded by importing the module, and a strict refusal that reaches extraction or starts a session. |
| `tests/integration/catalog_commits/test_contract_preflight_real_catalog.py` | A catalog derivation that cannot read the table Spark wrote, a stamp that is not read, and a strict refusal that sends a request (AC-5; a required CI gate). |
| `tests/unit/toolchain/test_no_write_without_contract_check.py` | A bronze write outside the materializer or before the pass, a `spark.sql` statement not rendered by a builder, and any `SELECT *` insert. Package-scoped over `janus/writers` and `janus/runtime`, with anchors and detector meta-tests. |
| `tests/unit/models/test_schema_config_contract_only.py` | A tolerated `schema.mode`, `schema.path` or `quality.allow_schema_evolution`, and the active-contract rule back in the policy (AC-6). |
| `tests/unit/models/test_phase_policy_boundary.py` | A `status == "active"` decision anywhere but `registry/loader.py::_require_active_contract`. |
| `tests/unit/observability/test_runs_table_contract.py`, `test_run_record_projection.py`, `test_openlineage_facets.py`, `test_example_queries_and_docs.py` | Runs-table schema or projection drift for the three enforcement columns, a facet without the four fields, and a published query or register row out of step with the schema. |
