# Data contracts

## What a contract is and where it lives

Each bronze table has one ODCS v3.2.0 YAML contract at `conf/contracts/<domain>/<table>.yaml`. Co-writers share that file. Source YAML points to it with `schema.contract`. Git records declared intent; the Iceberg catalog records actual table schema and history.

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
| `schema[0].properties[]` | Bronze field metadata: `name`, `businessName`, `description`, derived `logicalType`, JANUS `physicalType`, `required`, `unique`, `primaryKey`, `classification`, and `customProperties` entries for `sourceField` and `sourceFormat`. A `struct` may contain nested `properties`; an `array` may contain `items`. | `name` and `physicalType`; other values have JANUS defaults |
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
    value: lenient
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

The CLI writes `status: draft` and `janus.draftedFrom`; it never runs inside `run` or `run-all`. Review actual samples, column names and order, nested types, nullability, descriptions, classification, ownership, and agreement with `quality.required_fields` / `unique_fields`. Inference can mistake all-null fields, numeric-looking text, dates and sparse data. Keep inferred order (alphabetical for JSON handoffs, source order for CSV). Set `required: true` only when observed null count is zero and a reviewer confirms the rule. Reordering needs a new version. After review, set `status: active` and run contract and bronze differential gates.


## Versioning and compatibility

Use semver for `version`; bump it when declared fields or meaning change. `schema_version` changes on any contract byte edit, even if semver does not. `janus.enforcement` governs the pre-write malformed-row check for JSON and CSV handoffs; `janus.compatibility` (`additive`, `backward`, `frozen`) is recorded for evolution governance.

## Malformed JSON and CSV rows

JANUS reads JSON, JSONL and CSV handoffs in Spark `PERMISSIVE` mode with a reserved corrupt-record
column. A `strict` contract refuses a batch before the bronze write when the malformed count
exceeds `janus.maxMalformedRows` (default `0`). A `lenient` contract writes Spark's parsed values,
including any nulls from failed type parses, and records the count as a warning. The validation
report's `data.malformed_rows` check carries the count and up to five scrubbed samples. Samples are
limited to 500 characters. The reader column is removed before normalization and never belongs to
the bronze table. Parquet handoffs skip this check. A source can explicitly set `mode: FAILFAST`
in its Spark read options to make Spark reject malformed input at read time.

Spark's default `multiLine: true` for JSON treats a page as one document. A single drifted record
can mark every row in that page as corrupt while preserving good parsed values; the count is rows
Spark could not vouch for. JSONL and CSV count individual records. In CSV, a bad header can mark
every subsequent data row corrupt.

## Identity in every run record

`schema_version` is SHA-256 of contract bytes, computed once at registry load. `contract_id` and `contract_version` come from YAML. The fields travel through run-metadata JSON, lineage JSON and `metadata.runs`; JSON omits absent values and the table columns are nullable. Bronze output datasets carry the standard OpenLineage `SchemaDatasetFacet` from the plan. The [drift query](queries/observability/config-version-drift.sql) shows changes by source; see the [runs-table guide](queryable-observability.md) for v1 rows.

