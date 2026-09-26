# OpenLineage mapping contract

For transport configuration, event-file locations, runs-table queries, batch/replay semantics,
and troubleshooting, see [Queryable observability operations](queryable-observability.md).

JANUS maps run lifecycle records to OpenLineage **core specification 2-0-2**. Every event
sets:

```text
schemaURL = https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent
```

The exact published schema is vendored at
`tests/fixtures/openlineage/OpenLineage-2-0-2.json`. Its recorded SHA-256 is
`69f68bee00b9beac88a87059c0102410e7bb05f3f43c46d02a0409831eceb0d2`.
Tests check the vendored bytes before validating START, COMPLETE, and FAIL events. Schema
updates are therefore deliberate diffs rather than an unpinned dependency on “latest”.

## Identity and lifecycle

- OpenLineage requires a UUID run id. JANUS derives UUIDv5 from the readable `run_id`,
  under a UUIDv5 namespace derived from the JANUS repository URL. The same JANUS id always
  has the same OpenLineage id, including across processes and replays. The readable id also
  remains in the `janusRun` facet.
- A job is a source, not an execution. Its namespace is `janus://<environment>` and its
  name is `source_id`, so it stays stable across runs and separates environments.
- `running`, `succeeded`, and `failed` map to `START`, `COMPLETE`, and `FAIL`. JANUS does not
  emit `ABORT`, `RUNNING`, or `OTHER`.
- START uses the persisted `RunMetadata.started_at`; terminal events use
  `LineageRecord.emitted_at`. The mapper never reads a clock.

## Datasets and facets

Iceberg input datasets come only from the validated `SourceRegistry.graph`. The mapper does
not parse strategy metadata, configs, or warehouse state to rediscover dependencies.
Iceberg inputs and bronze outputs share a namespace derived from the configured catalog
name and warehouse; their names are the validated table identifiers. This makes a
consumer input identity match its producer's bronze output identity.

Raw artifacts are output datasets too. Local paths follow the OpenLineage `file` namespace
convention; object-store paths use their scheme and authority. A raw artifact and raw
write result for the same path are emitted once, while checksum and write details remain
in the custom facet. Configured-but-unwritten targets are not presented as outputs.

Standard facets carry job documentation, source-code location and config version, job
type, failure messages, output row counts, and the bronze schema. The SchemaDatasetFacet
is rendered from the DataContract already carried on the execution plan, with no file or
catalog reads during emission. It appears only on bronze outputs in terminal COMPLETE and
FAIL events; raw artifacts and START events have no schema facet. Field types use JANUS's
engine-neutral vocabulary (for example timestamptz, decimal(18,2), and struct), and nested
struct properties are represented as nested fields. Job documentation uses the contract
purpose when one is present and falls back to the source name otherwise. The facet schema
is pinned at tests/fixtures/openlineage/SchemaDatasetFacet-1-1-1.json, with its SHA-256
recorded beside it. Quality results remain run-scoped because JANUS checks may describe
config, input data, or output rather than one dataset.

All JANUS-specific data lives in one `janusRun` run facet. Its versioned schema is
`docs/schemas/openlineage/JanusRunFacet.json`. It retains every `LineageRecord` field plus
the checkpoint decision, quality summary, metadata-zone evidence paths, start time, and
declared input provenance. The mapper's module-level field table is intentionally exhaustive;
a new `LineageRecord` field fails the test suite until its mapping decision is recorded.

The OpenLineage parent facet is not emitted. `pipeline_run_id` is available, but JANUS does
not persist the parent job namespace and name; inventing them would produce a false parent.
Pipeline correlation instead remains explicit in `janusRun.run_attributes`.

## Client decision

JANUS does not depend on `openlineage-python`. The event envelope is small and fully
determined by existing immutable records, while adopting the client would add another
runtime dependency and HTTP stack. Direct schema validation of JANUS's own serialized
payload is stronger evidence for this mapping than validating client-created objects. The
JSON Schema validator is consequently a dev-only dependency.

This is a revisitable rejection. If OpenLineage specification drift makes maintaining the
envelope costly, the client can replace the mapper behind the transport seam without
changing run records or observer persistence.
