# Spark Runtime, Storage, Readers, Writers, and Normalization Base

The short version is: JANUS now has a shared I/O layer.

Up to this step, the project had a planner, runtime configuration, and metadata persistence, but it still did not have one common place to answer the practical questions every strategy will eventually hit.

Where does a raw artifact go?
How do we resolve that path consistently across local and cluster-shaped environments?
Which code is responsible for preserving a raw payload, and which code is responsible for writing a parsed dataset?
What is the minimum normalization behavior we want before source-specific logic starts to appear?

This step fills that gap.

The result is a small shared layer for Spark bootstrap, storage resolution, raw artifact persistence, Spark-backed reads and writes, and a normalization base that adds execution metadata without smuggling source behavior into the common code.

## What was added

- `src/janus/utils/storage.py` now defines the storage contract for `raw`, `bronze`, and `metadata` zones.
- `src/janus/utils/spark.py` now provides a shared Spark runtime bootstrap built on the existing environment configuration.
- `src/janus/readers/spark.py` now provides a generic Spark dataset reader for configured outputs and extracted artifacts.
- `src/janus/readers/__init__.py` now exposes the reader API for downstream imports.
- `src/janus/writers/raw.py` now provides deterministic file-oriented persistence for raw artifacts.
- `src/janus/writers/spark.py` now provides generic Spark DataFrame writes for parsed outputs.
- `src/janus/writers/__init__.py` now exposes the writer API for downstream imports.
- `src/janus/normalizers/base.py` now provides the minimum shared normalization behavior for execution metadata columns.
- `src/janus/normalizers/__init__.py` now exposes the normalization API for downstream imports.
- `tests/unit/writers/test_storage_and_writers.py` covers storage resolution and raw artifact persistence, plus Spark-backed bronze writing when Spark is available.
- `tests/unit/readers/test_spark_dataset_reader.py` covers reading extracted raw artifacts through the generic Spark reader when Spark is available.
- `tests/unit/readers/test_spark_reader_options.py` covers the reader-side JSON option defaults without requiring Spark to be installed.

## The storage contract

The storage layer is intentionally small.

`StorageLayout` takes the already-validated environment configuration and turns it into absolute roots for the three JANUS zones:

- `raw`
- `bronze`
- `metadata`

From there, it resolves configured output targets into deterministic runtime paths.

One detail matters here: source configs still describe their outputs in the checked-in contract, but the runtime is free to relocate the actual storage roots through environment configuration. That keeps planning metadata stable while still allowing local and cluster-like runs to land in different physical locations.

In practice, that means a source output like `data/raw/example/source_a` can resolve under a different configured raw root without forcing each strategy to know how those path substitutions work.

## The raw writer

The raw layer now has its own writer because raw persistence is not the same thing as writing a parsed dataset.

`RawArtifactWriter` is file-oriented on purpose.

It supports four write shapes:

- `write_bytes(...)`
- `write_text(...)`
- `write_json(...)`
- `write_json_lines(...)`

The intended default for raw extraction is preservation, not interpretation.

If a strategy downloads a CSV, ZIP, XLSX, or an API response body and wants to keep the raw payload exactly as received, the expected entry point is `write_bytes(...)` or `write_text(...)` depending on what the extractor already has in hand.

The JSON helpers exist for the cases where the extraction code already has structured records in memory and wants the raw zone to hold a stable serialized representation. That is useful, but it is not byte-for-byte preservation of an upstream payload. The writer makes that distinction explicit instead of pretending all raw writes are equivalent.

Every raw write resolves a deterministic path, creates parent directories, persists the payload, computes a checksum, and returns both:

- an `ExtractedArtifact` for downstream normalization handoff;
- a `WriteResult` for metadata and lineage work.

That keeps the raw layer operationally useful without turning it into a parser.

## The Spark reader

The shared reader is intentionally generic.

`SparkDatasetReader` reads:

- explicit artifact paths;
- an `ExtractionResult` artifact set;
- a configured JANUS output zone.

The supported formats are the formats we need for the early source work:

- `json`
- `jsonl`
- `csv`
- `parquet`
- `text`
- `binary`

The reader can also receive a Spark schema object from the runtime. During normal execution, JANUS generates a reader schema from `schema.contract` when the normalization handoff format matches `spark.input_format`. That is what lets headerless CSV file sources land with contract-defined column names instead of anonymous `_c0`, `_c1`, and so on.

One behavior worth calling out is JSON handling.

The raw writer serializes `json` payloads as ordinary multi-line JSON documents, while `jsonl` remains line-oriented. The reader now reflects that distinction instead of treating both the same way. By default:

- `json` reads use `multiLine=true`;
- `jsonl` reads stay line-oriented;
- explicit options can still override the defaults when a source needs different behavior.

That keeps the common reader flexible without making each future strategy rediscover the same format edge cases.

## The Spark writer

The Spark-side writer is the complement to the raw writer.

`SparkDatasetWriter` assumes the input is already a DataFrame and is ready to be written as a structured dataset rather than as a raw payload.

It resolves the output path through the shared storage layout, applies the configured or explicit format, applies the configured write mode, and respects configured partition columns. For bronze writes it can also apply the source-configured repartitioning before persistence.

This is the writer later strategy implementations are expected to use once raw artifacts have been read and normalized into Spark.

That separation matters.

The raw writer preserves payloads.
The Spark writer persists datasets.
They are related, but they are not interchangeable.

## The normalization base

The normalization layer introduced here is deliberately minimal.

It does not flatten a source payload.
It does not rename business columns.
It does not carry source-specific assumptions.

`BaseNormalizer` adds execution and ingestion metadata that every downstream bronze dataset is likely to need anyway:

- run id;
- source id and source name;
- environment;
- strategy family and variant;
- ingestion timestamp;
- ingestion date.

After adding those columns, it reorders the DataFrame so the metadata fields stay at the front.

That gives later source-specific normalization work a stable starting point without pushing business logic into the shared layer too early.

## What the tests lock down

The tests for this step focus on the contracts and behaviors that later strategy work will depend on.

They cover:

- resolving raw, bronze, and metadata output paths through the shared storage layout;
- writing a raw JSON artifact to the resolved raw path and capturing its checksum in metadata;
- defaulting JSON reads to multi-line document mode while keeping JSONL line-oriented;
- allowing explicit Spark read options to override the default JSON reader behavior;
- applying explicit schemas when reading headerless CSV artifacts;
- reading extracted raw artifacts through the generic reader when Spark is available;
- writing normalized bronze output through the generic Spark writer when Spark is available.

The focused verification for this step passed with:

- `python -m py_compile ...`
- `python -m ruff check src/janus/utils/storage.py src/janus/utils/spark.py src/janus/readers src/janus/writers src/janus/normalizers tests/unit/readers tests/unit/writers`
- `python -m pytest tests/unit/readers/test_spark_reader_options.py tests/unit/readers/test_spark_dataset_reader.py tests/unit/writers/test_storage_and_writers.py`

In the current shell, the Spark-backed tests are skipped when `pyspark` is not installed. The non-Spark coverage remains active so the shared option and storage behaviors still stay protected even in a lighter development environment.

## What this step still does not do

This step still does not:

- implement the concrete API, file, or catalog strategies;
- decide how one specific source should paginate, flatten, or clean its records;
- move checkpoint logic into the reader or writer layers;
- add source-specific parsing shortcuts to the generic I/O code;
- replace a source-local normalizer when one source really needs special treatment.

That boundary is intentional.

The shared layer now gives later strategy tasks a common way to:

- persist raw artifacts;
- read them back through Spark;
- add minimal execution metadata;
- write structured outputs;
- do all of that through deterministic paths and environment-driven storage.

That is enough foundation for the next source-family implementations without turning the common I/O layer into a pile of source exceptions.
