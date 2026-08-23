# JANUS Source Onboarding Guide

This guide is for adding a new source without turning JANUS into a pile of exceptions.

The short version is:

1. prove the source belongs in JANUS;
2. classify it into an existing strategy family and variant;
3. express as much as possible in source config;
4. reuse the family runtime;
5. add a source hook only when there is a clear, narrow mismatch;
6. add tests and documentation for anything reusable.

## Before you add a source

Only onboard a source if it is:

- federal;
- clearly public and legally reusable;
- documented enough to access responsibly;
- useful for evaluating JANUS as a framework;
- representative of a real ingestion pattern.

If the source is private, protected, anti-bot guarded, or outside the federal scope, it does not belong in phase 1.

## Step 1: classify the source first

Choose the source family before you write code.

### Use `api` when:

- the source exposes records through HTTP requests;
- pagination, rate limits, headers, or checkpoint query parameters matter;
- the main unit of extraction is an API response payload.

Supported variants today:

- `page_number_api`
- `offset_api`
- `cursor_api`
- `date_window_api`

### Use `file` when:

- the main unit of extraction is a file or package;
- discovery, version selection, checksums, or archive extraction matter;
- the important handoff is a persisted file artifact rather than an API record page.

Supported variants today:

- `static_file`
- `versioned_file`
- `archive_package`

### Use `catalog` when:

- the source is a metadata registry;
- the value is in datasets, organizations, groups, and resources;
- JANUS should ingest metadata about resources, not the resource payloads themselves.

Supported variants today:

- `metadata_catalog`
- `resource_catalog`

If you cannot classify the source cleanly, stop there. Do not start coding until the classification is clear.

## Step 2: start from the existing contract

Use `conf/sources/example/example_source.yaml` as the template for new source definitions.

Registry discovery is recursive, so files like `conf/sources/ibge/sidra.yaml` are valid. When one provider file needs several endpoint-backed source contracts, declare them under a top-level `sources:` list.

There are two valid top-level shapes:

1. One source per file.

```yaml
source_id: transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh
name: Portal da Transparencia - Servidores agregados por orgao
...
```

2. Several source entries in one provider file.

```yaml
sources:
  - source_id: ibge_pib_brasil
    name: IBGE - PIB nacional a precos correntes
    ...
  - source_id: ibge_agro_abacaxi_pronaf
    name: IBGE - Quantidade produzida de abacaxi
    ...
```

Important detail: `sources:` must contain a YAML list. Each entry starts with `-`. If you repeat keys like `source_id`, `access`, or `outputs` under one mapping instead of using a list item, the file is invalid YAML.

Every source should define:

- identity fields such as `source_id`, `name`, `owner`, `domain`, and `enabled`;
- family metadata such as `source_type`, `strategy`, and `strategy_variant`;
- the `access` block for endpoints, paths, auth, pagination, rate limits, and formats;
- the `extraction` block for refresh mode, retry policy, dead-letter budget, and checkpoint semantics;
- the `schema` block;
- the `spark` block;
- the `outputs` block for `raw`, `bronze`, and `metadata`;
- the `quality` block.

Keep `source_type` and `strategy` aligned. In the current JANUS design, family and strategy are the same concept.

### The default posture, and what to do when your source does not fit it

JANUS validates every source against a **phase-scope policy**: public sources, `federation_level: federal`,
and `strategy` equal to `source_type`. A config that breaks any of these is rejected at load time:

```
Invalid source config: conf/sources/example/my_source.yaml
- public_access: must be true because JANUS only supports public federal sources in phase 1
```

**That is a policy decision, not a bug.** It says what JANUS has chosen to onboard so far, not what the code
can do. If you believe your source should be an exception — a state-level dataset, a credentialed feed — the
answer is not to edit `models/source_config.py`. The rules live in `src/janus/models/config/policy.py` as a
swappable `ValidationPolicy`, and broadening them is a scope decision someone makes deliberately, with its
own order. Take it there.

### Adding a new variant to an existing family

One edit. Add the name to `SUPPORTED_STRATEGY_VARIANTS` in `src/janus/models/config/constants.py`:

```python
"api": frozenset(
    {"cursor_api", "date_window_api", "offset_api", "page_number_api", "keyset_api"}
),
```

Then implement the variant's behaviour in the family strategy. The planner picks it up automatically — it
builds its dispatch table from the registry, so there is no second list to keep in step.

### Adding a new family

Two edits, and the second is easy to forget:

1. the registry entry in `constants.py`;
2. the implementation binding in `StrategyCatalog.with_defaults` (`src/janus/planner/core.py`).

`tests/unit/planner/test_strategy_registry_drift.py` fails if you do one and not the other, in either
direction — a family with no implementation, or an implementation for a family nobody registered.

### What The Main Blocks Currently Support

The registry contract is intentionally small. The most important current options are:

### `access`

- API and catalog sources require `access.base_url` or `access.url`.
- File sources require `access.url`, `access.path`, or `access.discovery_pattern`.
- `access.format` must match the payload you expect JANUS to fetch, such as `json`, `jsonl`, `csv`, `parquet`, `text`, or `binary`.
- `access.method` must be a supported HTTP method such as `GET` or `POST`.
- `access.auth.type` controls auth shape. Supported values today are `none`, `header_token`, `bearer_token`, `query_token`, and `basic`.
- `access.pagination.type` must match the chosen family variant when pagination is used: `page_number`, `offset`, `cursor`, or `none`.
- `access.pagination.past_end_status_codes` is an optional 4xx list meaning "you asked past the last page", defaulting to `[404, 416]`. It only matters when `concurrency > 1`.
- `access.pagination.total_count_field` is an optional dotted path to a total-record count, used to cap concurrent look-ahead exactly.
- `access.rate_limit` carries `requests_per_minute`, optional `backoff_seconds`, and optional `concurrency`.
- `access.rate_limit.concurrency` above `1` is an API-family capability with a validated precondition — see [Concurrent pagination](#concurrent-pagination).
- `access.params` is the home for static literal request parameters.
- `access.request_inputs` is an optional API/catalog block for bounded runtime request contexts before pagination starts.
- `access.parameter_bindings` is an optional API/catalog block for request parameters resolved from the current request input or checkpoint state.
- `access.link_resolver` is a file-source option for remote URL discovery. Supported values are `auto`, `direct`, `html_links`, and `nextcloud_webdav`.
- `access.remote_file_pattern` filters files discovered from a remote URL before download.
- `access.file_pattern` filters local file discovery and archive members before Spark handoff.

### API and Catalog Request Shaping

For API and catalog sources, keep the request contract split by responsibility:

- `access.params` holds fixed literals that should be sent on every request.
- `access.request_inputs` defines the bounded outer request contexts JANUS should resolve before pagination starts.
- `access.parameter_bindings` maps request parameter names to runtime values from the current request input or the current checkpoint.
- hooks remain the escape hatch when the request still needs source-specific preparation after those three layers are exhausted.

If `access.request_inputs` is omitted, JANUS keeps the existing one-stream behavior with `type: none`.

Supported request-input types:

- `none`
- `date_window` with `start`, `end`, and `step`
- `iceberg_rows` with `namespace`, `table_name`, `columns`, and optional `distinct`
- `combined` with an `inputs:` list of two or more `date_window` or `iceberg_rows` entries

Supported binding sources:

- `checkpoint_value`
- `request_input.window_start`
- `request_input.window_end`
- `request_input.<field>` for fields from `iceberg_rows` columns, or from any sub-input when `type: combined`

Keep the boundaries sharp:

- use `access.params` for literals such as `situacao: TODAS`;
- use `access.parameter_bindings` for values such as `mesAno`, `dataIdaDe`, `dataIdaAte`, or a detail identifier that changes per request input;
- use a hook only when the request shape is still irregular, such as custom signing, request-body generation, or a one-off cursor rule.

Request inputs do not replace pagination. They sit outside it. Choose the strategy variant and `access.pagination` that match the request loop, then add `access.request_inputs` only when the endpoint also needs a bounded outer context.

### Example: Monthly Parameter Binding

This pattern is a good fit when the endpoint still paginates normally, but also expects one bounded date context per request stream.

```yaml
strategy: api
strategy_variant: page_number_api

access:
  params:
    situacao: TODAS
  request_inputs:
    type: date_window
    start: 2025-01-01
    end: 2025-03-31
    step: month
  parameter_bindings:
    mesAno:
      from: request_input.window_end
      format: "%Y%m"
    dataIdaDe:
      from: request_input.window_start
      format: "%Y-%m-%d"
    dataIdaAte:
      from: request_input.window_end
      format: "%Y-%m-%d"
  pagination:
    type: page_number
    page_param: pagina
    size_param: tamanhoPagina
    page_size: 50
```

JANUS resolves one monthly window at a time, binds the three runtime parameters for that window, and then runs the normal page loop inside each request stream.

### Example: Iceberg-Driven Identifier Iteration

This pattern is a good fit for detail endpoints that depend on identifiers JANUS already wrote to Bronze Iceberg.

```yaml
strategy: api
strategy_variant: page_number_api

access:
  params:
    situacao: ATIVO
  request_inputs:
    type: iceberg_rows
    namespace: bronze_transparencia
    table_name: orgaos__siape
    columns:
      orgao_codigo: codOrgaoExercicioSiape
    distinct: true
  parameter_bindings:
    codigoOrgao:
      from: request_input.orgao_codigo
  pagination:
    type: page_number
    page_param: pagina
    size_param: tamanhoPagina
    page_size: 50
```

JANUS loads one projected row per distinct `codOrgaoExercicioSiape` value, binds `codigoOrgao` from that request input, and keeps the shared pagination, retry, raw-persistence, and metadata flow unchanged inside each request stream.

### Example: Combined Identifier and Date Window

This pattern is a good fit when an endpoint requires both an upstream entity identifier and a bounded date context per request stream — neither input alone is sufficient.

```yaml
strategy: api
strategy_variant: page_number_api

access:
  request_inputs:
    type: combined
    inputs:
      - type: iceberg_rows
        namespace: bronze_transparencia
        table_name: orgaos
        columns:
          orgao_codigo: codigo
        distinct: true
      - type: date_window
        start: 2025-01-01
        end: 2025-12-31
        step: month
  parameter_bindings:
    codigoOrgao:
      from: request_input.orgao_codigo
    dataInicio:
      from: request_input.window_start
      format: "%Y-%m-%d"
    dataFinal:
      from: request_input.window_end
      format: "%Y-%m-%d"
  pagination:
    type: page_number
    page_param: pagina
    size_param: tamanhoPagina
    page_size: 50
```

JANUS computes the Cartesian product of the two input streams and runs one request stream per combination. With 5 org codes and 12 monthly windows, that produces 60 request streams, each bound with its own `codigoOrgao`, `dataInicio`, and `dataFinal`.

### Concurrent pagination

`access.rate_limit.concurrency` is the only knob that makes JANUS issue more than one request at a time. It is a **capability with a precondition**, not a throughput dial you can turn up on any source. Read this section before setting it above `1`.

#### What concurrency actually does

With `concurrency: N` (N > 1), the API strategy submits up to N pages before the current one has answered. It has no way of knowing whether page 4 exists while page 3 is still in flight, so it **predicts a full page each time**: "page 3 returned `page_size` records, therefore page 4 probably exists." Results are still committed in request order, so artifacts and records are identical to a sequential run — only the request timing differs.

That prediction is the whole point and the whole risk. Every request beyond the first one of a request stream is a guess.

#### When it is allowed

| Family | `concurrency > 1` | Behavior |
|---|---|---|
| `api` + `page_number` or `offset` | allowed | speculative fan-out, as described here |
| `api` + `cursor` | **rejected at config load** | the next request is only knowable from the current response, so there is nothing to predict |
| `api` + `none` | **rejected at config load** | a single page cannot be paginated ahead |
| `catalog` | accepted, **inert** | the catalog strategy paginates sequentially and never reads the key |
| `file` | accepted, **inert** | parallel downloads are unimplemented; the value only reaches a log field |

The `api` rules are enforced by `SourceConfig.from_mapping`, so a `cursor_api` source with `concurrency: 2` fails to load with an issue at `access.rate_limit.concurrency` that names the offending pagination type. The `catalog` and `file` rows are accepted for backward compatibility only — setting the key there buys nothing, so leave it at `1` rather than implying throughput the runtime does not deliver.

#### The precondition, stated plainly

Because JANUS speculates, it *will* eventually request a page past the last one. The API must answer that request in one of exactly two ways:

1. an **empty `200`** (or a short page), or
2. one of the statuses in `access.pagination.past_end_status_codes` — by default `404` and `416`.

Both are read as a clean end of stream: no artifact is written, no counter moves, the request stream finishes normally, and extraction continues with the next request input.

Anything else is a genuine failure and will dead-letter the request input, by design:

- a `500`, a timeout, or a redirect loop past the end;
- a `200` that repeats the last page forever (JANUS cannot detect this and will loop until the page-number space is exhausted — do not enable concurrency on such an API);
- a past-end status on the **first** request of a request stream — that request was never speculative, so a `404` there means a broken endpoint, not an ended stream;
- a past-end status that a later, already-resolved page contradicts by returning records. That raises `ApiPastEndConflictError` naming the conflicting request index, rather than silently truncating the dataset.

`past_end_status_codes` must be 4xx and may not include `408` or `429` — a status cannot mean both "retry me" and "the stream ended". An explicit empty list (`past_end_status_codes: []`) opts out entirely and restores raise-on-4xx behavior.

#### How to check a new API in three commands

Verify the precondition before you raise `concurrency`. Export your token once, then run three requests: the **first** page, the **last** page with data, and **one past it**. Only the third answer decides whether concurrency is safe.

```bash
export TOKEN="…"                 # never paste the token into the YAML or a commit
BASE=https://api.portaldatransparencia.gov.br/api-de-dados/orgaos-siape
H="chave-api-dados: $TOKEN"

# 1. first page — confirms auth, page_size, and that pagina/tamanhoPagina are honored
curl -s -o /dev/null -w '%{http_code} ' -H "$H" "$BASE?pagina=1&tamanhoPagina=15"
curl -s -H "$H" "$BASE?pagina=1&tamanhoPagina=15" | jq 'length'

# 2. last page with data — walk or bisect until the count drops below page_size
curl -s -H "$H" "$BASE?pagina=43&tamanhoPagina=15" | jq 'length'

# 3. one page past the end — this is the answer that matters
curl -s -w '\nHTTP %{http_code}\n' -H "$H" "$BASE?pagina=44&tamanhoPagina=15"
```

Worked example, run against `orgaos-siape` on **2026-08-02** (token redacted):

| Request | Status | Records | Reading |
|---|---|---|---|
| `pagina=1` | `200` | 15 | full page — stream continues |
| `pagina=43` | `200` | 10 | short page — this is the last page (640 records total) |
| `pagina=44` | `200` | 0 (`[]`) | **empty `200`** — precondition 1 satisfied ✅ |
| `pagina=99999` | `200` | 0 (`[]`) | still empty, never a 4xx — no surprise far past the end |

That endpoint qualifies for `concurrency > 1`. Two checks are worth adding before you trust the result:

- **Probe far past the end** (`pagina=99999`), not just `last + 1`. Speculation overshoots by up to `concurrency − 1`, and some endpoints change their answer only at large page numbers.
- **Confirm the page parameter is actually honored.** Compare the *bodies* of two different pages, not just their status:

  ```bash
  curl -s -H "$H" "$BASE?pagina=1&tamanhoPagina=15" | sha256sum
  curl -s -H "$H" "$BASE?pagina=2&tamanhoPagina=15" | sha256sum
  ```

  Identical digests mean the endpoint ignores `pagina` and serves one fixed list. `/api-de-dados/licitacoes/modalidades` does exactly this — every page number and every `tamanhoPagina` returns the same 14 records. Such an endpoint must stay at `concurrency: 1`: it is a single-page stream, so speculation buys nothing, and if its record count ever reached `page_size` JANUS would page forever.

Record the date and the observed answer in a YAML comment next to `past_end_status_codes`. An endpoint that answers with an empty `200` never exercises that list at all — the paginator's short-page rule ends the stream first — so leaving it at the default is correct; the comment is what proves someone checked.

One caveat seen in practice: a `4xx` past the end is only evidence of end-of-stream if it is **reproducible**. `licitacoes/ugs` intermittently answers `400 {"Erro na API":"Erro ao executar a consulta"}` at extreme page numbers while answering an empty `200` at its real boundary — that is a transient server fault, and adding `400` to `past_end_status_codes` would convert a real error into a silent truncation. Re-run the third command a few times before declaring a status terminal.

#### How to reduce over-fetch to zero

Without a known total, ending the stream costs at most `concurrency − 1` wasted in-flight requests, all cancelled the moment the end is detected, and the past-end page itself is requested exactly once. That is the structural bound.

If the payload exposes a record total, the bound becomes exact and over-fetch drops to **zero**:

- set `access.pagination.total_count_field` to its dotted path (e.g. `meta.total`); or
- if the total is somewhere the generic discovery cannot reach, override `ApiHook.resolve_total_records` in a source hook.

Without either, JANUS still probes common names (`total`, `totalCount`, `totalElements`, … at the root or inside `meta` / `metadata` / `pagination`). `count` is deliberately **not** probed — many APIs use it for "records on this page", which would produce a one-page ceiling. If your total really is `count`, opt in with `total_count_field: count`.

The tail-degradation rule makes an unreliable total safe: past the reported total, JANUS does **not** stop — it drops to one in-flight request at a time and keeps going until the paginator or a past-end status says the stream is over. A stale, cached, or shrinking total costs a little parallelism at the tail and can never truncate a dataset.

#### When to leave it at 1

- **Tight rate limits.** Receita Federal is capped at 6 req/min; concurrency cannot beat a throttle, it only makes the queue deeper.
- **Endpoints documented as single-writer or unstable under parallel reads.**
- **Any API whose past-end behavior you have not verified** with the three commands above. This is the default answer. `concurrency: 1` is always correct; `concurrency: 10` is correct only with evidence.

#### What a concurrent run reports

Concurrent runs emit extra extraction metadata that sequential runs do not: `speculative_request_count`, `speculative_discarded_count`, `past_end_terminated_count`, `past_end_status`, `total_records_reported`, and `lookahead_ceiling_source`. These are the *only* legitimate differences between a concurrent and a sequential run of the same source — artifacts, records, and checkpoints are identical. `pagination_concurrency` is reported on every run.

### `extraction`

- `extraction.mode` is one of `full_refresh`, `snapshot`, or `incremental`.
- `extraction.retry` currently supports `max_attempts`, `backoff_strategy`, and `backoff_seconds`.
- `extraction.dead_letter_max_items` is an optional execution budget for item-level self-healing. After request-level retries are exhausted for one request input or file candidate, JANUS records a dead letter and continues while the run stays within this budget. A value of `0` preserves fail-fast behavior but still records the item so a later `--resume` can skip it.
- `extraction.checkpoint_field` and `extraction.checkpoint_strategy` matter only when the source is incremental.
- `extraction.checkpoint_strategy` is one of `none`, `max_value`, or `date_window`.
- `extraction.lookback_days` is optional and only matters when incremental checkpoint values are time-based.

Practical rule: if `extraction.mode` is `incremental`, define a real `checkpoint_field` and use a checkpoint strategy other than `none`.

### `schema`

The `schema` block is narrower than it may look at first glance. Today it supports only:

- `mode`
- `path`

Supported values are:

- `mode: infer`
- `mode: explicit`

Examples:

```yaml
schema:
  mode: infer
```

```yaml
schema:
  mode: explicit
  path: conf/schemas/transparencia/servidores_por_orgao_schema.json
```

If `mode` is `explicit`, `path` is required. JANUS does not currently support inline field definitions, inline types, or other schema metadata in the source YAML.

The schema path may point to a Spark `StructType` JSON document or to a JSON field contract that JANUS can read as field names. Supported field-contract shapes are:

- a JSON array of field names;
- a mapping with `fields` or `columns`;
- a mapping with `schema.fields` or `schema.columns`.

During execution, JANUS passes an explicit Spark schema to the reader when the normalization handoff format matches `spark.input_format`. The same schema file also feeds the quality gate's expected-field check.

### `spark`

- `spark.input_format` tells JANUS how the normalization handoff should be read.
- `spark.write_mode` is one of `append`, `overwrite`, or `ignore`.
- `spark.repartition` is optional.
- `spark.partition_by` is an optional list of partition columns.
- `spark.read_options` is an optional string-to-string mapping for reader options such as CSV header, separator, or encoding.

A source that combines `extraction.mode: full_refresh` with `spark.write_mode: overwrite` and an Iceberg
bronze target **keeps its snapshot history across runs**. Each run replaces every row, but the previous
run stays readable: you can `SELECT ... VERSION AS OF <snapshot_id>` to compare against yesterday's bronze,
or roll the table back to it after a bad load. Query `<table>.history` and `<table>.snapshots` to see what
is available. Two things are worth knowing as a source owner. First, a **schema or partition change resets
that history by design** — if a run drops a column, changes a column's type, or changes
`spark.partition_by`, JANUS recreates the table and starts a fresh snapshot log, recording
`history_reset_reason` in the run metadata so the reset is visible rather than assumed. Adding a new column
does *not* reset history. Second, snapshots are retained indefinitely — nothing expires them today — so the
storage a full-refresh source occupies grows with each run.

### `outputs`

- Each source defines `outputs.raw`, `outputs.bronze`, and `outputs.metadata`.
- Each output target must define `path` and `format`.
- Only `outputs.bronze` may define `namespace` and `table_name`.
- `namespace` and `table_name` are only valid when `outputs.bronze.format` is `iceberg`.

### `quality`

- `quality.required_fields` is an optional list of fields that must be present.
- `quality.unique_fields` is an optional list of uniqueness hints.
- `quality.allow_schema_evolution` is a boolean flag.

If you are unsure whether a field belongs in config, check the example source and the typed contract before inventing a new key.

## Step 3: prefer config reuse over code

Before touching Python, ask:

- Can this source fit an existing family and variant?
- Can the odd behavior be expressed through config already?
- Are output paths, schema mode, and checkpoint rules declarative?
- Is the only missing piece a runtime secret value that should come from the environment?

If the answer is yes, stay in YAML.

Examples of behavior that should stay in config:

- base URL or direct file URL;
- auth mode and env var names;
- page size or offset parameter names;
- file URL resolver mode, remote file filters, and archive-member filters;
- checkpoint field and checkpoint strategy;
- explicit schema path and Spark read options;
- output formats and paths, plus optional `outputs.bronze.namespace` and `outputs.bronze.table_name` for Iceberg bronze targets;
- quality rules such as required fields and uniqueness keys.

## Step 4: choose the lightest extension point

When config is not enough, choose the smallest honest extension.

### Use an existing strategy as-is when:

- the source follows the family pattern already;
- only source values differ;
- there is no real runtime mismatch.

### Add a source hook when:

- the source mostly fits the family;
- one or two edges differ in a source-specific way;
- the change can stay isolated to that source.

Good hook use cases:

- a custom next-cursor field for one API;
- a checkpoint query parameter that does not match the checkpoint field name;
- a version token hidden in one file naming convention;
- archive-member selection rules for one package layout;
- a non-standard wrapper around one catalog payload.

### Improve the family strategy when:

- the behavior is reusable across sources;
- a second source would benefit from the same logic;
- the change belongs to the pattern, not the source identity.

If the only implementation plan is "copy the nearest source and adjust it," you are in the wrong extension point.

## Step 5: keep hooks on a short leash

Hooks are allowed, but they are not a second home for generic logic.

A good hook is:

- tied to one source;
- small enough to explain in a paragraph;
- named after the source, not after a vague generic behavior;
- covered by focused tests.

A bad hook is:

- branching on several source ids;
- reimplementing strategy loops;
- doing work the registry should have validated already;
- hiding business rules that really belong in a reusable family helper.

When in doubt, make the hook smaller or redesign the family boundary.

## Step 6: define outputs and checkpoints deliberately

Do not treat outputs as an afterthought.

Each source must define:

- a raw path for preserved upstream artifacts;
- a bronze path for minimally standardized structured data;
- a metadata path for run records, checkpoints, lineage, and validations.

If `outputs.bronze.format` is `iceberg`, the source may also define `outputs.bronze.namespace` and `outputs.bronze.table_name` when the default path-derived Iceberg identifier is not the desired bronze table name.

Checkpoint choices should match the family:

- API sources usually checkpoint by record timestamp or field value;
- file sources usually checkpoint by publication/version token;
- catalog sources usually checkpoint by metadata freshness, not underlying business facts.

If a source cannot explain its checkpoint semantics clearly, incremental mode is probably premature.

## Step 7: test what you changed

At minimum, add or update tests for the behavior you introduced.

Typical coverage includes:

- source-config validation or registry loading;
- planner resolution for the chosen family and variant;
- strategy behavior for the new source shape;
- hook behavior when a hook is necessary;
- metadata, checkpoint, or validation side effects when they changed.

Do not rely on a manual run as the only proof that a new pattern belongs in JANUS.

## Step 8: update the docs when the pattern changed

Update the contributor docs when you introduce:

- a new reusable strategy variant;
- a new hook boundary contributors are expected to use;
- a new reproducibility requirement;
- a new anti-pattern the project should reject explicitly.

If the code learns a new reusable pattern and the docs stay silent, the next contributor will reverse-engineer it from implementation details, which is exactly what JANUS is trying to avoid.

## Onboarding checklist

Use this checklist before considering a source "added":

- The source is federal, public, and in scope.
- The family and variant are explicit.
- The YAML contract is complete and validated.
- Secrets are referenced by env var name only.
- Output paths land in raw, bronze, and metadata zones.
- If bronze uses Iceberg, any explicit `namespace` and `table_name` are intentional and documented.
- Checkpoint semantics are intentional, or incremental mode is disabled.
- Existing strategy behavior was reused wherever possible.
- Any hook is narrow, source-local, and tested.
- Any new reusable pattern is reflected in the docs.

## Anti-patterns

Do not onboard a source by:

- copying an old implementation and renaming identifiers;
- adding `if source_id == ...` in generic strategy code;
- teaching the registry about one source's business rules;
- hardcoding tokens, cookies, usernames, or passwords in YAML;
- writing outside the configured storage layout;
- treating raw persistence as optional because the payload looks simple;
- skipping validation and metadata because a first run "looked fine".

JANUS grows by reusable patterns and explicit contracts. If a source does not fit those rules yet, the right move is to improve the framework boundary, not to sneak around it.
