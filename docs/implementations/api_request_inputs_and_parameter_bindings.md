# Request Inputs and Parameter Bindings

The short version is: JANUS now has a bounded request-shaping layer for API and catalog sources that need runtime request parameters or one request stream per upstream identifier.

This work extends the existing registry, planner, API runtime, and catalog runtime. It does not introduce a workflow engine, cross-source orchestration, or a second request subsystem.

## Where this fits

The request-shaping layer sits between base request construction and pagination:

`source YAML -> registry -> planner -> API/catalog strategy -> request inputs -> parameter bindings -> pagination -> raw artifacts -> Spark handoff -> metadata`

That placement matters.

Pagination is still the reusable inner loop for page, offset, cursor, or non-paginated execution. Request inputs are the bounded outer loop that decides how many logical request streams JANUS should execute before that pagination logic begins.

## Contract boundary

The feature adds two optional API/catalog blocks under `access`:

- `request_inputs`
- `parameter_bindings`

It deliberately keeps `access.params` unchanged as the home for static literal request parameters.

In practice, the split is:

- `access.params` for literals that do not change during the run
- `access.request_inputs` for bounded runtime contexts
- `access.parameter_bindings` for request params resolved from the current request input or checkpoint state
- hooks for the irregular cases that still do not fit those declarative layers

If `access.request_inputs` is omitted, JANUS falls back to the existing one-stream behavior with `type: none`.

## Supported request inputs

The contract stays intentionally small.

### `none`

This is the default. JANUS runs one logical request stream and pagination behaves exactly as it did before request inputs existed.

### `date_window`

This request-input type generates deterministic daily or monthly windows from:

- `start`
- `end`
- `step`

Each generated request input exposes:

- `window_start`
- `window_end`

That allows endpoints to bind parameters such as `mesAno`, `dataIdaDe`, or `dataIdaAte` without moving those rules into a source hook.

### `iceberg_rows`

This request-input type reads one existing Iceberg table through the active Spark session, projects the configured columns into request-input field names, optionally applies `distinct`, orders the result deterministically, and returns one logical request stream per projected row.

The supported fields are:

- `upstream_source_id` (required, non-empty producer source ID)
- `namespace`
- `table_name`
- `columns`
- `distinct`

Every Iceberg leaf, including each member of `combined.inputs`, requires the producer declaration. It identifies who produces the referenced table; it does not change the table read, projection, or binding behavior. Producer existence and table matching belong to registry graph validation.

The scope is intentionally narrow:

- one upstream table
- projected columns only
- no joins
- no arbitrary SQL

### `combined`

Use this type when an endpoint requires parameters from more than one independent source — for example, an entity identifier from an upstream Iceberg table and a date range per month.

Declare two or more atomic inputs under `inputs:`. JANUS resolves each sub-input independently, computes the Cartesian product, and merges each combination into one flat request context. Parameter bindings resolve against that merged context as normal.

Constraints:

- `inputs` must contain at least two entries.
- Each entry must be `date_window` or `iceberg_rows`. Nesting `combined` or `none` is not supported.
- Field names must not conflict across entries. Two `date_window` entries, for example, both expose `window_start` and `window_end`, which is rejected at registry load time.

## Supported parameter bindings

The supported binding sources are:

- `checkpoint_value`
- `request_input.window_start`
- `request_input.window_end`
- `request_input.<field>` — any field exposed by the active request input; for `combined`, this covers fields from all sub-inputs

Bindings may also declare an optional `format` for date-like values. This is what allows a request window to become `%Y%m` for one endpoint and `%Y-%m-%d` for another without adding endpoint-specific code.

The registry and runtime keep the binding surface explicit. JANUS rejects:

- unsupported binding sources
- request-input bindings when request inputs are effectively `none`
- window bindings that do not match `date_window`
- Iceberg bindings that reference fields outside `access.request_inputs.columns`
- duplicate parameter keys declared in both `access.params` and `access.parameter_bindings`

## Runtime behavior

At extraction time, API and catalog strategies execute in this order:

1. Build the base request with auth and any checkpoint-aware default params.
2. Load or synthesize the configured request inputs.
3. Resolve `access.parameter_bindings` for the current request input.
4. Merge the resolved values with static `access.params`.
5. Run the existing pagination flow for that bound request stream.
6. Persist raw artifacts and extraction metadata with request-input context.

Catalog sources also use bound values to fill `{name}` placeholders in `access.path` or `access.url`. Bound values that do not match a path placeholder remain query parameters.

This keeps the implementation aligned with the JANUS foundation:

- metadata-driven where the behavior is declarative
- strategy-based instead of source-specific branching
- hooks preserved as isolated escape hatches
- safe defaults and deterministic outputs

## Deterministic raw artifacts and metadata

Multi-input runs must not collide in the raw zone.

For one-stream runs, JANUS keeps the existing layout under `pages/`.

For multi-input runs, JANUS writes each request stream under a stable folder such as:

- `request-input-000001/page-0001.json`
- `request-input-000002/page-0001.json`

The extraction metadata now includes request-input context as well, including:

- `request_input_type`
- `request_input_count`
- `bound_parameter_names`
- `upstream_namespace`
- `upstream_table_name`
- `upstream_column_names`

That keeps request-input execution observable without inventing a new metadata model.

## Hook boundary

The new contract reduces the need for hooks, but it does not remove them.

Hooks are still the right tool when a source needs behavior such as:

- custom request preparation beyond parameter binding
- non-standard cursor lookup
- response-specific record extraction
- checkpoint parameter naming that does not fit the default path

What should not move into hooks anymore is reusable request shaping that can be described safely through `access.request_inputs` and `access.parameter_bindings`.

## Coverage

The implementation is covered at three levels:

- registry validation tests in `tests/unit/registry/test_source_registry.py`
- request-input and binding unit tests in `tests/unit/strategies/api/test_request_inputs.py`
- end-to-end API runtime tests in `tests/unit/strategies/api/test_api_strategy.py`
- catalog request-input runtime tests in `tests/unit/strategies/catalog/test_catalog_strategy.py`

That coverage locks down:

- contract validation
- date-window generation
- Iceberg row loading
- combined input Cartesian product and merged binding resolution
- binding resolution
- deterministic raw artifact naming
- backward compatibility for existing static-param API sources

The result stays intentionally small enough to explain during source onboarding without introducing new platform concepts beyond the request-facing strategy families.
