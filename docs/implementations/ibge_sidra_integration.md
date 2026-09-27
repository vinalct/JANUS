# IBGE SIDRA Integration

This note documents the IBGE delivery completed in JANUS for the SIDRA Aggregates API, with emphasis on what was implemented and how the source is modeled inside the framework.

The IBGE integration fits the JANUS API strategy family, but it also introduces a source-specific detail that matters: SIDRA `view=flat` responses are not plain record lists. They carry a small header row that describes the meaning of the keys used by the following rows. Because of that, the transport path can remain generic, while the response interpretation needs a dedicated hook.

## What was implemented

The delivery includes two IBGE sources:

- `ibge_pib_brasil` in `conf/sources/ibge/sidra.yaml`
- `ibge_agro_abacaxi_pronaf` in `conf/sources/ibge/sidra.yaml`

Both sources are modeled as:

- `strategy: api`
- `strategy_variant: date_window_api`
- `source_hook: ibge.sidra_flat`
- `schema.contract` pointing to the reviewed IBGE table contract

Both write to the standard JANUS zones:

- raw: exact API response pages and normalized handoff artifacts
- bronze: Iceberg table materialized from the Spark-readable handoff
- metadata: run information, checkpoints, and lineage

The implementation also includes the IBGE hook registry wiring, fixtures, and integration tests.

## Source definitions

### `ibge_pib_brasil`

This source uses IBGE aggregate `5938` and variable `37`.

It is a compact example of the SIDRA flat shape, with a small number of dimensions. It is useful as the baseline source because it keeps the response easy to inspect.

### `ibge_agro_abacaxi_pronaf`

This source uses IBGE aggregate `1712` and variable `214`, with classification filters applied in the request.

It was added to exercise a different response shape. While the PIB source uses a shorter dimension layout, this source returns dimensions up to `D9`. That makes it a better test of whether the parser is following the SIDRA pattern itself instead of accidentally fitting one endpoint.

## Hook implementation

The hook lives in `src/janus/hooks/ibge/__init__.py`.

The generic hook id is:

```text
ibge.sidra_flat
```

The planner loads built-in hooks by default, so a source that declares `source_hook: ibge.sidra_flat` can be planned and executed directly through the normal CLI path.

The hook is responsible for only one thing: translating a SIDRA flat payload into a normalized bronze handoff.

It does not replace the API strategy. The reusable API strategy still handles:

- request construction
- retries
- rate limiting
- raw response persistence
- checkpoint plumbing
- Spark handoff

## Why SIDRA needs a hook

A SIDRA `view=flat` payload looks like a list of JSON objects, but the first object is not a business record. It is the description of the fields used in the remaining rows.

A simplified example:

```json
[
  {
    "D1C": "Brasil (Código)",
    "D1N": "Brasil",
    "D2C": "Ano (Código)",
    "D2N": "Ano",
    "D3C": "Variável (Código)",
    "D3N": "Variável",
    "MC": "Unidade de Medida (Código)",
    "MN": "Unidade de Medida",
    "NC": "Nível Territorial (Código)",
    "NN": "Nível Territorial",
    "V": "Valor"
  },
  {
    "D1C": "1",
    "D1N": "Brasil",
    "D2C": "2023",
    "D2N": "2023",
    "D3C": "37",
    "D3N": "Produto Interno Bruto a preços correntes",
    "MC": "40",
    "MN": "Mil Reais",
    "NC": "1",
    "NN": "Brasil",
    "V": "10943345439"
  }
]
```

The generic API strategy should not be responsible for deciding that `D2N` means the display value for a period dimension, or that `V` is the measure. That interpretation belongs in a small source-family adapter.

## How the hook normalizes SIDRA

The hook reads the first row as the description of the payload shape, then parses the remaining rows using that header information.

The normalization output has three parts.

### 1. JANUS metadata added by the base normalizer

These columns are common to bronze datasets and support lineage and reproducibility:

- `janus_run_id`
- `janus_source_id`
- `janus_source_name`
- `janus_environment`
- `janus_strategy_family`
- `janus_strategy_variant`
- `ingestion_timestamp`
- `ingestion_date`

### 2. Stable top-level SIDRA fields

These are promoted because they are useful across SIDRA sources and help with checkpointing and quick identification of the record:

- `aggregate_id`
- `value`
- `sidra_value_label`
- `sidra_period_code`
- `sidra_period_name`
- `sidra_variable_code`
- `sidra_variable_name`

The period and variable fields are derived from the parsed dimensions, not copied from explicit raw keys.

### 3. Preserved source structure

The rest of the payload meaning is kept in nested fields:

- `sidra_attributes`
- `sidra_attribute_count`
- `sidra_dimensions`
- `sidra_dimension_count`

`sidra_dimensions` stores the `D1..Dn` dimensions as an array of structs with the fields below:

- `id`
- `position`
- `code`
- `name`
- `code_label`
- `label`

For one entry such as:

```text
id         = D4
label      = Produtos da lavoura temporária
name       = Abacaxi
code       = 4844
code_label = Produtos da lavoura temporária (Código)
position   = 4
```

The intended reading is:

- dimension name: `Produtos da lavoura temporária`
- chosen member: `Abacaxi`
- chosen member code: `4844`

`sidra_attributes` follows the same idea for non-`D*` fields such as territorial level and unit.

## Bronze modeling choice

The IBGE bronze dataset is intentionally not the final analytical model.

In JANUS, bronze should remain close to the source while still being readable by Spark and stable enough for validation, checkpointing, and reruns. For SIDRA, that means preserving the dimensional structure instead of flattening every label into a top-level column.

That choice avoids turning query-specific header labels into schema. Different SIDRA aggregates can expose different dimensions, different ordering, and different descriptive labels. Keeping those as nested data makes the bronze contract more stable.

The tradeoff is readability. A compact `show()` output in PySpark can make `sidra_dimensions` look harder to read than it actually is, because Spark prints nested structs without their field names. The data itself remains explicit; the display is what gets compact.

## How to interpret one bronze row

A bronze row should be understood as one observation plus its dimensional coordinates.

For the agricultural example, a row means:

- aggregate `1712`
- period `2006`
- variable `Quantidade produzida`
- product `Abacaxi`
- producer condition `Proprietário`
- area-total group `Total`
- activity group `Total`
- harvested-area group `Total`
- Pronaf group `Total`
- territorial context `Brasil`
- unit `Mil frutos`
- measured value `456692`

That is, in practice, one point in a multidimensional statistical cube.

## Tests and fixtures

Fixtures live under:

```text
tests/fixtures/ibge/
```

Integration tests live under:

```text
tests/integration/ibge/
```

The delivered test coverage checks:

- registry loading for both IBGE sources
- planner resolution of the hook id
- raw response persistence
- normalized JSONL handoff generation
- dynamic parsing for both shorter and larger dimension layouts
- quality-gate compatibility for the bronze contract
- metadata and checkpoint behavior

A focused fixture without `NC` and `NN` was also added to confirm that the parser does not depend on those optional fields being present.

## Scope of this delivery

This implementation ends at a solid bronze contract for IBGE SIDRA.

It does not yet create a silver model with source-aware business columns such as product, period, producer condition, and activity group flattened into an analyst-facing table. That would be the natural next modeling step for consumption, but it belongs to a later layer.

## Summary

The IBGE delivery adds a reusable SIDRA flat integration to JANUS.

The API transport remains generic. The SIDRA-specific interpretation stays isolated in a small hook. The bronze output preserves the observation, its dimensional coordinates, and the operational metadata required by the framework, while remaining flexible enough to support more than one IBGE aggregate shape.
