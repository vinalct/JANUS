from pathlib import Path

import pytest

from janus.models.source_config import (
    CombinedRequestInputsConfig,
    DateWindowRequestInputsConfig,
    IcebergRowsRequestInputsConfig,
    SourceConfig,
    SourceConfigValidationError,
)
from janus.registry import SourceNotFoundError, load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_checked_in_registry_lists_enabled_sources():
    registry = load_registry(PROJECT_ROOT)

    enabled_source_ids = [source.source_id for source in registry.list_sources()]

    assert enabled_source_ids == ["federal_open_data_example"]


def test_checked_in_registry_returns_typed_source_config():
    registry = load_registry(PROJECT_ROOT)

    source = registry.get_source("federal_open_data_example")

    assert isinstance(source, SourceConfig)
    assert source.strategy == "api"
    assert source.strategy_variant == "page_number_api"
    assert source.access.auth.type == "header_token"
    assert source.access.request_inputs.type == "none"
    assert source.access.parameter_bindings is None
    assert source.outputs.bronze.path == "data/bronze/example/federal_open_data_example"
    assert source.quality.required_fields == ("id", "updated_at")
    assert str(source.config_path).endswith("conf/sources/example/example_source.yaml")


def test_checked_in_registry_loads_cnpj_entity_source_contracts():
    registry = load_registry(PROJECT_ROOT)

    source = registry.get_source(
        "receita_federal__cnpj__empresas_full_refresh",
        include_disabled=True,
    )

    assert source.enabled is False
    assert source.source_hook is None
    assert source.access.remote_file_pattern == "Empresas*.zip"
    assert source.schema.mode == "explicit"
    assert source.schema.path == "conf/schemas/receita_federal/cnpj/empresas_schema.json"
    assert source.outputs.raw.path == "data/raw/receita_federal/cnpj/empresas"
    assert source.outputs.bronze.namespace == "bronze__receita_federal"
    assert source.outputs.bronze.table_name == "cnpj_empresas"


def test_registry_discovers_sources_in_nested_domain_directories(tmp_path):
    project_root = _create_project(
        tmp_path,
        {
            "ibge/pib.yaml": _valid_source_yaml("ibge_pib_brasil", enabled=True),
            "inep/censo.yaml": _valid_source_yaml("inep_censo_escolar", enabled=True),
        },
    )

    registry = load_registry(project_root)

    assert [source.source_id for source in registry.list_sources()] == [
        "ibge_pib_brasil",
        "inep_censo_escolar",
    ]


def test_registry_loads_multiple_sources_from_one_grouped_yaml_file(tmp_path):
    project_root = _create_project(
        tmp_path,
        {
            "ibge/sidra.yaml": _grouped_sources_yaml(
                _valid_source_yaml("ibge_pib_brasil", enabled=True),
                _valid_source_yaml("ibge_agro_abacaxi_pronaf", enabled=True),
            )
        },
    )

    registry = load_registry(project_root)

    assert [source.source_id for source in registry.list_sources()] == [
        "ibge_pib_brasil",
        "ibge_agro_abacaxi_pronaf",
    ]


def test_grouped_source_yaml_prefixes_validation_errors_with_entry_index(tmp_path):
    broken_yaml = _valid_source_yaml("broken_grouped_source", enabled=True).replace(
        "strategy_variant: page_number_api\n",
        "strategy_variant: not_real\n",
        1,
    )
    project_root = _create_project(
        tmp_path,
        {
            "ibge/sidra.yaml": _grouped_sources_yaml(
                _valid_source_yaml("valid_grouped_source", enabled=True),
                broken_yaml,
            )
        },
    )

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    message = str(exc_info.value)
    assert "sources[1].strategy_variant: must be one of" in message


def test_registry_parses_optional_bronze_iceberg_namespace_and_table(tmp_path):
    source_yaml = _valid_source_yaml("named_bronze_source", enabled=True).replace(
        "  bronze:\n"
        "    path: data/bronze/example/named_bronze_source\n"
        "    format: iceberg\n",
        "  bronze:\n"
        "    path: data/bronze/example/named_bronze_source\n"
        "    format: iceberg\n"
        "    namespace: curated\n"
        "    table_name: named_bronze_table\n",
    )
    project_root = _create_project(tmp_path, {"named.yaml": source_yaml})

    source = load_registry(project_root).get_source("named_bronze_source")

    assert source.outputs.bronze.namespace == "curated"
    assert source.outputs.bronze.table_name == "named_bronze_table"


def test_registry_rejects_legacy_bronze_table_key(tmp_path):
    source_yaml = _valid_source_yaml("legacy_table_source", enabled=True).replace(
        "  bronze:\n"
        "    path: data/bronze/example/legacy_table_source\n"
        "    format: iceberg\n",
        "  bronze:\n"
        "    path: data/bronze/example/legacy_table_source\n"
        "    format: iceberg\n"
        "    table: old_name\n",
    )
    project_root = _create_project(tmp_path, {"legacy.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert "outputs.bronze.table: is not supported; use table_name" in str(exc_info.value)


def test_registry_excludes_disabled_sources_by_default(tmp_path):
    project_root = _create_project(
        tmp_path,
        {
            "enabled.yaml": _valid_source_yaml("enabled_source", enabled=True),
            "disabled.yaml": _valid_source_yaml("disabled_source", enabled=False),
        },
    )

    registry = load_registry(project_root)

    assert [source.source_id for source in registry.list_sources()] == ["enabled_source"]
    assert [source.source_id for source in registry.list_sources(enabled_only=False)] == [
        "disabled_source",
        "enabled_source",
    ]
    with pytest.raises(SourceNotFoundError, match="disabled"):
        registry.get_source("disabled_source")


def test_registry_parses_spark_read_options_mapping(tmp_path):
    source_yaml = _valid_source_yaml("csv_source", enabled=True).replace(
        "  write_mode: append\n",
        "  write_mode: append\n"
        "  read_options:\n"
        "    header: \"true\"\n"
        "    sep: \";\"\n"
        "    encoding: \"ISO-8859-1\"\n",
        1,
    )
    project_root = _create_project(
        tmp_path,
        {"csv.yaml": source_yaml},
    )

    source = load_registry(project_root).get_source("csv_source")

    assert source.spark.read_options == {
        "header": "true",
        "sep": ";",
        "encoding": "ISO-8859-1",
    }


def test_invalid_source_yaml_has_actionable_validation_errors(tmp_path):
    project_root = _create_project(
        tmp_path,
        {
            "broken.yaml": """
source_id: broken_source
name: Broken Source
owner: janus
enabled: true
source_type: api
strategy: api
strategy_variant: not_real
federation_level: federal
domain: example
public_access: true

access:
  base_url: https://example.invalid
  method: GET
  format: json
  auth:
    type: header_token
  pagination:
    type: page_number
  rate_limit:
    requests_per_minute: 10

extraction:
  mode: incremental
  retry:
    max_attempts: 3
    backoff_strategy: fixed
    backoff_seconds: 1

schema:
  mode: explicit

spark:
  input_format: json
  write_mode: append

outputs:
  raw:
    format: json
  bronze:
    path: data/bronze/example/broken
    format: iceberg
  metadata:
    path: data/metadata/example/broken
    format: json

quality:
  allow_schema_evolution: true
""",
        },
    )

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    message = str(exc_info.value)
    assert "broken.yaml" in message
    assert "strategy_variant: must be one of" in message
    assert "access.auth.env_var: is required for token-based auth" in message
    assert (
        "access.auth.header_name: is required when access.auth.type is "
        "'header_token'" in message
    )
    assert (
        "access.pagination.page_param: is required when access.pagination.type "
        "is 'page_number'" in message
    )
    assert (
        "extraction.checkpoint_field: is required when extraction.mode is "
        "'incremental'" in message
    )
    assert "schema.path: is required when schema.mode is 'explicit'" in message
    assert "outputs.raw.path: is required" in message


def test_duplicate_source_ids_fail_fast(tmp_path):
    project_root = _create_project(
        tmp_path,
        {
            "first.yaml": _valid_source_yaml("duplicate_source", enabled=True),
            "second.yaml": _valid_source_yaml("duplicate_source", enabled=True),
        },
    )

    with pytest.raises(ValueError, match="Duplicate source_id 'duplicate_source'"):
        load_registry(project_root)


def test_registry_parses_date_window_request_inputs_and_parameter_bindings(tmp_path):
    source_yaml = _valid_source_yaml("windowed_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  params:\n"
        "    situacao: TODAS\n"
        "  request_inputs:\n"
        "    type: date_window\n"
        "    start: 2025-01-01\n"
        "    end: 2025-12-31\n"
        "    step: month\n"
        "  parameter_bindings:\n"
        "    mesAno:\n"
        "      from: request_input.window_end\n"
        "      format: \"%Y%m\"\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"windowed.yaml": source_yaml})

    source = load_registry(project_root).get_source("windowed_source")

    assert source.access.params == {"situacao": "TODAS"}
    assert source.access.request_inputs.type == "date_window"
    assert source.access.request_inputs.start.isoformat() == "2025-01-01"
    assert source.access.request_inputs.end.isoformat() == "2025-12-31"
    assert source.access.request_inputs.step == "month"
    assert source.access.parameter_bindings is not None
    assert source.access.parameter_bindings["mesAno"].from_ == "request_input.window_end"
    assert source.access.parameter_bindings["mesAno"].format == "%Y%m"


def test_registry_parses_iceberg_rows_request_inputs_and_parameter_bindings(tmp_path):
    source_yaml = _valid_source_yaml("detail_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  request_inputs:\n"
        "    type: iceberg_rows\n"
        "    namespace: bronze_transparencia\n"
        "    table_name: emendas_parlamentares__emendas\n"
        "    columns:\n"
        "      emenda_id: id\n"
        "    distinct: true\n"
        "  parameter_bindings:\n"
        "    id:\n"
        "      from: request_input.emenda_id\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"detail.yaml": source_yaml})

    source = load_registry(project_root).get_source("detail_source")

    assert source.access.request_inputs.type == "iceberg_rows"
    assert source.access.request_inputs.namespace == "bronze_transparencia"
    assert source.access.request_inputs.table_name == "emendas_parlamentares__emendas"
    assert source.access.request_inputs.columns == {"emenda_id": "id"}
    assert source.access.request_inputs.distinct is True
    assert source.access.parameter_bindings is not None
    assert source.access.parameter_bindings["id"].from_ == "request_input.emenda_id"
    assert source.access.parameter_bindings["id"].format is None


def test_registry_rejects_request_input_bindings_without_request_inputs(tmp_path):
    source_yaml = _valid_source_yaml("broken_binding_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  parameter_bindings:\n"
        "    mesAno:\n"
        "      from: request_input.window_end\n"
        "      format: \"%Y%m\"\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"broken_binding.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert (
        "access.parameter_bindings.mesAno.from: requires access.request_inputs to declare a non-'none' type"  # noqa: E501
        in str(exc_info.value)
    )


def test_registry_rejects_duplicate_static_and_bound_request_params(tmp_path):
    source_yaml = _valid_source_yaml("duplicate_param_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  params:\n"
        "    id: fixed\n"
        "  request_inputs:\n"
        "    type: iceberg_rows\n"
        "    namespace: bronze_transparencia\n"
        "    table_name: emendas_parlamentares__emendas\n"
        "    columns:\n"
        "      emenda_id: id\n"
        "  parameter_bindings:\n"
        "    id:\n"
        "      from: request_input.emenda_id\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"duplicate_param.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert (
        "access.parameter_bindings.id: duplicates access.params.id; declare the parameter in only one place"  # noqa: E501
        in str(exc_info.value)
    )


def test_registry_parses_past_end_status_codes_and_total_count_field(tmp_path):
    """The declared end-of-stream contract survives the real YAML entry point.

    Concurrent pagination reads these two keys off the loaded config, so a source that
    declares them must get the normalized (sorted, de-duplicated) tuple back — and a
    source that omits them must get the default set, not an empty one.
    """
    declared_yaml = _valid_source_yaml("past_end_source", enabled=True).replace(
        "    page_size: 100\n",
        "    page_size: 100\n"
        "    past_end_status_codes: [416, 404, 404]\n"
        "    total_count_field: meta.total\n",
        1,
    )
    project_root = _create_project(
        tmp_path,
        {
            "past_end.yaml": declared_yaml,
            "default_past_end.yaml": _valid_source_yaml("default_past_end_source", enabled=True),
        },
    )

    registry = load_registry(project_root)

    declared = registry.get_source("past_end_source")
    assert declared.access.pagination.past_end_status_codes == (404, 416)
    assert declared.access.pagination.total_count_field == "meta.total"

    defaulted = registry.get_source("default_past_end_source")
    assert defaulted.access.pagination.past_end_status_codes == (404, 416)
    assert defaulted.access.pagination.total_count_field is None


def test_registry_rejects_concurrency_above_one_without_speculative_pagination(tmp_path):
    source_yaml = (
        _valid_source_yaml("cursor_concurrency_source", enabled=True)
        .replace("strategy_variant: page_number_api", "strategy_variant: cursor_api", 1)
        .replace(
            "    type: page_number\n"
            "    page_param: page\n"
            "    size_param: page_size\n"
            "    page_size: 100\n",
            "    type: cursor\n    cursor_param: next\n",
            1,
        )
        .replace("    concurrency: 1\n", "    concurrency: 8\n", 1)
    )
    project_root = _create_project(tmp_path, {"cursor_concurrency.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    message = str(exc_info.value)
    assert "access.rate_limit.concurrency: must be 1 unless access.pagination.type is" in message
    assert "'cursor' pagination" in message


def _create_project(tmp_path: Path, sources: dict[str, str]) -> Path:
    conf_dir = tmp_path / "conf"
    sources_dir = conf_dir / "sources"
    sources_dir.mkdir(parents=True)
    (conf_dir / "app.yaml").write_text(
        "registry:\n  sources_dir: conf/sources\n  file_pattern: \"*.yaml\"\n",
        encoding="utf-8",
    )
    for file_name, content in sources.items():
        file_path = sources_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content.lstrip(), encoding="utf-8")
    return tmp_path


def _grouped_sources_yaml(*source_yamls: str) -> str:
    entries = "\n".join(_as_grouped_entry(source_yaml) for source_yaml in source_yamls)
    return f"sources:\n{entries}\n"


def _as_grouped_entry(source_yaml: str) -> str:
    lines = source_yaml.strip().splitlines()
    if not lines:
        return "  - {}"

    rendered = [f"  - {lines[0]}"]
    rendered.extend(f"    {line}" if line else "" for line in lines[1:])
    return "\n".join(rendered)


def _valid_source_yaml(source_id: str, *, enabled: bool) -> str:
    enabled_value = "true" if enabled else "false"
    return f"""
source_id: {source_id}
name: {source_id}
owner: janus
enabled: {enabled_value}
source_type: api
strategy: api
strategy_variant: page_number_api
federation_level: federal
domain: example
public_access: true

access:
  base_url: https://example.invalid
  path: /records
  method: GET
  format: json
  timeout_seconds: 30
  auth:
    type: none
  pagination:
    type: page_number
    page_param: page
    size_param: page_size
    page_size: 100
  rate_limit:
    requests_per_minute: 10
    concurrency: 1

extraction:
  mode: full_refresh
  retry:
    max_attempts: 3
    backoff_strategy: fixed
    backoff_seconds: 1

schema:
  mode: infer

spark:
  input_format: json
  write_mode: append

outputs:
  raw:
    path: data/raw/example/{source_id}
    format: json
  bronze:
    path: data/bronze/example/{source_id}
    format: iceberg
  metadata:
    path: data/metadata/example/{source_id}
    format: json

quality:
  allow_schema_evolution: true
"""


def test_registry_rejects_invalid_date_window_request_inputs(tmp_path):
    source_yaml = _valid_source_yaml("broken_window_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  request_inputs:\n"
        "    type: date_window\n"
        "    start: not-a-date\n"
        "    step: year\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"broken_window.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    message = str(exc_info.value)
    assert "access.request_inputs.start: must be a YYYY-MM-DD date" in message
    assert "access.request_inputs.end: is required" in message
    assert "access.request_inputs.step: must be one of: day, month" in message


def test_registry_rejects_invalid_iceberg_request_inputs(tmp_path):
    source_yaml = _valid_source_yaml("broken_iceberg_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  request_inputs:\n"
        "    type: iceberg_rows\n"
        "    namespace: bronze_transparencia\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"broken_iceberg.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    message = str(exc_info.value)
    assert "access.request_inputs.table_name: is required" in message
    assert "access.request_inputs.columns: is required" in message


def test_registry_rejects_unsupported_parameter_binding_source(tmp_path):
    source_yaml = _valid_source_yaml("unsupported_binding_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  parameter_bindings:\n"
        "    id:\n"
        "      from: runtime.identifier\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"unsupported_binding.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert (
        "access.parameter_bindings.id.from: must be 'checkpoint_value', "
        "'request_input.window_start', 'request_input.window_end', or "
        "'request_input.<field>'" in str(exc_info.value)
    )


def test_registry_rejects_request_input_bindings_when_request_inputs_are_invalid(tmp_path):
    source_yaml = _valid_source_yaml("invalid_request_input_type_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  request_inputs:\n"
        "    type: unsupported\n"
        "  parameter_bindings:\n"
        "    id:\n"
        "      from: request_input.emenda_id\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"invalid_request_inputs.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    message = str(exc_info.value)
    assert (
        "access.request_inputs.type: must be one of: combined, date_window, iceberg_rows, none"
        in message
    )
    assert (
        "access.parameter_bindings.id.from: requires access.request_inputs to declare a non-'none' type"  # noqa: E501
        in message
    )


def test_registry_parses_combined_request_inputs_and_parameter_bindings(tmp_path):
    source_yaml = _valid_source_yaml("combined_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  request_inputs:\n"
        "    type: combined\n"
        "    inputs:\n"
        "      - type: iceberg_rows\n"
        "        namespace: bronze_transparencia\n"
        "        table_name: orgaos\n"
        "        columns:\n"
        "          orgao_codigo: codigo\n"
        "      - type: date_window\n"
        "        start: 2025-01-01\n"
        "        end: 2025-03-31\n"
        "        step: month\n"
        "  parameter_bindings:\n"
        "    codigoOrgao:\n"
        "      from: request_input.orgao_codigo\n"
        "    dataInicio:\n"
        "      from: request_input.window_start\n"
        "      format: \"%Y-%m-%d\"\n"
        "    dataFinal:\n"
        "      from: request_input.window_end\n"
        "      format: \"%Y-%m-%d\"\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"combined.yaml": source_yaml})

    source = load_registry(project_root).get_source("combined_source")

    assert source.access.request_inputs.type == "combined"
    assert isinstance(source.access.request_inputs, CombinedRequestInputsConfig)
    assert len(source.access.request_inputs.inputs) == 2

    iceberg_sub = source.access.request_inputs.inputs[0]
    assert isinstance(iceberg_sub, IcebergRowsRequestInputsConfig)
    assert iceberg_sub.namespace == "bronze_transparencia"
    assert iceberg_sub.table_name == "orgaos"
    assert iceberg_sub.columns == {"orgao_codigo": "codigo"}

    window_sub = source.access.request_inputs.inputs[1]
    assert isinstance(window_sub, DateWindowRequestInputsConfig)
    assert window_sub.start.isoformat() == "2025-01-01"
    assert window_sub.end.isoformat() == "2025-03-31"
    assert window_sub.step == "month"

    assert source.access.parameter_bindings is not None
    assert source.access.parameter_bindings["codigoOrgao"].from_ == "request_input.orgao_codigo"
    assert source.access.parameter_bindings["dataInicio"].from_ == "request_input.window_start"
    assert source.access.parameter_bindings["dataInicio"].format == "%Y-%m-%d"
    assert source.access.parameter_bindings["dataFinal"].from_ == "request_input.window_end"


def test_registry_rejects_combined_request_inputs_with_too_few_entries(tmp_path):
    source_yaml = _valid_source_yaml("combined_short_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  request_inputs:\n"
        "    type: combined\n"
        "    inputs:\n"
        "      - type: date_window\n"
        "        start: 2025-01-01\n"
        "        end: 2025-03-31\n"
        "        step: month\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"combined_short.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert (
        "access.request_inputs.inputs: must contain at least 2 entries when type is 'combined'"
        in str(exc_info.value)
    )


def test_registry_rejects_combined_request_inputs_with_field_name_conflicts(tmp_path):
    source_yaml = _valid_source_yaml("combined_conflict_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  request_inputs:\n"
        "    type: combined\n"
        "    inputs:\n"
        "      - type: date_window\n"
        "        start: 2025-01-01\n"
        "        end: 2025-03-31\n"
        "        step: month\n"
        "      - type: date_window\n"
        "        start: 2025-01-01\n"
        "        end: 2025-03-31\n"
        "        step: month\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"combined_conflict.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert "conflict with another input in this combined config" in str(exc_info.value)


def test_registry_rejects_combined_parameter_binding_referencing_unknown_field(tmp_path):
    source_yaml = _valid_source_yaml("combined_bad_binding_source", enabled=True).replace(
        "  auth:\n"
        "    type: none\n",
        "  request_inputs:\n"
        "    type: combined\n"
        "    inputs:\n"
        "      - type: iceberg_rows\n"
        "        namespace: bronze_transparencia\n"
        "        table_name: orgaos\n"
        "        columns:\n"
        "          orgao_codigo: codigo\n"
        "      - type: date_window\n"
        "        start: 2025-01-01\n"
        "        end: 2025-03-31\n"
        "        step: month\n"
        "  parameter_bindings:\n"
        "    bad_param:\n"
        "      from: request_input.nonexistent_field\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, {"combined_bad_binding.yaml": source_yaml})

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert (
        "must reference one of the combined input fields" in str(exc_info.value)
    )
