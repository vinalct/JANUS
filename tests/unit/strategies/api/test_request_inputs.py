from datetime import date

import pytest

from janus.models import (
    CombinedRequestInputsConfig,
    DateWindowRequestInputsConfig,
    IcebergRowsRequestInputsConfig,
    ParameterBinding,
    RequestInputsConfig,
)
from janus.strategies.api.request_inputs import (
    ApiParameterBindingError,
    ApiRequestInputLoadError,
    load_request_inputs,
    merge_request_params,
    resolve_parameter_bindings,
    validate_iceberg_request_input_source,
)


class FakeAliasedColumn:
    def __init__(self, source_name: str, alias_name: str) -> None:
        self.source_name = source_name
        self.alias_name = alias_name


class FakeColumn:
    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    def alias(self, alias_name: str) -> FakeAliasedColumn:
        return FakeAliasedColumn(self.source_name, alias_name)


class FakeRow:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = dict(values)

    def asDict(self, recursive: bool = False) -> dict[str, object]:
        del recursive
        return dict(self._values)


class FakeDataFrame:
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        columns: tuple[str, ...] | None = None,
    ) -> None:
        self._rows = [dict(row) for row in rows]
        if columns is not None:
            self.columns = columns
        elif rows:
            self.columns = tuple(rows[0])
        else:
            self.columns = ()

    def __getitem__(self, key: str) -> FakeColumn:
        return FakeColumn(key)

    def select(self, *selected_columns: FakeAliasedColumn) -> "FakeDataFrame":
        projected_rows = []
        for row in self._rows:
            projected_rows.append(
                {selected.alias_name: row[selected.source_name] for selected in selected_columns}
            )
        projected_column_names = tuple(selected.alias_name for selected in selected_columns)
        return FakeDataFrame(projected_rows, columns=projected_column_names)

    def distinct(self) -> "FakeDataFrame":
        seen = set()
        unique_rows = []
        for row in self._rows:
            key = tuple((column_name, row[column_name]) for column_name in self.columns)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        return FakeDataFrame(unique_rows, columns=self.columns)

    def sort(self, *column_names: str) -> "FakeDataFrame":
        sorted_rows = sorted(
            self._rows,
            key=lambda row: tuple(row[column_name] for column_name in column_names),
        )
        return FakeDataFrame(sorted_rows, columns=self.columns)

    def collect(self) -> list[FakeRow]:
        return [FakeRow(row) for row in self._rows]


class FakeCatalog:
    def __init__(self, tables: dict[str, FakeDataFrame]) -> None:
        self._tables = dict(tables)

    def tableExists(self, table_identifier: str) -> bool:
        return table_identifier in self._tables


class FakeSparkSession:
    def __init__(self, tables: dict[str, FakeDataFrame]) -> None:
        self._tables = dict(tables)
        self.catalog = FakeCatalog(self._tables)

    def table(self, table_identifier: str) -> FakeDataFrame:
        return self._tables[table_identifier]


def test_load_request_inputs_preserves_one_stream_behavior_for_none():
    assert load_request_inputs(RequestInputsConfig(type="none")) == (None,)


def test_load_request_inputs_generates_daily_windows():
    request_inputs = DateWindowRequestInputsConfig(
        type="date_window",
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        step="day",
    )

    assert load_request_inputs(request_inputs) == (
        {
            "window_start": date(2025, 1, 1),
            "window_end": date(2025, 1, 1),
        },
        {
            "window_start": date(2025, 1, 2),
            "window_end": date(2025, 1, 2),
        },
        {
            "window_start": date(2025, 1, 3),
            "window_end": date(2025, 1, 3),
        },
    )


def test_load_request_inputs_generates_monthly_windows_bounded_by_interval():
    request_inputs = DateWindowRequestInputsConfig(
        type="date_window",
        start=date(2025, 1, 15),
        end=date(2025, 3, 10),
        step="month",
    )

    assert load_request_inputs(request_inputs) == (
        {
            "window_start": date(2025, 1, 15),
            "window_end": date(2025, 1, 31),
        },
        {
            "window_start": date(2025, 2, 1),
            "window_end": date(2025, 2, 28),
        },
        {
            "window_start": date(2025, 3, 1),
            "window_end": date(2025, 3, 10),
        },
    )


def test_resolve_parameter_bindings_supports_checkpoint_value():
    resolved = resolve_parameter_bindings(
        {"updated_at": ParameterBinding(from_="checkpoint_value")},
        checkpoint_value="2025-01-31T00:00:00Z",
    )

    assert resolved == {"updated_at": "2025-01-31T00:00:00Z"}


def test_resolve_parameter_bindings_formats_generated_date_window_values():
    bindings = {
        "dataIdaDe": ParameterBinding(
            from_="request_input.window_start",
            format="%Y-%m-%d",
        ),
        "mesAno": ParameterBinding(
            from_="request_input.window_end",
            format="%Y%m",
        ),
    }
    request_input = load_request_inputs(
        DateWindowRequestInputsConfig(
            type="date_window",
            start=date(2025, 1, 1),
            end=date(2025, 1, 31),
            step="month",
        )
    )[0]

    resolved = resolve_parameter_bindings(bindings, request_input=request_input)

    assert resolved == {
        "dataIdaDe": "2025-01-01",
        "mesAno": "202501",
    }


def test_resolve_parameter_bindings_rejects_missing_request_input_fields():
    with pytest.raises(ApiParameterBindingError) as exc_info:
        resolve_parameter_bindings(
            {"id": ParameterBinding(from_="request_input.emenda_id")},
            request_input={"other_id": "123"},
        )

    assert str(exc_info.value) == (
        "access.parameter_bindings.id.from: request_input.emenda_id is not available "
        "in the current request input"
    )


def test_resolve_parameter_bindings_rejects_invalid_date_formats():
    with pytest.raises(ApiParameterBindingError) as exc_info:
        resolve_parameter_bindings(
            {
                "mesAno": ParameterBinding(
                    from_="request_input.window_end",
                    format="%Q",
                )
            },
            request_input={"window_end": date(2025, 1, 31)},
        )

    assert str(exc_info.value) == (
        "access.parameter_bindings.mesAno.format: contains unsupported strftime directive '%Q'"
    )


def test_resolve_parameter_bindings_rejects_formatted_non_date_values():
    with pytest.raises(ApiParameterBindingError) as exc_info:
        resolve_parameter_bindings(
            {
                "mesAno": ParameterBinding(
                    from_="request_input.window_end",
                    format="%Y%m",
                )
            },
            request_input={"window_end": "2025-01-31"},
        )

    assert str(exc_info.value) == (
        "access.parameter_bindings.mesAno.format: requires a date or datetime value, got str"
    )


def test_resolve_parameter_bindings_rejects_non_scalar_values():
    with pytest.raises(ApiParameterBindingError) as exc_info:
        resolve_parameter_bindings(
            {"id": ParameterBinding(from_="request_input.emenda_id")},
            request_input={"emenda_id": {"id": "123"}},
        )

    assert str(exc_info.value) == (
        "access.parameter_bindings.id.from: resolved to unsupported type dict; "
        "expected a scalar, date, or datetime"
    )


def test_merge_request_params_rejects_duplicate_keys():
    with pytest.raises(ApiParameterBindingError) as exc_info:
        merge_request_params({"id": "fixed"}, {"id": "runtime"})

    assert str(exc_info.value) == (
        "access.params and access.parameter_bindings must not define the same "
        "request parameter(s): id"
    )


def test_validate_iceberg_request_input_source_rejects_missing_tables():
    request_inputs = IcebergRowsRequestInputsConfig(
        type="iceberg_rows",
        upstream_source_id="emendas_parlamentares__emendas",
        namespace="bronze_transparencia",
        table_name="emendas_parlamentares__emendas",
        columns={"emenda_id": "id"},
    )

    with pytest.raises(ApiRequestInputLoadError) as exc_info:
        validate_iceberg_request_input_source(request_inputs, available_columns=None)

    assert str(exc_info.value) == (
        "access.request_inputs: configured Iceberg table "
        "'bronze_transparencia.emendas_parlamentares__emendas' does not exist"
    )


def test_validate_iceberg_request_input_source_rejects_missing_columns():
    request_inputs = IcebergRowsRequestInputsConfig(
        type="iceberg_rows",
        upstream_source_id="emendas_parlamentares__emendas",
        namespace="bronze_transparencia",
        table_name="emendas_parlamentares__emendas",
        columns={"emenda_id": "id"},
    )

    with pytest.raises(ApiRequestInputLoadError) as exc_info:
        validate_iceberg_request_input_source(
            request_inputs,
            available_columns=("codigo",),
        )

    assert str(exc_info.value) == (
        "access.request_inputs.columns.emenda_id: source column 'id' was not found "
        "in Iceberg table 'bronze_transparencia.emendas_parlamentares__emendas'"
    )


def test_load_request_inputs_reads_projected_iceberg_rows_with_binding_field_names():
    request_inputs = IcebergRowsRequestInputsConfig(
        type="iceberg_rows",
        upstream_source_id="emendas_parlamentares__emendas",
        namespace="bronze_transparencia",
        table_name="emendas_parlamentares__emendas",
        columns={
            "autor": "parlamentar",
            "emenda_id": "id",
        },
    )
    spark = FakeSparkSession(
        {
            "bronze_transparencia.emendas_parlamentares__emendas": FakeDataFrame(
                [
                    {"id": "2", "parlamentar": "Bruno", "ignored": "x"},
                    {"id": "1", "parlamentar": "Ana", "ignored": "y"},
                ]
            )
        }
    )

    assert load_request_inputs(request_inputs, spark=spark) == (
        {
            "autor": "Ana",
            "emenda_id": "1",
        },
        {
            "autor": "Bruno",
            "emenda_id": "2",
        },
    )


def test_load_request_inputs_applies_distinct_to_projected_iceberg_rows_only():
    request_inputs = IcebergRowsRequestInputsConfig(
        type="iceberg_rows",
        upstream_source_id="emendas_parlamentares__emendas",
        namespace="bronze_transparencia",
        table_name="emendas_parlamentares__emendas",
        columns={"emenda_id": "id"},
        distinct=True,
    )
    spark = FakeSparkSession(
        {
            "bronze_transparencia.emendas_parlamentares__emendas": FakeDataFrame(
                [
                    {"id": "2", "parlamentar": "Bruno"},
                    {"id": "1", "parlamentar": "Ana"},
                    {"id": "1", "parlamentar": "Carla"},
                ]
            )
        }
    )

    assert load_request_inputs(request_inputs, spark=spark) == (
        {"emenda_id": "1"},
        {"emenda_id": "2"},
    )


def test_load_request_inputs_rejects_iceberg_rows_without_spark():
    request_inputs = IcebergRowsRequestInputsConfig(
        type="iceberg_rows",
        upstream_source_id="emendas_parlamentares__emendas",
        namespace="bronze_transparencia",
        table_name="emendas_parlamentares__emendas",
        columns={"emenda_id": "id"},
    )

    with pytest.raises(ApiRequestInputLoadError) as exc_info:
        load_request_inputs(request_inputs)

    assert str(exc_info.value) == (
        "access.request_inputs: iceberg_rows inputs require an active SparkSession"
    )


def test_load_request_inputs_combined_computes_cartesian_product_of_iceberg_and_date_window():
    iceberg_input = IcebergRowsRequestInputsConfig(
        type="iceberg_rows",
        upstream_source_id="orgaos",
        namespace="bronze_transparencia",
        table_name="orgaos",
        columns={"orgao_codigo": "codigo"},
    )
    window_input = DateWindowRequestInputsConfig(
        type="date_window",
        start=date(2025, 1, 1),
        end=date(2025, 2, 28),
        step="month",
    )
    request_inputs = CombinedRequestInputsConfig(
        type="combined",
        inputs=(iceberg_input, window_input),
    )
    spark = FakeSparkSession(
        {
            "bronze_transparencia.orgaos": FakeDataFrame(
                [
                    {"codigo": "26000"},
                    {"codigo": "52000"},
                ]
            )
        }
    )

    result = load_request_inputs(request_inputs, spark=spark)

    assert result == (
        {
            "orgao_codigo": "26000",
            "window_start": date(2025, 1, 1),
            "window_end": date(2025, 1, 31),
        },
        {
            "orgao_codigo": "26000",
            "window_start": date(2025, 2, 1),
            "window_end": date(2025, 2, 28),
        },
        {
            "orgao_codigo": "52000",
            "window_start": date(2025, 1, 1),
            "window_end": date(2025, 1, 31),
        },
        {
            "orgao_codigo": "52000",
            "window_start": date(2025, 2, 1),
            "window_end": date(2025, 2, 28),
        },
    )


def test_load_request_inputs_combined_resolves_bindings_from_merged_context():
    bindings = {
        "codigoOrgao": ParameterBinding(from_="request_input.orgao_codigo"),
        "mesAno": ParameterBinding(from_="request_input.window_end", format="%Y%m"),
    }
    iceberg_input = IcebergRowsRequestInputsConfig(
        type="iceberg_rows",
        upstream_source_id="orgaos",
        namespace="bronze_transparencia",
        table_name="orgaos",
        columns={"orgao_codigo": "codigo"},
    )
    window_input = DateWindowRequestInputsConfig(
        type="date_window",
        start=date(2025, 1, 1),
        end=date(2025, 1, 31),
        step="month",
    )
    request_inputs = CombinedRequestInputsConfig(
        type="combined",
        inputs=(iceberg_input, window_input),
    )
    spark = FakeSparkSession(
        {"bronze_transparencia.orgaos": FakeDataFrame([{"codigo": "26000"}])}
    )

    combined_contexts = load_request_inputs(request_inputs, spark=spark)

    resolved = resolve_parameter_bindings(bindings, request_input=combined_contexts[0])

    assert resolved == {"codigoOrgao": "26000", "mesAno": "202501"}
