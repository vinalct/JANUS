"""Shared doubles for the scoped request-input session seam (api + catalog families)."""

from __future__ import annotations

from typing import Any

import pytest

from janus.runtime.spark_lifecycle import SparkSessionProvider


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
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = dict(values)

    def asDict(self, recursive: bool = False) -> dict[str, Any]:
        del recursive
        return dict(self._values)


class FakeDataFrame:
    """Just enough of the DataFrame surface that the iceberg_rows lookup uses."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
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

    def select(self, *selected_columns: FakeAliasedColumn) -> FakeDataFrame:
        projected_rows = [
            {selected.alias_name: row[selected.source_name] for selected in selected_columns}
            for row in self._rows
        ]
        return FakeDataFrame(
            projected_rows,
            columns=tuple(selected.alias_name for selected in selected_columns),
        )

    def distinct(self) -> FakeDataFrame:
        seen = set()
        unique_rows = []
        for row in self._rows:
            key = tuple((column_name, row[column_name]) for column_name in self.columns)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        return FakeDataFrame(unique_rows, columns=self.columns)

    def sort(self, *column_names: str) -> FakeDataFrame:
        sorted_rows = sorted(
            self._rows,
            key=lambda row: tuple(row[column_name] for column_name in column_names),
        )
        return FakeDataFrame(sorted_rows, columns=self.columns)

    def collect(self) -> list[FakeRow]:
        return [FakeRow(row) for row in self._rows]


class RecordingSparkSession:
    """Minimal ``spark.table(...)`` double that also records its own teardown."""

    def __init__(self, tables: dict[str, list[dict[str, Any]]]) -> None:
        self._tables = {
            identifier: FakeDataFrame(rows) for identifier, rows in tables.items()
        }
        self.catalog = _RecordingCatalog(self._tables)
        self.stopped = False

    def table(self, table_identifier: str) -> FakeDataFrame:
        return self._tables[table_identifier]

    def stop(self) -> None:
        self.stopped = True


class _RecordingCatalog:
    def __init__(self, tables: dict[str, FakeDataFrame]) -> None:
        self._tables = dict(tables)

    def tableExists(self, table_identifier: str) -> bool:
        return table_identifier in self._tables


class SpySparkSessionProvider(SparkSessionProvider):
    """A real provider over a fake session, with an event trail."""

    def __init__(self, session: Any, events: list[str]) -> None:
        super().__init__({}, {}, session_factory=lambda: session)
        self.events = events
        self.start_count = 0
        self.stop_count = 0
        self._live = False

    def get(self) -> Any:
        session = super().get()
        if not self._live:
            self._live = True
            self.start_count += 1
            self.events.append("spark_start")
        return session

    def stop(self) -> None:
        super().stop()
        if self._live:
            self._live = False
            self.stop_count += 1
            self.events.append("spark_stop")


@pytest.fixture
def spy_spark_provider():
    """Return a factory building a spy provider over the given fake tables."""

    def build(
        tables: dict[str, list[dict[str, Any]]],
        events: list[str],
    ) -> SpySparkSessionProvider:
        return SpySparkSessionProvider(RecordingSparkSession(tables), events)

    return build
