"""How the bronze writer executes a full refresh, proven without a Spark session.

The mechanism decision is already covered purely in ``test_overwrite_plan.py``; what is left
to pin is the *wiring* — which statements the writer emits, in which order, what it records in
the write metadata, and that the session config it borrows is handed back. A stub session that
records SQL strings covers all of that on a host with no PySpark, so the fast gate catches a
regression that would otherwise need the Iceberg suite to surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from janus.models import BronzeWriteIntent, ExecutionPlan, RunContext, SourceConfig
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter

PARTITION_OVERWRITE_MODE = "spark.sql.sources.partitionOverwriteMode"

ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}

OVERWRITE_INTENT = BronzeWriteIntent(
    strategy="replace_table",
    configured_mode="overwrite",
    partition_columns=("ingestion_date",),
)

# A normalized bronze batch, trimmed to what the writer actually inspects.
BASE_COLUMNS = (
    ("event_id", "string"),
    ("amount", "bigint"),
    ("ingestion_date", "date"),
)
PARTITIONS = ("ingestion_date",)


class FakeType:
    """Stands in for a Spark ``DataType`` — either a scalar or a partition struct."""

    def __init__(self, simple: str, fields: tuple[FakeField, ...] = ()) -> None:
        self._simple = simple
        self.fields = fields

    def simpleString(self) -> str:
        return self._simple


@dataclass(frozen=True)
class FakeField:
    name: str
    dataType: FakeType


class FakeSchema:
    def __init__(self, fields: tuple[FakeField, ...]) -> None:
        self.fields = fields

    @classmethod
    def of_columns(cls, columns: tuple[tuple[str, str], ...]) -> FakeSchema:
        return cls(tuple(FakeField(name, FakeType(type_)) for name, type_ in columns))


class FakeTable:
    def __init__(self, schema: Any) -> None:
        self.schema = schema


class FakeCatalog:
    def __init__(self, session: FakeSparkSession) -> None:
        self._session = session

    def tableExists(self, identifier: str) -> bool:
        self._session.table_exists_calls.append(identifier)
        return self._session.table_exists

    def dropTempView(self, name: str) -> None:
        self._session.dropped_temp_views.append(name)


class FakeConf:
    def __init__(self, session: FakeSparkSession, values: dict[str, str]) -> None:
        self._session = session
        self._values = dict(values)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._values.get(key, default)

    def set(self, key: str, value: str) -> None:
        self._values[key] = value
        self._session.operations.append(("conf_set", key, value))

    def unset(self, key: str) -> None:
        self._values.pop(key, None)
        self._session.operations.append(("conf_unset", key))

    def as_dict(self) -> dict[str, str]:
        return dict(self._values)


class FakeSparkSession:
    """Records every statement and session-config mutation the writer performs."""

    def __init__(
        self,
        *,
        table_exists: bool = True,
        target_columns: tuple[tuple[str, str], ...] = BASE_COLUMNS,
        target_partitions: tuple[str, ...] | None = PARTITIONS,
        conf_values: dict[str, str] | None = None,
        failing_statement: str | None = None,
    ) -> None:
        self.table_exists = table_exists
        self.target_columns = target_columns
        self.target_partitions = target_partitions
        self.failing_statement = failing_statement
        self.operations: list[tuple[str, ...]] = []
        self.table_calls: list[str] = []
        self.table_exists_calls: list[str] = []
        self.dropped_temp_views: list[str] = []
        self.catalog = FakeCatalog(self)
        self.conf = FakeConf(self, conf_values or {})

    @property
    def statements(self) -> list[str]:
        return [operation[1] for operation in self.operations if operation[0] == "sql"]

    def sql(self, statement: str) -> None:
        self.operations.append(("sql", statement))
        if self.failing_statement and self.failing_statement in statement:
            raise RuntimeError("commit failed")

    def table(self, identifier: str) -> FakeTable:
        self.table_calls.append(identifier)
        if identifier.endswith(".partitions"):
            if self.target_partitions is None:
                raise RuntimeError("partitions metadata table is unavailable")
            partition_struct = FakeType(
                "struct",
                tuple(
                    FakeField(column, FakeType("date"))
                    for column in self.target_partitions
                ),
            )
            return FakeTable(
                FakeSchema(
                    (
                        FakeField("partition", partition_struct),
                        FakeField("record_count", FakeType("bigint")),
                    )
                )
            )
        return FakeTable(FakeSchema.of_columns(self.target_columns))


class FakeRDD:
    def getNumPartitions(self) -> int:
        return 4


class FakeDataFrame:
    def __init__(
        self,
        session: FakeSparkSession,
        columns: tuple[tuple[str, str], ...] = BASE_COLUMNS,
    ) -> None:
        self.sparkSession = session
        self.schema = FakeSchema.of_columns(columns)
        self.rdd = FakeRDD()
        self.temp_views: list[str] = []

    def createOrReplaceTempView(self, name: str) -> None:
        self.temp_views.append(name)

    def count(self) -> int:
        return 2


def _write(
    tmp_path: Path,
    session: FakeSparkSession,
    *,
    source_columns: tuple[tuple[str, str], ...] = BASE_COLUMNS,
    partition_by: list[str] | None = None,
    source_id: str = "bronze_overwrite_fixture",
):
    plan = _plan(tmp_path, partition_by=partition_by, source_id=source_id)
    writer = SparkDatasetWriter(
        StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path)
    )
    frame = FakeDataFrame(session, source_columns)
    return writer.write(
        frame, plan, "bronze", intent=OVERWRITE_INTENT, count_records=True
    )


def test_matching_schema_emits_insert_overwrite_and_no_replace(tmp_path):
    session = FakeSparkSession()

    _write(tmp_path, session)

    assert any("INSERT OVERWRITE" in statement for statement in session.statements)
    assert not any("REPLACE TABLE" in statement for statement in session.statements)


def test_first_write_still_creates_the_table(tmp_path):
    session = FakeSparkSession(table_exists=False)

    _write(tmp_path, session)

    assert any("CREATE TABLE" in statement for statement in session.statements)
    assert not any("INSERT OVERWRITE" in statement for statement in session.statements)
    # A first write has no history to preserve and no target schema to reconcile against.
    assert session.table_calls == []


def test_added_column_emits_alter_then_insert_overwrite(tmp_path):
    session = FakeSparkSession()

    result = _write(
        tmp_path, session, source_columns=(*BASE_COLUMNS, ("extra", "string"))
    )

    kinds = [
        "alter" if "ADD COLUMNS" in statement else "insert"
        for statement in session.statements
        if "ADD COLUMNS" in statement or "INSERT OVERWRITE" in statement
    ]
    assert kinds == ["alter", "insert"]
    assert "`extra` string" in next(
        statement for statement in session.statements if "ADD COLUMNS" in statement
    )
    assert result.metadata_as_dict()["overwrite_mechanism"] == "insert_overwrite"
    assert "history_reset_reason" not in result.metadata_as_dict()


def test_dropped_column_falls_back_to_replace_table(tmp_path):
    session = FakeSparkSession(target_columns=(*BASE_COLUMNS, ("legacy", "string")))

    result = _write(tmp_path, session)

    assert any("REPLACE TABLE" in statement for statement in session.statements)
    assert not any("INSERT OVERWRITE" in statement for statement in session.statements)
    metadata = result.metadata_as_dict()
    assert metadata["overwrite_mechanism"] == "replace_table"
    assert "legacy" in metadata["history_reset_reason"]


def test_unreadable_partition_spec_falls_back(tmp_path):
    session = FakeSparkSession(target_partitions=None)

    result = _write(tmp_path, session)

    assert any("REPLACE TABLE" in statement for statement in session.statements)
    assert (
        result.metadata_as_dict()["history_reset_reason"]
        == "target partition spec could not be read"
    )


def test_unpartitioned_table_is_a_known_empty_spec_not_a_fallback(tmp_path):
    session = FakeSparkSession(target_partitions=())

    result = _write(tmp_path, session, partition_by=[])

    assert any("INSERT OVERWRITE" in statement for statement in session.statements)
    assert result.metadata_as_dict()["overwrite_mechanism"] == "insert_overwrite"


def test_partition_overwrite_mode_is_pinned_and_restored(tmp_path):
    session = FakeSparkSession(conf_values={PARTITION_OVERWRITE_MODE: "dynamic"})

    _write(tmp_path, session)

    pin = ("conf_set", PARTITION_OVERWRITE_MODE, "static")
    insert = next(
        operation
        for operation in session.operations
        if operation[0] == "sql" and "INSERT OVERWRITE" in operation[1]
    )
    assert session.operations.index(pin) < session.operations.index(insert)
    # A session profile that pins `dynamic` globally gets it back untouched.
    assert session.conf.as_dict()[PARTITION_OVERWRITE_MODE] == "dynamic"
    assert session.operations[-1] == ("conf_set", PARTITION_OVERWRITE_MODE, "dynamic")


def test_partition_overwrite_mode_is_unset_again_when_it_was_absent(tmp_path):
    session = FakeSparkSession()

    _write(tmp_path, session)

    assert ("conf_set", PARTITION_OVERWRITE_MODE, "static") in session.operations
    assert ("conf_unset", PARTITION_OVERWRITE_MODE) in session.operations
    assert PARTITION_OVERWRITE_MODE not in session.conf.as_dict()


def test_partition_overwrite_mode_restored_when_the_statement_raises(tmp_path):
    session = FakeSparkSession(
        conf_values={PARTITION_OVERWRITE_MODE: "dynamic"},
        failing_statement="INSERT OVERWRITE",
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        _write(tmp_path, session)

    assert session.conf.as_dict()[PARTITION_OVERWRITE_MODE] == "dynamic"


def test_write_result_mode_is_unchanged(tmp_path):
    session = FakeSparkSession()

    result = _write(tmp_path, session)

    # The mechanism is an implementation detail; `mode` still reports the Iceberg mode so
    # run metadata, lineage and the CLI summary are untouched.
    assert result.mode == "overwrite"
    assert result.records_written == 2
    assert result.partition_by == PARTITIONS
    assert result.metadata_as_dict()["overwrite_mechanism"] == "insert_overwrite"


def test_temp_view_is_dropped_on_both_paths(tmp_path):
    insert_session = FakeSparkSession()
    _write(tmp_path, insert_session)

    fallback_session = FakeSparkSession(target_partitions=("event_id",))
    _write(tmp_path, fallback_session)

    for session in (insert_session, fallback_session):
        assert len(session.dropped_temp_views) == 1
        assert session.dropped_temp_views[0].startswith("janus_bronze_")


def test_temp_view_is_dropped_when_the_statement_raises(tmp_path):
    session = FakeSparkSession(failing_statement="INSERT OVERWRITE")

    with pytest.raises(RuntimeError, match="commit failed"):
        _write(tmp_path, session)

    assert len(session.dropped_temp_views) == 1


def test_dotted_source_id_stages_through_a_single_quoted_view(tmp_path):
    """Regression: an unsanitized name would be quoted as `janus_bronze_receita`.`federal_...`.

    Spark would then look for a temp view inside a namespace nobody configured and fail the
    write with an AnalysisException naming a table that does not exist.
    """
    session = FakeSparkSession()

    result = _write(tmp_path, session, source_id="receita.federal.cnpj")

    view_name = session.dropped_temp_views[0]
    assert "." not in view_name
    insert = next(
        statement for statement in session.statements if "INSERT OVERWRITE" in statement
    )
    assert f"`{view_name}`" in insert
    assert result.metadata_as_dict()["overwrite_mechanism"] == "insert_overwrite"


def _plan(
    tmp_path: Path,
    *,
    partition_by: list[str] | None = None,
    source_id: str = "bronze_overwrite_fixture",
) -> ExecutionPlan:
    run_context = RunContext.create(
        run_id="run-bronze-overwrite-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 7, 8, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(
        _source_config(tmp_path, partition_by=partition_by, source_id=source_id),
        run_context,
    )


def _source_config(
    tmp_path: Path,
    *,
    partition_by: list[str] | None,
    source_id: str = "bronze_overwrite_fixture",
) -> SourceConfig:
    payload: dict[str, Any] = {
        "source_id": source_id,
        "name": source_id,
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": {
            "base_url": "https://example.invalid",
            "path": "/events",
            "method": "GET",
            "format": "json",
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 10,
            },
            "rate_limit": {
                "requests_per_minute": None,
                "concurrency": 1,
                "backoff_seconds": 5,
            },
        },
        "extraction": {
            "mode": "full_refresh",
            "dead_letter_max_items": 0,
            "retry": {
                "max_attempts": 3,
                "backoff_strategy": "fixed",
                "backoff_seconds": 1,
            },
        },
        "schema": {"mode": "infer"},
        "spark": {
            "input_format": "json",
            "write_mode": "overwrite",
            "repartition": 1,
            "partition_by": ["ingestion_date"] if partition_by is None else partition_by,
        },
        "outputs": {
            "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
            "bronze": {
                "path": f"data/bronze/example/{source_id}",
                "format": "iceberg",
                "namespace": "bronze_test",
                "table_name": source_id,
            },
            "metadata": {
                "path": f"data/metadata/example/{source_id}",
                "format": "json",
            },
        },
        "quality": {
            "required_fields": ["event_id"],
            "unique_fields": [],
            "allow_schema_evolution": True,
        },
    }
    config_path = tmp_path / "conf" / "sources" / f"{source_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return SourceConfig.from_mapping(payload, config_path)
