"""The ``metadata.runs`` declaration as real PyIceberg numbers it; needs pyarrow and pyiceberg."""

from __future__ import annotations

import pytest

import janus.observability.iceberg_sink as sink
from tests.unit.observability.test_pyiceberg_append_sink import V2_COLUMNS, V3_COLUMNS


def test_a_fresh_table_is_created_at_the_prefix_pyiceberg_numbers_as_declared():
    pytest.importorskip("pyarrow")
    schema_module = pytest.importorskip("pyiceberg.schema")
    dependencies = sink._load_engine_dependencies()
    creation = sink._creation_columns()
    created = schema_module.assign_fresh_schema_ids(
        sink._declared_schema(dependencies, creation)
    )
    declared = sink._schema_field_signatures(sink._declared_schema(dependencies))

    assert sink._schema_field_signatures(created) == declared[: len(creation)]
    plan = sink._plan_additive_evolution(sink._schema_field_signatures(created), declared)
    assert plan is not None
    assert [field[1] for field in plan.columns] == [*V2_COLUMNS, *V3_COLUMNS]
