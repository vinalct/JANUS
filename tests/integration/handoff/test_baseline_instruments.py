"""Small real-Spark smoke for the measurement harness; 10,000 pages are opt-in."""

from tests.support.handoff_baseline import measure_pages


def test_measurement_harness_reads_real_pages_and_materializes_three_fresh_tables(spark, tmp_path):
    measured = measure_pages(spark, tmp_path, page_count=3)
    assert measured["baseline_read_partitions"] >= 1
    assert len(measured["baseline_materialize_seconds"]) == 3
    assert all(seconds > 0 for seconds in measured["baseline_materialize_seconds"])
    assert all(count > 0 for count in measured["baseline_bronze_file_count"])
    assert measured["baseline_bronze_row_count"] == [3, 3, 3]
    assert measured["baseline_synthetic_golden"]["row_count"] == "3"
    assert measured["sidecar_bytes"] == 3 * 65
