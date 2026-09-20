-- Spark SQL. The run window is [2026-09-01, 2026-10-01) in UTC and applies
-- to started_at. The emitted_at lower bound also prunes Iceberg day partitions.
WITH ranked_runs AS (
    SELECT
        run_id,
        source_id,
        source_name,
        status,
        started_at,
        ended_at,
        emitted_at,
        duration_seconds,
        failure_reason,
        failure_reason_truncated,
        failure_reason_length,
        error_type,
        run_metadata_path,
        lineage_path,
        record_schema_version,
        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY emitted_at DESC
        ) AS row_rank
    FROM janus.metadata.runs
    WHERE emitted_at >= TIMESTAMP '2026-09-01 00:00:00'
)
SELECT
    source_id,
    source_name,
    run_id,
    started_at,
    ended_at,
    emitted_at,
    duration_seconds,
    failure_reason,
    failure_reason_truncated,
    failure_reason_length,
    error_type,
    run_metadata_path,
    lineage_path,
    record_schema_version
FROM ranked_runs
WHERE row_rank = 1
  AND status = 'failed'
  AND started_at >= TIMESTAMP '2026-09-01 00:00:00'
  AND started_at < TIMESTAMP '2026-10-01 00:00:00'
ORDER BY started_at, source_id, run_id;
