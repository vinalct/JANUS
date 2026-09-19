-- Spark SQL. Daily source volume and runtime, after append-only de-duplication.
WITH ranked_runs AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY emitted_at DESC
        ) AS row_rank
    FROM janus.metadata.runs
    WHERE emitted_at >= TIMESTAMP '2026-09-01 00:00:00'
),
latest_runs AS (
    SELECT *
    FROM ranked_runs
    WHERE row_rank = 1
      AND started_at >= TIMESTAMP '2026-09-01 00:00:00'
      AND started_at < TIMESTAMP '2026-10-01 00:00:00'
)
SELECT
    DATE_TRUNC('DAY', started_at) AS started_day,
    source_id,
    source_name,
    environment,
    strategy_family,
    strategy_variant,
    extraction_mode,
    source_hook,
    bronze_table_identifier,
    bronze_write_mode,
    COUNT(*) AS run_count,
    SUM(records_extracted) AS records_extracted,
    SUM(records_written) AS records_written,
    SUM(artifact_count) AS artifacts_persisted,
    AVG(duration_seconds) AS average_duration_seconds
FROM latest_runs
GROUP BY
    DATE_TRUNC('DAY', started_at),
    source_id,
    source_name,
    environment,
    strategy_family,
    strategy_variant,
    extraction_mode,
    source_hook,
    bronze_table_identifier,
    bronze_write_mode
ORDER BY started_day, source_id;
