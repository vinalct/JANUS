-- Spark SQL. The run window is [2026-09-01, 2026-10-01) in UTC and applies
-- to started_at. Failed check names and validation-report paths keep the answer actionable.
WITH ranked_runs AS (
    SELECT
        run_id,
        source_id,
        quality_outcome,
        quality_checks_passed,
        quality_checks_failed,
        quality_checks_skipped,
        quality_failed_checks,
        validation_report_path,
        started_at,
        emitted_at,
        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY emitted_at DESC
        ) AS row_rank
    FROM janus.metadata.runs
    WHERE emitted_at >= TIMESTAMP '2026-09-01 00:00:00'
)
SELECT
    source_id,
    COUNT(*) AS breached_runs,
    SUM(quality_checks_passed) AS checks_passed,
    SUM(quality_checks_failed) AS checks_failed,
    SUM(quality_checks_skipped) AS checks_skipped,
    SORT_ARRAY(ARRAY_DISTINCT(FLATTEN(COLLECT_LIST(quality_failed_checks)))) AS failed_checks,
    SORT_ARRAY(COLLECT_SET(validation_report_path)) AS validation_report_paths
FROM ranked_runs
WHERE row_rank = 1
  AND quality_outcome = 'failed'
  AND started_at >= TIMESTAMP '2026-09-01 00:00:00'
  AND started_at < TIMESTAMP '2026-10-01 00:00:00'
GROUP BY source_id
ORDER BY breached_runs DESC, source_id;
