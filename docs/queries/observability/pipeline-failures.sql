-- Spark SQL. Replace the literal with the order-14 pipeline run to investigate.
WITH ranked_runs AS (
    SELECT
        run_id,
        pipeline_run_id,
        pipeline_attempt,
        trigger,
        source_id,
        source_name,
        status,
        started_at,
        emitted_at,
        failure_reason,
        error_type,
        run_metadata_path,
        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY emitted_at DESC
        ) AS row_rank
    FROM janus.metadata.runs
    WHERE pipeline_run_id = 'replace-with-pipeline-run-id'
)
SELECT
    pipeline_run_id,
    pipeline_attempt,
    trigger,
    source_id,
    source_name,
    run_id,
    started_at,
    emitted_at,
    failure_reason,
    error_type,
    run_metadata_path
FROM ranked_runs
WHERE row_rank = 1
  AND status = 'failed'
ORDER BY pipeline_attempt, source_id, run_id;
