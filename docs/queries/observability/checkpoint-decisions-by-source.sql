-- Spark SQL. NULL checkpoint_decision rows are runs where no checkpoint write was attempted.
-- `reset` is an operator action (`janus checkpoint set|clear`), never a run's own decision, so it
-- adds no row here: the checkpoint history entry records it, naming the operator and the reason.
WITH ranked_runs AS (
    SELECT
        run_id,
        source_id,
        source_name,
        checkpoint_field,
        checkpoint_strategy,
        checkpoint_value,
        checkpoint_decision,
        checkpoint_advanced,
        checkpoint_history_path,
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
    source_name,
    checkpoint_field,
    checkpoint_strategy,
    checkpoint_decision,
    checkpoint_advanced,
    COUNT(*) AS decision_count,
    MAX_BY(checkpoint_value, emitted_at) AS latest_checkpoint_value,
    SORT_ARRAY(COLLECT_SET(checkpoint_history_path)) AS checkpoint_history_paths
FROM ranked_runs
WHERE row_rank = 1
  AND checkpoint_decision IS NOT NULL
GROUP BY
    source_id,
    source_name,
    checkpoint_field,
    checkpoint_strategy,
    checkpoint_decision,
    checkpoint_advanced
ORDER BY source_id, checkpoint_decision;
