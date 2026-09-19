-- Spark SQL. One row for the first observed config and each subsequent change.
WITH ranked_runs AS (
    SELECT
        run_id,
        source_id,
        source_name,
        config_version,
        source_config_path,
        started_at,
        emitted_at,
        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY emitted_at DESC
        ) AS row_rank
    FROM janus.metadata.runs
),
latest_runs AS (
    SELECT
        *,
        LAG(config_version) OVER (
            PARTITION BY source_id
            ORDER BY started_at, emitted_at, run_id
        ) AS previous_config_version
    FROM ranked_runs
    WHERE row_rank = 1
)
SELECT
    source_id,
    source_name,
    run_id,
    started_at AS changed_at,
    emitted_at,
    previous_config_version,
    config_version,
    source_config_path
FROM latest_runs
WHERE previous_config_version IS NULL
   OR previous_config_version <> config_version
ORDER BY source_id, changed_at, emitted_at;
