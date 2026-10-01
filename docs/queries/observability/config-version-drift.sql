-- Spark SQL. One row for the first observed config or contract and each subsequent change.
-- What the contract then decided per run (preflight, schema evolution, malformed rows) is
-- answered by schema-drift-by-source.sql.
WITH ranked_runs AS (
    SELECT
        run_id,
        source_id,
        source_name,
        config_version,
        schema_version,
        contract_id,
        contract_version,
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
        ) AS previous_config_version,
        LAG(schema_version) OVER (
            PARTITION BY source_id
            ORDER BY started_at, emitted_at, run_id
        ) AS previous_schema_version
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
    previous_schema_version,
    schema_version,
    contract_id,
    contract_version,
    source_config_path
FROM latest_runs
WHERE previous_config_version IS NULL
   OR previous_config_version <> config_version
   OR NOT (previous_schema_version <=> schema_version)
ORDER BY source_id, changed_at, emitted_at;
