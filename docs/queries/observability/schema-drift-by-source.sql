-- Spark SQL. Every run where the data contract changed what JANUS did, per source in start order:
-- a preflight that found the table would evolve, refused it, or could not reach the catalog,
-- a bronze table the writer changed, rows that did not parse into the contract's types, a new
-- contract declaration, or a run stopped by one of the four enforcement errors.
-- No window is applied. To add one, keep the half-open [start, end) UTC shape on started_at and
-- prune with emitted_at >= start, as quality-breaches-by-source.sql does.
WITH ranked_runs AS (
    SELECT
        run_id,
        source_id,
        started_at,
        emitted_at,
        status,
        schema_version,
        contract_id,
        contract_version,
        contract_preflight_outcome,
        schema_evolution,
        malformed_rows,
        error_type,
        failure_reason,
        validation_report_path,
        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY emitted_at DESC
        ) AS row_rank
    FROM janus.metadata.runs
),
latest_runs AS (
    SELECT
        *,
        LAG(schema_version) OVER (
            PARTITION BY source_id
            ORDER BY started_at, emitted_at, run_id
        ) AS previous_schema_version,
        LAG(contract_version) OVER (
            PARTITION BY source_id
            ORDER BY started_at, emitted_at, run_id
        ) AS previous_contract_version
    FROM ranked_runs
    WHERE row_rank = 1
)
SELECT
    source_id,
    run_id,
    started_at,
    contract_id,
    contract_version,
    previous_contract_version,
    schema_version,
    contract_preflight_outcome,
    schema_evolution,
    malformed_rows,
    status,
    error_type,
    failure_reason,
    validation_report_path
FROM latest_runs
WHERE contract_preflight_outcome IN ('will_evolve', 'refused', 'catalog_unavailable')
   OR schema_evolution <> 'none'
   OR malformed_rows > 0
   OR schema_version <> previous_schema_version
   OR error_type IN (
       'ContractViolationError',
       'MalformedRowsError',
       'ContractPreflightError',
       'SchemaEvolutionRefusedError'
   )
ORDER BY source_id, started_at, emitted_at, run_id;
