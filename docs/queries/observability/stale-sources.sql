-- Spark SQL. Which sources have not succeeded inside their declared freshness window?
-- Mirrors janus status for sources represented in the table with matching declarations.
-- Bind window_start_day (UTC date), window_start/window_end (UTC timestamps), and
-- evaluated_at (UTC timestamp). The emission window is [window_start, window_end).
-- emitted_at predicates prune Iceberg's hidden day(emitted_at) partition emitted_at_day,
-- which is not a selectable column. Set window_start_day to the UTC date of window_start.
-- A success older than the window is invisible here: SQL reports never_ran if only failed
-- runs remain, while janus status scans the whole available history and finds that success.
-- Undeclared sources are reported but never breach. No row means no table evidence.
WITH ranked_runs AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY emitted_at DESC
        ) AS row_rank
    FROM janus.metadata.runs
    WHERE emitted_at >= CAST(:window_start_day AS TIMESTAMP)
      AND emitted_at >= :window_start
      AND emitted_at < :window_end
),
latest_runs AS (
    SELECT *
    FROM ranked_runs
    WHERE row_rank = 1
),
per_source AS (
    SELECT
        source_id,
        environment,
        MAX(CASE WHEN status = 'succeeded' THEN ended_at END) AS last_success_at,
        MAX(freshness_max_age_seconds) AS freshness_max_age_seconds,
        MAX(duration_budget_seconds) AS duration_budget_seconds,
        SUM(CASE WHEN duration_budget_outcome = 'exceeded' THEN 1 ELSE 0 END)
            AS budget_exceeded_runs
    FROM latest_runs
    GROUP BY source_id, environment
),
aged_sources AS (
    SELECT
        *,
        CAST(unix_micros(:evaluated_at) - unix_micros(last_success_at) AS DOUBLE)
            / 1000000 AS age_seconds
    FROM per_source
)
SELECT
    source_id,
    environment,
    last_success_at,
    freshness_max_age_seconds,
    duration_budget_seconds,
    budget_exceeded_runs,
    age_seconds,
    CASE
        WHEN freshness_max_age_seconds IS NULL THEN 'undeclared'
        WHEN last_success_at IS NULL THEN 'never_ran'
        WHEN age_seconds > freshness_max_age_seconds THEN 'stale'
        ELSE 'fresh'
    END AS outcome
FROM aged_sources
ORDER BY source_id, environment;
