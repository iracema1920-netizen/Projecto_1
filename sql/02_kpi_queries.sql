/* ============================================================================
   OGP1 Reliability - KPI QUERIES
   ----------------------------------------------------------------------------
   Each query starts with a line "-- name: <id>". The Python script
   src/02_load_sql.py runs every query and saves the result to
   reports/sql_results/<id>.csv, so you can also open the answers in Excel.

   You can run any single query yourself in DB Browser for SQLite,
   DBeaver or VS Code (SQLite extension) against data/processed/ogp1_reliability.db

   SQL skills shown: CTEs, CASE pivots, LEFT JOINs, aggregations,
   window functions (SUM OVER, LAG, RANK), correlated subqueries, date maths.
   ========================================================================== */


-- name: q01_table_row_counts
-- Sanity check after loading: how many rows landed in each table?
SELECT 'dim_asset' AS table_name, COUNT(*) AS n_rows FROM dim_asset
UNION ALL SELECT 'dim_tag', COUNT(*) FROM dim_tag
UNION ALL SELECT 'fact_sensor_hourly', COUNT(*) FROM fact_sensor_hourly
UNION ALL SELECT 'fact_run_status', COUNT(*) FROM fact_run_status
UNION ALL SELECT 'fact_event', COUNT(*) FROM fact_event
UNION ALL SELECT 'fact_alarm', COUNT(*) FROM fact_alarm;


-- name: q02_operating_hours_by_status
-- Hours spent in each state per rotating asset (CASE WHEN = SQL pivot table)
SELECT
    asset_id,
    SUM(CASE WHEN status = 'Running'     THEN 1 ELSE 0 END) AS running_h,
    SUM(CASE WHEN status = 'Stopped'     THEN 1 ELSE 0 END) AS stopped_standby_h,
    SUM(CASE WHEN status = 'Maintenance' THEN 1 ELSE 0 END) AS maintenance_h,
    SUM(CASE WHEN status = 'Tripped'     THEN 1 ELSE 0 END) AS tripped_h,
    SUM(CASE WHEN status = 'Unknown'     THEN 1 ELSE 0 END) AS unknown_h,
    COUNT(*)                                                AS calendar_h
FROM fact_run_status
GROUP BY asset_id
ORDER BY asset_id;


-- name: q03_reliability_kpis
-- Core reliability KPIs per rotating asset
--   MTBF = running hours / number of failures            (higher is better)
--   MTTR = average downtime per failure                  (lower is better)
--   Availability = (calendar - maintenance - tripped) / calendar
--   Inherent availability Ai = MTBF / (MTBF + MTTR)
WITH hours AS (
    SELECT asset_id,
           SUM(CASE WHEN status = 'Running' THEN 1 ELSE 0 END)                        AS running_h,
           SUM(CASE WHEN status IN ('Maintenance','Tripped') THEN 1 ELSE 0 END)        AS unavailable_h,
           COUNT(*)                                                                    AS calendar_h
    FROM fact_run_status
    GROUP BY asset_id
),
fails AS (
    SELECT asset_id,
           SUM(is_failure)                                        AS n_failures,
           SUM(is_unplanned)                                      AS n_unplanned_events,
           SUM(CASE WHEN is_failure = 1 THEN Downtime_h END)      AS failure_downtime_h,
           AVG(CASE WHEN is_failure = 1 THEN Downtime_h END)      AS mttr_h
    FROM fact_event
    GROUP BY asset_id
)
SELECT
    h.asset_id,
    a.equipment_class,
    a.criticality,
    h.running_h,
    COALESCE(f.n_failures, 0)                                         AS n_failures,
    COALESCE(f.n_unplanned_events, 0)                                 AS n_unplanned_events,
    ROUND(1.0 * h.running_h / NULLIF(f.n_failures, 0), 0)             AS mtbf_h,
    ROUND(f.mttr_h, 1)                                                AS mttr_h,
    ROUND(100.0 * (h.calendar_h - h.unavailable_h) / h.calendar_h, 2) AS availability_pct,
    ROUND(100.0 * (1.0 * h.running_h / f.n_failures)
          / ((1.0 * h.running_h / f.n_failures) + f.mttr_h), 2)       AS inherent_availability_pct
FROM hours h
JOIN dim_asset a ON a.asset_id = h.asset_id
LEFT JOIN fails f ON f.asset_id = h.asset_id
ORDER BY mtbf_h;


-- name: q04_failure_mode_pareto
-- Which failure modes cost the most downtime? (Pareto 80/20 with a window function)
WITH fm AS (
    SELECT asset_id,
           FailureModeCode,
           FailureMode,
           COUNT(*)        AS n_events,
           SUM(Downtime_h) AS downtime_h
    FROM fact_event
    WHERE is_unplanned = 1
    GROUP BY asset_id, FailureModeCode, FailureMode
)
SELECT
    asset_id,
    FailureModeCode,
    FailureMode,
    n_events,
    downtime_h,
    ROUND(100.0 * downtime_h / SUM(downtime_h) OVER (), 1)                         AS pct_of_downtime,
    ROUND(100.0 * SUM(downtime_h) OVER (ORDER BY downtime_h DESC
                                        ROWS UNBOUNDED PRECEDING)
                / SUM(downtime_h) OVER (), 1)                                      AS cumulative_pct
FROM fm
ORDER BY downtime_h DESC;


-- name: q05_downtime_by_year_asset
-- Is reliability getting better or worse? Unplanned downtime by year
SELECT
    asset_id,
    SUM(CASE WHEN year = 2023 THEN Downtime_h ELSE 0 END) AS downtime_2023_h,
    SUM(CASE WHEN year = 2024 THEN Downtime_h ELSE 0 END) AS downtime_2024_h,
    SUM(CASE WHEN year = 2025 THEN Downtime_h ELSE 0 END) AS downtime_2025_h,
    SUM(Downtime_h)                                       AS total_downtime_h,
    COUNT(*)                                              AS n_events
FROM fact_event
WHERE is_unplanned = 1
GROUP BY asset_id
ORDER BY total_downtime_h DESC;


-- name: q06_pf_interval_by_failure_mode
-- P-F interval = time from first detectable degradation (P) to functional failure (F).
-- This is the "warning window" a predictive maintenance program can exploit.
SELECT
    FailureModeCode,
    FailureMode,
    COUNT(*)                         AS n_events,
    ROUND(AVG(pf_interval_days), 1)  AS avg_pf_days,
    ROUND(MIN(pf_interval_days), 1)  AS min_pf_days,
    ROUND(MAX(pf_interval_days), 1)  AS max_pf_days,
    ROUND(AVG(Downtime_h), 0)        AS avg_downtime_h
FROM fact_event
WHERE has_degradation_onset = 1
GROUP BY FailureModeCode, FailureMode
ORDER BY avg_pf_days DESC;


-- name: q07_time_between_failures
-- Calendar days between consecutive failures of the same asset (LAG window function)
SELECT
    asset_id,
    EventID,
    FailureModeCode,
    EventStart,
    LAG(EventStart) OVER (PARTITION BY asset_id ORDER BY EventStart)        AS previous_failure,
    ROUND(julianday(EventStart)
          - julianday(LAG(EventStart) OVER (PARTITION BY asset_id ORDER BY EventStart)), 1)
                                                                            AS days_since_previous
FROM fact_event
WHERE is_failure = 1
  AND asset_id IN ('K-101', 'GT-101', 'P-201A', 'P-201B')
ORDER BY asset_id, EventStart;


-- name: q08_alarm_load_kpis
-- Alarm system performance vs. ISA-18.2 / EEMUA 191 benchmarks
-- (benchmark: average < 144 alarms/day per operator console = ~6 per hour)
WITH daily AS (
    SELECT date, COUNT(*) AS n_alarms
    FROM fact_alarm
    GROUP BY date
)
SELECT
    (SELECT COUNT(*) FROM fact_alarm)                                     AS total_alarms,
    ROUND((SELECT COUNT(*) FROM fact_alarm) / 1095.0, 2)                  AS avg_alarms_per_day,
    MAX(n_alarms)                                                          AS max_alarms_in_a_day,
    COUNT(*)                                                               AS days_with_alarms,
    (SELECT SUM(is_standing) FROM fact_alarm)                             AS standing_alarms_24h_plus,
    (SELECT ROUND(100.0 * SUM(CASE WHEN Priority = 'High' THEN 1 ELSE 0 END) / COUNT(*), 1)
       FROM fact_alarm)                                                    AS pct_high_priority
FROM daily;


-- name: q09_bad_actor_alarms
-- Top 10 "bad actor" tags: a few tags usually create most of the alarms
WITH t AS (
    SELECT Tag, asset_id, Descriptor, COUNT(*) AS n_alarms,
           ROUND(AVG(Duration_h), 1) AS avg_duration_h
    FROM fact_alarm
    GROUP BY Tag, asset_id, Descriptor
)
SELECT
    RANK() OVER (ORDER BY n_alarms DESC)                                       AS rank_no,
    Tag, asset_id, Descriptor, n_alarms, avg_duration_h,
    ROUND(100.0 * n_alarms / SUM(n_alarms) OVER (), 1)                         AS pct_of_alarms,
    ROUND(100.0 * SUM(n_alarms) OVER (ORDER BY n_alarms DESC ROWS UNBOUNDED PRECEDING)
                / SUM(n_alarms) OVER (), 1)                                    AS cumulative_pct
FROM t
ORDER BY n_alarms DESC
LIMIT 10;


-- name: q10_alarm_lead_time_before_failure
-- For each progressive failure: did the DCS alarm system warn BEFORE the trip,
-- and how much earlier than the failure? Compare with the true P-F interval.
SELECT
    e.EventID,
    e.asset_id,
    e.FailureModeCode,
    e.DegradationOnset,
    e.EventStart,
    e.pf_interval_days,
    (SELECT MIN(a.AlarmStart)
       FROM fact_alarm a
      WHERE a.asset_id = e.asset_id
        AND a.AlarmStart >= e.DegradationOnset
        AND a.AlarmStart <= e.EventStart)                         AS first_alarm_after_onset,
    ROUND((julianday(e.EventStart) -
           julianday((SELECT MIN(a.AlarmStart)
                        FROM fact_alarm a
                       WHERE a.asset_id = e.asset_id
                         AND a.AlarmStart >= e.DegradationOnset
                         AND a.AlarmStart <= e.EventStart))) * 24, 0) AS alarm_lead_time_h
FROM fact_event e
WHERE e.has_degradation_onset = 1
  AND e.is_unplanned = 1
ORDER BY e.EventStart;


-- name: q11_monthly_pump_vibration
-- Time-series SQL: monthly DE bearing vibration of each pump, ONLY while running
-- (joining the sensor fact table to the run-status fact table)
SELECT
    substr(s.ts, 1, 7)                                                   AS month,
    ROUND(AVG(CASE WHEN s.tag = 'OGP1.P201A_VT5010' THEN s.value END), 2) AS p201a_vib_mm_s,
    ROUND(MAX(CASE WHEN s.tag = 'OGP1.P201A_VT5010' THEN s.value END), 2) AS p201a_vib_max,
    ROUND(AVG(CASE WHEN s.tag = 'OGP1.P201B_VT5010' THEN s.value END), 2) AS p201b_vib_mm_s,
    ROUND(MAX(CASE WHEN s.tag = 'OGP1.P201B_VT5010' THEN s.value END), 2) AS p201b_vib_max
FROM fact_sensor_hourly s
JOIN fact_run_status r
  ON r.ts = s.ts
 AND r.asset_id = CASE WHEN s.tag LIKE '%P201A%' THEN 'P-201A' ELSE 'P-201B' END
WHERE s.tag IN ('OGP1.P201A_VT5010', 'OGP1.P201B_VT5010')
  AND r.status = 'Running'
GROUP BY substr(s.ts, 1, 7)
ORDER BY month;


-- name: q12_data_quality_by_asset
-- Data quality is a KPI too: % valid hours per asset (worst tags first)
SELECT
    asset_id,
    COUNT(*)                  AS n_tags,
    ROUND(AVG(pct_good), 2)   AS avg_pct_good,
    ROUND(MIN(pct_good), 2)   AS worst_tag_pct_good,
    (SELECT t2.short_tag FROM dim_tag t2
      WHERE t2.asset_id = t.asset_id AND t2.pct_good IS NOT NULL
      ORDER BY t2.pct_good LIMIT 1) AS worst_tag
FROM dim_tag t
WHERE pct_good IS NOT NULL
GROUP BY asset_id
ORDER BY avg_pct_good;
