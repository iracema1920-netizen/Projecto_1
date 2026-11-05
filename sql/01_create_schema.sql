/* ============================================================================
   OGP1 Reliability Database - STAR SCHEMA
   ----------------------------------------------------------------------------
   Written for SQLite (free, no server needed). The same DDL runs on
   SQL Server / PostgreSQL with small type changes:
       TEXT      -> VARCHAR(n)        REAL -> FLOAT / DOUBLE PRECISION
       TIMESTAMP -> DATETIME2 (SQL Server) / TIMESTAMP (PostgreSQL)

   Dimension tables  : dim_asset, dim_tag
   Fact tables       : fact_sensor_hourly (long format), fact_run_status,
                       fact_event, fact_alarm
   ========================================================================== */

DROP TABLE IF EXISTS fact_sensor_hourly;
DROP TABLE IF EXISTS fact_run_status;
DROP TABLE IF EXISTS fact_alarm;
DROP TABLE IF EXISTS fact_event;
DROP TABLE IF EXISTS dim_tag;
DROP TABLE IF EXISTS dim_asset;

-- Equipment master data (one row per functional location)
CREATE TABLE dim_asset (
    asset_id         TEXT PRIMARY KEY,      -- e.g. 'K-101'
    tag_prefix       TEXT,                  -- e.g. 'K101'
    asset_name       TEXT,
    equipment_class  TEXT,
    criticality      TEXT,                  -- A = high, B = medium, C = low
    has_run_status   INTEGER                -- 1 if operating hours can be measured
);

-- Instrument / tag dictionary (one row per PI tag)
CREATE TABLE dim_tag (
    tag              TEXT PRIMARY KEY,      -- e.g. 'OGP1.K101_VT1010A'
    short_tag        TEXT,
    asset_id         TEXT REFERENCES dim_asset(asset_id),
    instrument_type  TEXT,                  -- ISA code: PT, TT, VT ...
    description_en   TEXT,
    unit             TEXT,
    limit_LL         REAL,
    limit_L          REAL,
    limit_H          REAL,
    limit_HH         REAL,
    pct_good         REAL                   -- % of hours with valid data
);

-- Hourly process values in LONG format (timestamp, tag, value).
-- Long format is how historians and SQL databases store time series:
-- adding a new tag means adding rows, not changing the table structure.
CREATE TABLE fact_sensor_hourly (
    ts     TIMESTAMP NOT NULL,
    tag    TEXT      NOT NULL REFERENCES dim_tag(tag),
    value  REAL,
    PRIMARY KEY (ts, tag)
);

-- Running status per rotating asset per hour
CREATE TABLE fact_run_status (
    ts        TIMESTAMP NOT NULL,
    asset_id  TEXT      NOT NULL REFERENCES dim_asset(asset_id),
    status    TEXT,                         -- Running / Stopped / Tripped / Maintenance / Unknown
    PRIMARY KEY (ts, asset_id)
);

-- Maintenance / failure / operational event log (CMMS-style)
CREATE TABLE fact_event (
    EventID                TEXT PRIMARY KEY,
    asset_id               TEXT,
    Asset                  TEXT,            -- original free-text value
    EventType              TEXT,
    FailureModeCode        TEXT,
    FailureMode            TEXT,
    Description            TEXT,
    DegradationOnset       TIMESTAMP,
    EventStart             TIMESTAMP,
    ReturnToService        TIMESTAMP,
    Downtime_h             REAL,
    DetectedBy             TEXT,
    RootCause              TEXT,
    Severity               TEXT,
    WorkOrder              TEXT,
    is_failure             INTEGER,
    is_unplanned           INTEGER,
    has_degradation_onset  INTEGER,
    pf_interval_days       REAL,
    failure_type           TEXT,
    year                   INTEGER,
    month                  TEXT
);

-- DCS alarm journal
CREATE TABLE fact_alarm (
    AlarmID         TEXT PRIMARY KEY,
    Tag             TEXT,
    asset_id        TEXT,
    Asset           TEXT,
    Descriptor      TEXT,
    AlarmType       TEXT,                   -- L, LL, H, HH
    Priority        TEXT,
    "Limit"         REAL,
    Units           TEXT,
    AlarmStart      TIMESTAMP,
    AlarmEnd        TIMESTAMP,
    Duration_h      REAL,
    PeakValue       REAL,
    is_standing     INTEGER,
    exceedance_pct  REAL,
    date            TEXT
);

-- Indexes make the KPI queries fast
CREATE INDEX ix_sensor_tag   ON fact_sensor_hourly(tag, ts);
CREATE INDEX ix_status_asset ON fact_run_status(asset_id, status);
CREATE INDEX ix_event_asset  ON fact_event(asset_id, EventStart);
CREATE INDEX ix_alarm_tag    ON fact_alarm(Tag, AlarmStart);
