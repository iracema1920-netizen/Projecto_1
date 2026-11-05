"""
STEP 2 - LOAD A SQL DATABASE AND RUN THE KPI QUERIES
=====================================================
Input : data/processed/*.csv (from step 1), sql/01_create_schema.sql, sql/02_kpi_queries.sql
Output: data/processed/ogp1_reliability.db   (SQLite database - open it in DB Browser / DBeaver)
        reports/sql_results/q*.csv            (one CSV per KPI query)

Why SQLite? It is a real SQL database stored in one file, needs no server and
is built into Python. Everything here works the same in SQL Server or
PostgreSQL (see the notes at the top of 01_create_schema.sql).

Run:  python src/02_load_sql.py
"""
import re
import sqlite3

import pandas as pd

import config as C

print("STEP 2 - Load SQL database and run KPI queries")

STATUS_MAP = {"OGP1.K101_RUN_STATUS": "K-101", "OGP1.GT101_RUN_STATUS": "GT-101",
              "OGP1.P201A_RUN_STATUS": "P-201A", "OGP1.P201B_RUN_STATUS": "P-201B"}
EXTRA_ASSETS = [("P-201 (A/B system)", "P201", "Export pump set (A/B)", "Pump system", "B", False),
                ("PLANT", None, "Whole plant", "Plant", "A", False),
                ("PI SYSTEM", None, "PI historian", "IT / OT system", "-", False)]
ASSET_COLS = ["asset_id", "tag_prefix", "asset_name", "equipment_class", "criticality", "has_run_status"]


def as_text(ts):
    """SQLite has no date type: store timestamps as 'YYYY-MM-DD HH:MM:SS' text."""
    return pd.to_datetime(ts).dt.strftime("%Y-%m-%d %H:%M:%S")


def schema_columns(con, table):
    """Column names of a table as defined in 01_create_schema.sql."""
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


# A fresh database every run, so re-running never duplicates rows
C.DB_PATH.unlink(missing_ok=True)
con = sqlite3.connect(C.DB_PATH)
con.executescript((C.SQL_DIR / "01_create_schema.sql").read_text(encoding="utf-8"))

# --- dimension tables ---------------------------------------------------------
dim_asset = pd.DataFrame(C.ASSETS + EXTRA_ASSETS, columns=ASSET_COLS)
dim_asset.to_sql("dim_asset", con, if_exists="append", index=False)
pd.read_csv(C.PROC / "tag_dictionary.csv").to_sql("dim_tag", con, if_exists="append", index=False)

# --- fact tables -------------------------------------------------------------
# Sensor data: wide (1 column per tag) -> long (ts, tag, value)
wide = pd.read_csv(C.PROC / "sensor_hourly_clean.csv", parse_dates=["Timestamp"])
long = wide.melt(id_vars="Timestamp", var_name="tag", value_name="value").dropna()
long["ts"] = as_text(long.Timestamp)
long[["ts", "tag", "value"]].to_sql("fact_sensor_hourly", con, if_exists="append",
                                    index=False, chunksize=50_000)

status = pd.read_csv(C.PROC / "run_status_hourly.csv", parse_dates=["Timestamp"])
rs = status.melt(id_vars="Timestamp", value_vars=list(STATUS_MAP), var_name="col", value_name="status")
rs["asset_id"] = rs.col.map(STATUS_MAP)
rs["ts"] = as_text(rs.Timestamp)
rs[["ts", "asset_id", "status"]].to_sql("fact_run_status", con, if_exists="append", index=False)

ev = pd.read_csv(C.PROC / "events_clean.csv")
for c in ["DegradationOnset", "EventStart", "ReturnToService"]:
    ev[c] = as_text(ev[c])
ev[schema_columns(con, "fact_event")].to_sql("fact_event", con, if_exists="append", index=False)

al = pd.read_csv(C.PROC / "alarms_clean.csv")
for c in ["AlarmStart", "AlarmEnd"]:
    al[c] = as_text(al[c])
al[schema_columns(con, "fact_alarm")].to_sql("fact_alarm", con, if_exists="append", index=False)
con.commit()

# --- run every KPI query (blocks start with '-- name: qXX_...') ---------------
out_dir = C.REPORTS / "sql_results"
out_dir.mkdir(parents=True, exist_ok=True)
sql_text = (C.SQL_DIR / "02_kpi_queries.sql").read_text(encoding="utf-8")
blocks = re.split(r"^-- name:\s*", sql_text, flags=re.M)[1:]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
for block in blocks:
    name, query = block.split("\n", 1)
    name = name.strip()
    result = pd.read_sql_query(query, con)
    result.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"\n--- {name} ({len(result)} rows) ---")
    print(result.head(12).to_string(index=False))

con.close()
print(f"\n  database: {C.DB_PATH}\n  query results: {out_dir}")