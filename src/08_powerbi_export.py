"""
STEP 8 - EXPORT A STAR SCHEMA FOR POWER BI  +  AN EXCEL KPI WORKBOOK
=====================================================================
Input : everything in data/processed/ and reports/sql_results/
Output: powerbi/data/*.csv                         tables to import in Power BI Desktop
        reports/OGP1_Reliability_KPI_Summary.xlsx  KPI workbook for managers who live in Excel

Star schema (relationships to create in Power BI - Model view):

    dim_date[Date]       1 --- *  fact_asset_daily[Date], fact_health_daily[Date],
                                  fact_events[EventDate], fact_alarms[AlarmDate],
                                  fact_sensor_daily[Date]
    dim_asset[asset_id]  1 --- *  fact_asset_daily, fact_health_daily, fact_events,
                                  fact_alarms, dim_tag
    dim_tag[tag]         1 --- *  fact_sensor_daily[tag]

Run:  python src/08_powerbi_export.py
"""
import pandas as pd
from openpyxl.styles import Font, PatternFill

import config as C
from style import EXCEL_HEADER

print("STEP 8 - Power BI export")
P = C.PROC
STATUS_MAP = {"OGP1.K101_RUN_STATUS": "K-101", "OGP1.GT101_RUN_STATUS": "GT-101",
              "OGP1.P201A_RUN_STATUS": "P-201A", "OGP1.P201B_RUN_STATUS": "P-201B"}
EXTRA_ASSETS = [("P-201 (A/B system)", "P201", "Export pump set (A/B)", "Pump system", "B", False),
                ("PLANT", None, "Whole plant", "Plant", "A", False),
                ("PI SYSTEM", None, "PI historian", "IT / OT system", "-", False)]


def most_common(s):
    """Most frequent non-empty value (None if there is none)."""
    m = s.mode()
    return m.iloc[0] if len(m) else None


# ---- dimensions ---------------------------------------------------------------
assets = pd.DataFrame(C.ASSETS + EXTRA_ASSETS,
                      columns=["asset_id", "tag_prefix", "asset_name", "equipment_class",
                               "criticality", "has_run_status"])
assets["sort_order"] = range(1, len(assets) + 1)

dates = pd.DataFrame({"Date": pd.date_range("2023-01-01", "2025-12-31", freq="D")})
dates["Year"] = dates.Date.dt.year
dates["Quarter"] = "Q" + dates.Date.dt.quarter.astype(str)
dates["MonthNo"] = dates.Date.dt.month
dates["Month"] = dates.Date.dt.strftime("%b")
dates["YearMonth"] = dates.Date.dt.strftime("%Y-%m")
dates["WeekNo"] = dates.Date.dt.isocalendar().week.astype(int)

tags = pd.read_csv(P / "tag_dictionary.csv")

# ---- facts --------------------------------------------------------------------
ev = pd.read_csv(P / "events_clean.csv", parse_dates=["DegradationOnset", "EventStart", "ReturnToService"])
ev["EventDate"] = ev.EventStart.dt.normalize()
al = pd.read_csv(P / "alarms_clean.csv", parse_dates=["AlarmStart", "AlarmEnd"])
al["AlarmDate"] = al.AlarmStart.dt.normalize()

# hours per day in each running state, per machine
status = pd.read_csv(P / "run_status_hourly.csv", parse_dates=["Timestamp"])
st = status.melt(id_vars="Timestamp", value_vars=list(STATUS_MAP), var_name="col", value_name="status")
st["asset_id"] = st.col.map(STATUS_MAP)
st["Date"] = st.Timestamp.dt.normalize()
STATES = {"Running": "running_h", "Stopped": "standby_h", "Maintenance": "maintenance_h",
          "Tripped": "tripped_h", "Unknown": "unknown_h"}
counts = pd.crosstab([st.Date, st.asset_id], st.status)
other_states = [c for c in counts.columns if c not in STATES]      # unexpected states are kept
daily = (counts.reindex(columns=[*STATES, *other_states], fill_value=0)
               .rename(columns=STATES)
               .reset_index())
daily.columns.name = None
daily["calendar_h"] = daily[list(STATES.values())].sum(axis=1)

# daily mean / min / max of every sensor, in long format (Date, tag, ...)
sensors = pd.read_csv(P / "sensor_hourly_clean.csv", parse_dates=["Timestamp"], index_col="Timestamp")
sd = (sensors.resample("D").agg(["mean", "min", "max"])
             .stack(level=0, future_stack=True)
             .reset_index())
sd.columns = ["Date", "tag", "avg_value", "min_value", "max_value"]
sd = sd.dropna(subset=["avg_value"]).round({"avg_value": 3, "min_value": 3, "max_value": 3})

# daily machine health from the step-6 predictions
pr = pd.read_csv(P / "predictions_hourly.csv", parse_dates=["Timestamp"])
pr["Date"] = pr.Timestamp.dt.normalize()
pr["asset_id"] = pr.unit.replace({"K-101/GT-101": "K-101"})
hd = (pr.groupby(["Date", "unit", "asset_id", "dataset"])
        .agg(max_risk_ratio=("risk_ratio", "max"), max_health_index=("health_index", "max"),
             max_ml_risk=("ml_risk", "max"), alert_hours=("recommended_alert", "sum"),
             degrading_hours=("actual_degrading", "sum"), running_hours=("risk_ratio", "size"),
             main_driver=("hi_driver", most_common))
        .reset_index())
hd["risk_band"] = pd.cut(hd.max_risk_ratio, [-1, 0.7, 1.0, 1e9], labels=["Normal", "Watch", "Alert"])

# ---- write the Power BI tables ------------------------------------------------
read = lambda name: pd.read_csv(P / f"{name}.csv")
out = {
    "dim_date": dates, "dim_asset": assets, "dim_tag": tags,
    "fact_events": ev, "fact_alarms": al, "fact_asset_daily": daily,
    "fact_sensor_daily": sd, "fact_health_daily": hd,
    "kpi_reliability": read("reliability_kpis"),
    "kpi_weibull": read("weibull_results"),
    "kpi_model_metrics": read("model_metrics"),
    "kpi_model_event_detection": read("model_event_detection"),
    "kpi_feature_importance": read("feature_importance"),
    "kpi_data_quality": read("data_quality_summary"),
    "kpi_warning_time": read("warning_time_analysis"),
    "maint_watch_list": read("maint_watch_list"),
    "maint_strategy": read("maint_strategy_by_failure_mode"),
    "maint_alert_log_2025": read("maint_alert_log_2025"),
    "maint_alarm_actions": read("maint_alarm_actions"),
}
for name, df in out.items():
    df.to_csv(C.PBI / f"{name}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"  {name:28s} {len(df):>7,} rows")

# ---- Excel KPI workbook (formatted) ---------------------------------------------
sql_results = C.REPORTS / "sql_results"
sheets = {
    "Reliability KPIs": out["kpi_reliability"],
    "Weibull": out["kpi_weibull"],
    "Failure Pareto": pd.read_csv(sql_results / "q04_failure_mode_pareto.csv"),
    "Alarm KPIs": read("alarm_kpis").rename(columns={"Unnamed: 0": "kpi"}),
    "Bad actor alarms": pd.read_csv(sql_results / "q09_bad_actor_alarms.csv"),
    "Model performance": out["kpi_model_metrics"],
    "Failure detection 2025": out["kpi_model_event_detection"],
    "Maintenance strategy": out["maint_strategy"],
    "Data quality": out["kpi_data_quality"],
    "Event log (clean)": ev.drop(columns="EventDate"),
}
xlsx = C.REPORTS / "OGP1_Reliability_KPI_Summary.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name, index=False)
        ws = xw.sheets[name]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 45)
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor=EXCEL_HEADER)
print("  Excel workbook:", xlsx)