"""
STEP 7 - MAINTENANCE ACTION LIST (what the supervisor does on Monday morning)
==============================================================================
Input : data/processed/events_clean.csv, predictions_hourly.csv, alarms_clean.csv
Output: data/processed/maint_strategy_by_failure_mode.csv  RCM-style strategy per failure mode
        data/processed/maint_watch_list.csv                current risk per machine + action
        data/processed/maint_alert_log_2025.csv            every 2025 alert episode, true or false
        data/processed/maint_alarm_actions.csv             bad-actor alarms + recommended fix
        reports/Maintenance_Action_List.xlsx               the same, formatted for the field
        reports/figures/fig15_inspection_intervals.png

Analytics only helps if it ends in a work order. This step turns the results of
steps 3-6 into decisions a maintenance team can act on:

  1. STRATEGY per failure mode (RCM logic):
       - progressive failure with a P-F interval  -> condition-based maintenance,
         inspection interval = P-F interval / 2   (classic RCM rule: you get at
         least one inspection inside the warning window)
       - sudden / spurious failure                -> root-cause analysis / redesign
  2. WATCH LIST: latest risk of each machine, the sensor that drives it and the
     check to perform.
  3. ALERT LOG 2025: every model alert with the outcome (real degradation or false
     alarm), for feedback with the team.
  4. ALARM ACTIONS: what to change in the DCS for each bad-actor tag.

Run:  python src/07_maintenance_action_list.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill

import config as C
from style import apply_style, save, BLUE, LIGHT, OK, WARNING, CRITICAL, EXCEL_HEADER

apply_style()
print("STEP 7 - Maintenance action list")

ev = pd.read_csv(C.PROC / "events_clean.csv", parse_dates=["DegradationOnset", "EventStart"])
pr = pd.read_csv(C.PROC / "predictions_hourly.csv", parse_dates=["Timestamp"])
al = pd.read_csv(C.PROC / "alarms_clean.csv")        # includes is_repeat_24h from step 4


# ---------------------------------------------------------------------------
# What to check when a given sensor drives the alert (maintenance know-how)
# ---------------------------------------------------------------------------
PUMP_PERFORMANCE = "Pump performance check (head vs flow): impeller wear / internal recirculation"
SENSOR_ACTION = {
    "VT":       "Vibration analysis (spectrum + phase); check alignment, bearing condition, looseness",
    "TT5012":   "Check bearing lubrication (oil level / grease), bearing temperature trend; plan bearing change",
    "TT1013":   "Check thrust bearing pads and axial position; review process thrust load",
    "TT1014":   "Check journal bearing oil supply and temperature; oil sample",
    "TT5018":   "Motor winding insulation test (IR / PI), check cooling fan and motor load",
    "LT5016":   "Check seal pot level and leakage; plan mechanical seal replacement",
    "PT5015":   "Check seal support system pressure (barrier fluid), leaks",
    "PDT2007":  "Plan inlet filter replacement at next opportunity; inspect filter house",
    "PDT1015":  "Switch to standby lube oil filter and replace element; oil sample",
    "FT1020":   "Dry gas seal: check primary vent flow trend, plan seal inspection / replacement",
    "ZT1012":   "Check axial displacement probe and thrust bearing; review surge history",
    "EFF_POLY": "Compressor fouling: plan online/offline wash; check suction strainer",
    "TT2008":   "Check GT lube oil cooler (fouling, fan), oil temperature control valve",
    "PT5002":   PUMP_PERFORMANCE,
    "FT5003":   PUMP_PERFORMANCE,
    "IT5004":   "Check motor current vs flow: mechanical drag or electrical issue",
    "TT2003":   "Check exhaust thermocouples and combustion (spread), fuel nozzles",
    "TT2004":   "Combustion inspection: exhaust temperature spread -> fuel nozzle / combustor",
}


def action_for(sensor):
    """Recommended field check for the sensor that drives an alert."""
    if not isinstance(sensor, str):
        return "No specific action"
    for key, action in SENSOR_ACTION.items():
        if key in sensor:
            return action
    return "Check sensor trend against operating conditions; verify instrument"


def most_common(s):
    """Most frequent value of a column (NaN if the column is empty)."""
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan


# ---------------------------------------------------------------------------
# 7.1 Strategy per failure mode (RCM-style)
# ---------------------------------------------------------------------------
fm = (ev[ev.is_unplanned == 1]
        .groupby(["asset_id", "FailureModeCode", "FailureMode", "failure_type"])
        .agg(events=("EventID", "count"), downtime_h=("Downtime_h", "sum"),
             min_pf_days=("pf_interval_days", "min"), avg_pf_days=("pf_interval_days", "mean"),
             detected_by=("DetectedBy", most_common))
        .reset_index())

progressive = fm.failure_type.eq("Progressive degradation")
long_pf = progressive & fm.min_pf_days.ge(2)
fm["recommended_strategy"] = np.select(
    [long_pf, progressive],
    ["Condition-based maintenance (CBM)", "CBM with continuous online monitoring (short P-F)"],
    default="Root-cause analysis / redesign (no warning time)")

# RCM rule: inspect at least twice inside the shortest P-F interval seen
fm["inspection_interval_days"] = np.where(long_pf, np.floor(fm.min_pf_days / 2).clip(lower=1), np.nan)
fm["monitoring_task"] = np.where(fm.recommended_strategy.str.startswith("Root"),
                                 "RCA team: review trip logic, control tuning, operating envelope",
                                 fm.detected_by.apply(action_for))
fm["priority"] = pd.cut(fm.downtime_h, [-1, 20, 100, np.inf], labels=["Low", "Medium", "High"])
fm = fm.sort_values("downtime_h", ascending=False)


# ---------------------------------------------------------------------------
# 7.2 Watch list = latest risk per machine (end of the data: 31-Dec-2025)
# ---------------------------------------------------------------------------
last = pr.sort_values("Timestamp").groupby("unit").tail(1).set_index("unit")
trend = (pr[pr.Timestamp >= pr.Timestamp.max() - pd.Timedelta("7D")]
           .groupby("unit").risk_ratio.agg(max_risk_7d="max", avg_risk_7d="mean"))

watch = last[["Timestamp", "risk_ratio", "health_index", "hi_driver"]].join(trend)
watch["status"] = pd.cut(watch.risk_ratio, [-np.inf, 0.7, 1.0, np.inf], labels=["Normal", "Watch", "Alert"])
watch["recommended_action"] = np.where(watch.status.eq("Normal"), "Routine rounds only",
                                       watch.hi_driver.apply(action_for))
watch = watch.reset_index().rename(columns={"Timestamp": "as_of", "hi_driver": "main_driver_sensor"})


# ---------------------------------------------------------------------------
# 7.3 Alert log 2025: each sustained alert episode and its outcome
# ---------------------------------------------------------------------------
te = pr[pr.dataset == "test"].sort_values(["unit", "Timestamp"]).copy()
on = te.recommended_alert.eq(1)
te["episode"] = (on & ~on.groupby(te.unit).shift(1, fill_value=False)).groupby(te.unit).cumsum()

ep = (te[on].groupby(["unit", "episode"])
        .agg(alert_start=("Timestamp", "min"), alert_end=("Timestamp", "max"),
             peak_risk=("risk_ratio", "max"), driver=("hi_driver", most_common),
             true_degradation_share=("actual_degrading", "mean"))
        .reset_index()
        .drop(columns="episode"))
ep["duration_h"] = ((ep.alert_end - ep.alert_start).dt.total_seconds() / 3600 + 1).astype(int)
ep["outcome"] = np.where(ep.true_degradation_share > 0.5, "True alert (degradation confirmed)",
                         "False alert (no failure followed)")
ep["recommended_action"] = ep.driver.apply(action_for)


# ---------------------------------------------------------------------------
# 7.4 Alarm actions for the bad actors
# ---------------------------------------------------------------------------
ba = (al.groupby(["Tag", "asset_id", "Descriptor"])
        .agg(alarms=("AlarmID", "count"), repeats_24h=("is_repeat_24h", "sum"),
             standing=("is_standing", "sum"), avg_duration_h=("Duration_h", "mean"))
        .sort_values("alarms", ascending=False)
        .head(10)
        .reset_index())


def alarm_fix(r):
    fixes = []
    if r.repeats_24h / r.alarms > 0.3:
        fixes.append("add deadband / on-delay (repeating)")
    if r.standing > 0:
        fixes.append(f"review {int(r.standing)} standing alarms (shelve or re-set limit)")
    if "FLR" in r.Tag:
        fixes.append("state-based suppression during compressor trip (consequential alarm)")
    if "EFF_POLY" in r.Tag:
        fixes.append("convert to a performance KPI / advisory, not an operator alarm")
    return "; ".join(fixes) if fixes else "keep - alarm is effective"


ba["recommended_fix"] = ba.apply(alarm_fix, axis=1)
ba["avg_duration_h"] = ba.avg_duration_h.round(1)


# ---------------------------------------------------------------------------
# Save CSVs + field-ready Excel
# ---------------------------------------------------------------------------
sheets = {"Watch list": watch, "Strategy by failure mode": fm,
          "Alert log 2025": ep, "Alarm actions": ba}
for df, name in zip(sheets.values(), ("maint_watch_list", "maint_strategy_by_failure_mode",
                                      "maint_alert_log_2025", "maint_alarm_actions")):
    df.to_csv(C.PROC / f"{name}.csv", index=False)

# soft fills for the status words, so the sheet can be read at a glance
STATUS_FILL = {"Normal": OK, "Watch": WARNING, "Alert": CRITICAL, "High": CRITICAL,
               "Medium": WARNING, "Low": OK}


def soft(hex_color, alpha=0.25):
    """Blend a colour with white (Excel has no transparency)."""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "".join(f"{round(255 - alpha * (255 - c)):02X}" for c in rgb)


xlsx = C.REPORTS / "Maintenance_Action_List.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name, index=False)
        ws = xw.sheets[name]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 70)
            for c in col[1:]:
                if c.value in STATUS_FILL and col[0].value in ("status", "priority"):
                    c.fill = PatternFill("solid", fgColor=soft(STATUS_FILL[c.value]))
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor=EXCEL_HEADER)

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
print("\n  WATCH LIST (as of end of data)")
print(watch[["unit", "status", "risk_ratio", "main_driver_sensor", "recommended_action"]].to_string(index=False))
print("\n  STRATEGY BY FAILURE MODE (top 10 by downtime)")
print(fm[["asset_id", "FailureModeCode", "downtime_h", "min_pf_days", "recommended_strategy",
          "inspection_interval_days"]].head(10).to_string(index=False))
n_true = ep.outcome.str.startswith("True").sum()
print(f"\n  2025 alert episodes: {len(ep)} ({n_true} true, {len(ep) - n_true} false)")


# ---------------------------------------------------------------------------
# fig15 - P-F interval and the recommended inspection interval per failure mode
# ---------------------------------------------------------------------------
cbm = fm[fm.recommended_strategy.str.startswith("Condition")].sort_values("min_pf_days")
labels = cbm.asset_id + "  " + cbm.FailureMode.str.slice(0, 32)
y = np.arange(len(cbm))

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.barh(y, cbm.min_pf_days, color=LIGHT, height=0.6, label="shortest P-F interval seen (days)")
ax.scatter(cbm.inspection_interval_days, y, color=BLUE, zorder=3, s=55, edgecolor="white",
           linewidth=1, label="recommended inspection interval = P-F / 2")
for yy, v in zip(y, cbm.inspection_interval_days):
    ax.text(v + 1.5, yy, f"every {v:.0f} d", va="center", fontsize=8, color=BLUE)
ax.set_yticks(y, labels, fontsize=8)
ax.set_xlabel("days")
ax.set_title("Condition-based maintenance: inspection intervals derived from the P-F interval")
ax.legend(loc="lower right")
save(fig, C.FIG / "fig15_inspection_intervals.png")
print("  saved:", xlsx)