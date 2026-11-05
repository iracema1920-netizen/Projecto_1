"""
STEP 4 - ALARM MANAGEMENT ANALYSIS (ISA-18.2 / EEMUA 191)
==========================================================
Input : data/processed/alarms_clean.csv, events_clean.csv
Output: reports/figures/fig06_alarm_bad_actors.png, fig07_alarm_trend.png
        data/processed/alarm_kpis.csv
        data/processed/alarms_clean.csv  (updated with linked_to_event / is_repeat_24h)

Questions answered:
  1. Is the alarm load acceptable for an operator?      (alarms/day, peaks)
  2. Which tags are "bad actors"?                        (Pareto)
  3. How many alarms are standing (stale) for > 24 h?   (EEMUA 191 metric)
  4. Is the priority distribution healthy?               (target ~80/15/5 low/med/high)
  5. Do alarms come with an event, or are they "nuisance"?

Run:  python src/04_alarm_analysis.py
"""
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from style import apply_style, save, PALETTE, BLUE, ORANGE, MUTED

apply_style()
print("STEP 4 - Alarm analysis")

al = pd.read_csv(C.PROC / "alarms_clean.csv", parse_dates=["AlarmStart", "AlarmEnd"])
ev = pd.read_csv(C.PROC / "events_clean.csv",
                 parse_dates=["DegradationOnset", "EventStart", "ReturnToService"])
n_days = (pd.Timestamp("2025-12-31") - pd.Timestamp("2023-01-01")).days + 1


# ---------------------------------------------------------------------------
# 4.1 Is each alarm linked to a logged event on the same asset?
#     Window = [degradation onset (or event start - 2 days), return to service + 1 day]
# ---------------------------------------------------------------------------
ev = ev.assign(asset_id=ev.asset_id.replace({"P-201 (A/B system)": "P-201A"}),
               win_start=ev.DegradationOnset.fillna(ev.EventStart - pd.Timedelta("2D")),
               win_end=ev.ReturnToService + pd.Timedelta("1D"))


def related_assets(asset):
    """Assets whose events can explain an alarm on `asset`."""
    if asset.startswith("P-201"):          # pump suction alarms <- low tank level (TK-501)
        return [asset, "TK-501"]
    if asset == "FLARE":                   # flaring <- compressor train trip or plant shutdown
        return ["FLARE", "K-101", "GT-101", "PLANT"]
    return [asset]


def is_linked(a):
    e = ev[ev.asset_id.isin(related_assets(a.asset_id))]
    return ((a.AlarmStart >= e.win_start) & (a.AlarmStart <= e.win_end)).any()


al["linked_to_event"] = al.apply(is_linked, axis=1).astype(int)


# ---------------------------------------------------------------------------
# 4.2 Repeating alarms: same tag + alarm type returns within 24 h of clearing.
#     Typical cause: no deadband / on-delay, so a slowly degrading value crosses
#     the limit again and again. ISA-18.2 calls this a nuisance alarm.
# ---------------------------------------------------------------------------
al = al.sort_values("AlarmStart")
prev_clear = al.groupby(["Tag", "AlarmType"]).AlarmEnd.shift()
al["hours_since_last_clear"] = (al.AlarmStart - prev_clear).dt.total_seconds() / 3600
al["is_repeat_24h"] = (al.hours_since_last_clear <= 24).astype(int)


# ---------------------------------------------------------------------------
# 4.3 KPIs
# ---------------------------------------------------------------------------
kpis = {
    "total_alarms": len(al),
    "avg_alarms_per_day": round(len(al) / n_days, 2),
    "max_alarms_per_day": int(al.AlarmStart.dt.date.value_counts().max()),
    "standing_alarms_24h_plus": int(al.is_standing.sum()),
    "pct_standing": round(100 * al.is_standing.mean(), 1),
    "pct_high_priority": round(100 * al.Priority.eq("High").mean(), 1),
    "pct_top10_tags": round(100 * al.Tag.value_counts().head(10).sum() / len(al), 1),
    "pct_linked_to_event": round(100 * al.linked_to_event.mean(), 1),
    "pct_repeat_within_24h": round(100 * al.is_repeat_24h.mean(), 1),
}
pd.Series(kpis).to_csv(C.PROC / "alarm_kpis.csv", header=["value"])
for name, value in kpis.items():
    print(f"  {name:28s} {value}")

by_tag = (al.groupby(["Tag", "Descriptor"])
            .agg(n=("AlarmID", "count"), standing=("is_standing", "sum"),
                 repeats=("is_repeat_24h", "sum"), linked=("linked_to_event", "mean"),
                 avg_dur=("Duration_h", "mean"))
            .sort_values("n", ascending=False)
            .reset_index())
print(by_tag.head(10).round(2).to_string(index=False))


# ---------------------------------------------------------------------------
# 4.4 Figure: bad actors (stacked: new alarm vs. repeat within 24 h)
# ---------------------------------------------------------------------------
top = by_tag.head(12).iloc[::-1]
labels = top.Tag.str.replace("OGP1.", "", regex=False) + " - " + top.Descriptor.str.slice(0, 30)
new_alarms = top.n - top.repeats

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(labels, new_alarms, color=BLUE, height=0.65, label="new alarm")
ax.barh(labels, top.repeats, left=new_alarms, color=ORANGE, height=0.65,
        edgecolor="white", linewidth=1.5,
        label="repeat within 24 h of clearing (nuisance -> add deadband / delay)")
ax.set_xlabel("number of alarms 2023-2025")
ax.set_title(f"Top-10 tags = {kpis['pct_top10_tags']} % of alarms; "
             f"{kpis['pct_repeat_within_24h']} % of all alarms are repeats")
ax.legend(loc="lower right")
save(fig, C.FIG / "fig06_alarm_bad_actors.png")


# ---------------------------------------------------------------------------
# 4.5 Figure: monthly alarm count by asset
# ---------------------------------------------------------------------------
monthly = (al.groupby([al.AlarmStart.dt.to_period("M"), "asset_id"]).size()
             .unstack(fill_value=0))
monthly = monthly[monthly.sum().sort_values(ascending=False).index]   # biggest asset at the bottom
colors = PALETTE[:monthly.shape[1]]
if monthly.shape[1] > len(PALETTE):                  # never invent extra colours: group the rest
    keep = monthly.columns[:len(PALETTE) - 1]
    monthly = monthly[keep].assign(Other=monthly.drop(columns=keep).sum(axis=1))
    colors = PALETTE[:len(keep)] + [MUTED]

fig, ax = plt.subplots(figsize=(12, 4))
monthly.plot(kind="bar", stacked=True, ax=ax, width=0.85, edgecolor="white", linewidth=0.6,
             color=colors)
ax.set_xticklabels(monthly.index.strftime("%b-%y"), rotation=90, fontsize=7)
ax.set(xlabel="", ylabel="alarms / month", title="Monthly alarm count by asset")
ax.grid(axis="x", visible=False)
ax.legend(title="asset", loc="upper left", bbox_to_anchor=(1.01, 1))
save(fig, C.FIG / "fig07_alarm_trend.png")

# save the alarm list with the two new columns (used by steps 6 and 7)
al.sort_values("AlarmID").to_csv(C.PROC / "alarms_clean.csv", index=False)
print("  figures saved")