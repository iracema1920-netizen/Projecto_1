"""
STEP 3 - RELIABILITY ANALYSIS (the "R" in RAM)
===============================================
Input : data/processed/events_clean.csv, run_status_hourly.csv, alarms_clean.csv
Output: reports/figures/fig01..fig05 PNG charts
        data/processed/reliability_kpis.csv, weibull_results.csv, warning_time_analysis.csv

Questions answered:
  1. How reliable / available is each critical machine?           (MTBF, MTTR, availability)
  2. Which failure modes cost the most downtime?                    (Pareto - where to focus)
  3. Is the failure pattern infant-mortality, random or wear-out?   (Weibull)
  4. How much warning did we get before each failure?               (P-F interval vs alarm lead time)

Run:  python src/03_reliability_analysis.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

import config as C
from style import (apply_style, save, BLUE, ORANGE, AQUA, VIOLET, MUTED, LIGHT,
                   WARNING, CRITICAL)

apply_style()
print("STEP 3 - Reliability analysis")

ev = pd.read_csv(C.PROC / "events_clean.csv",
                 parse_dates=["DegradationOnset", "EventStart", "ReturnToService"])
status = pd.read_csv(C.PROC / "run_status_hourly.csv", parse_dates=["Timestamp"],
                     index_col="Timestamp")
al = pd.read_csv(C.PROC / "alarms_clean.csv", parse_dates=["AlarmStart"])

STATUS_COL = {"K-101": "OGP1.K101_RUN_STATUS", "GT-101": "OGP1.GT101_RUN_STATUS",
              "P-201A": "OGP1.P201A_RUN_STATUS", "P-201B": "OGP1.P201B_RUN_STATUS"}
failures = ev[ev.is_failure == 1]


# ---------------------------------------------------------------------------
# 3.1 KPI table (same logic as SQL query q03)
# ---------------------------------------------------------------------------
rows = []
for asset, col in STATUS_COL.items():
    s = status[col]
    f = failures[failures.asset_id == asset]
    running_h = int(s.eq("Running").sum())
    mtbf_h = running_h / len(f) if len(f) else np.nan       # no failures -> MTBF unknown
    rows.append({
        "asset_id": asset,
        "running_h": running_h,
        "n_failures": len(f),
        "mtbf_h": round(mtbf_h) if len(f) else np.nan,
        "mtbf_days": round(mtbf_h / 24, 1),
        "mttr_h": round(f.Downtime_h.mean(), 1),
        "availability_pct": round(100 * (1 - s.isin(["Maintenance", "Tripped"]).mean()), 2),
        "failure_downtime_h": f.Downtime_h.sum(),
    })
kpi = pd.DataFrame(rows)
kpi.to_csv(C.PROC / "reliability_kpis.csv", index=False)
print(kpi.to_string(index=False))

# Redundant pumps (1-out-of-2): service is lost only if NEITHER pump is available
available = status[[STATUS_COL["P-201A"], STATUS_COL["P-201B"]]].isin(["Running", "Stopped"])
system_avail = 100 * available.any(axis=1).mean()
print(f"  P-201A/B system availability (1-out-of-2): {system_avail:.2f} %")

# Figure 1 - MTBF and availability
k = kpi.sort_values("mtbf_days")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].barh(k.asset_id, k.mtbf_days, color=BLUE, height=0.6)
ax[0].bar_label(ax[0].containers[0], fmt="%.0f d", padding=4)
ax[0].set(title="MTBF (days of running time)", xlabel="days")
ax[0].margins(x=0.12)                                       # room for the labels

bar_colors = [BLUE if v >= 97 else CRITICAL for v in k.availability_pct]   # red = below target
ax[1].barh(k.asset_id, k.availability_pct, color=bar_colors, height=0.6)
ax[1].bar_label(ax[1].containers[0], fmt="%.1f %%", padding=4)
ax[1].axvline(97, ls="--", color=MUTED, lw=1)
ax[1].text(97.1, 0.01, "target 97 %", color=MUTED, fontsize=8,
           transform=ax[1].get_xaxis_transform())                # y in axes fraction
ax[1].set(title="Time-based availability (red = below target)", xlim=(90, 100.8))

fig.suptitle("OGP1 rotating equipment - reliability KPIs 2023-2025", x=0.02, ha="left")
fig.tight_layout()
save(fig, C.FIG / "fig01_mtbf_availability.png")


# ---------------------------------------------------------------------------
# 3.2 Pareto of unplanned downtime by failure mode
#     Bars = downtime; the "vital few" (first 80 % of downtime) are highlighted.
# ---------------------------------------------------------------------------
unpl = ev[ev.is_unplanned == 1].copy()
unpl["label"] = unpl.asset_id + "  " + unpl.FailureMode.str.slice(0, 32)
par = unpl.groupby("label").Downtime_h.sum().sort_values(ascending=False)
par = par[par > 0].head(15)
cum_pct = 100 * par.cumsum() / unpl.Downtime_h.sum()
vital_few = cum_pct.shift(fill_value=0) < 80             # bars that make up the first 80 %

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(par))
ax.bar(x, par.values, color=np.where(vital_few, BLUE, LIGHT), width=0.7)
for xi, (h, c) in enumerate(zip(par.values, cum_pct.values)):
    ax.text(xi, h, f"{c:.0f} %", ha="center", va="bottom", fontsize=7.5, color=MUTED)
ax.set_xticks(x, par.index, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("Unplanned downtime (h)")
ax.set_title(f"Pareto: {vital_few.sum()} failure modes cause 80 % of unplanned downtime "
             "(label = cumulative %)")
save(fig, C.FIG / "fig02_downtime_pareto.png")


# ---------------------------------------------------------------------------
# 3.3 Event timeline (one lane per asset) - the "story" of three years
# ---------------------------------------------------------------------------
lanes = ["K-101", "GT-101", "P-201A", "P-201B", "E-401", "V-301", "TK-501", "PLANT"]
lane_y = {a: i for i, a in enumerate(lanes)}
event_colors = {"Failure / Trip": CRITICAL, "Corrective Maintenance": ORANGE,
                "Planned Maintenance": AQUA, "Process Upset": VIOLET,
                "Instrument Fault": MUTED}
on_lanes = ev[ev.asset_id.isin(lanes)]

fig, ax = plt.subplots(figsize=(12, 4.5))
for r in on_lanes[on_lanes.has_degradation_onset == 1].itertuples():   # degradation bars
    y = lane_y[r.asset_id]
    ax.plot([r.DegradationOnset, r.EventStart], [y, y], color=WARNING, alpha=0.45, lw=5,
            solid_capstyle="round")
for etype, colr in event_colors.items():
    d = on_lanes[on_lanes.EventType == etype]
    ax.scatter(d.EventStart, d.asset_id.map(lane_y), s=20 + d.Downtime_h * 1.5, color=colr,
               alpha=0.85, edgecolor="white", linewidth=0.8, label=etype, zorder=3)
ax.set_yticks(range(len(lanes)), lanes)
ax.set_ylim(len(lanes) - 0.5, -0.5)                          # first lane on top
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
for h in leg.legend_handles:
    h.set_sizes([40])
ax.set_title("Event timeline 2023-2025 (bubble size = downtime, yellow bar = degradation period)")
save(fig, C.FIG / "fig03_event_timeline.png")


# ---------------------------------------------------------------------------
# 3.4 Weibull analysis on operating hours between failures
#     - time is measured in RUNNING hours (a stopped pump does not age)
#     - the period after the last failure is "right-censored" (still surviving)
# ---------------------------------------------------------------------------
def tbf_running_hours(asset):
    """Running hours between failures, plus the censored time after the last one."""
    run_counter = status[STATUS_COL[asset]].eq("Running").cumsum()
    fail_times = failures[failures.asset_id == asset].EventStart.sort_values()
    counter_at_fail = run_counter.reindex(fail_times, method="ffill").fillna(0).to_numpy()
    tbf = np.diff(counter_at_fail, prepend=0)
    last = counter_at_fail[-1] if len(counter_at_fail) else 0
    return tbf[tbf > 0], run_counter.iloc[-1] - last


def weibull_mle(fail_times, censored_times):
    """Maximum-likelihood Weibull fit (beta = shape, eta = scale) with right-censoring."""
    t_f = np.asarray(fail_times, float)
    t_c = np.asarray(censored_times, float)

    def neg_log_lik(p):
        beta, eta = np.exp(p)                                  # log-params keep them > 0
        ll = np.sum(np.log(beta / eta) + (beta - 1) * np.log(t_f / eta) - (t_f / eta) ** beta)
        ll -= np.sum((t_c / eta) ** beta)
        return -ll

    res = optimize.minimize(neg_log_lik, x0=[0.0, np.log(t_f.mean())], method="Nelder-Mead")
    return np.exp(res.x)


def failure_pattern(beta):
    if beta < 0.9:
        return "infant mortality (decreasing failure rate)"
    if beta <= 1.1:
        return "random failures (constant failure rate)"
    return "wear-out (increasing failure rate)"


groups = {"Pumps (P-201A/B)": ["P-201A", "P-201B"],
          "Compressor train (K-101 + GT-101)": ["K-101", "GT-101"]}
wb_rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (gname, assets) in zip(axes, groups.items()):
    fails, cens = [], []
    for a in assets:
        t, c = tbf_running_hours(a)
        fails.extend(t)
        cens.append(c)
    beta, eta = weibull_mle(fails, cens)
    wb_rows.append({
        "group": gname, "n_failures": len(fails), "n_censored": len(cens),
        "beta_shape": round(beta, 2), "eta_scale_h": round(eta),
        "pattern": failure_pattern(beta),
        "reliability_30d_running": round(np.exp(-((720 / eta) ** beta)), 3),
        "B10_life_h": round(eta * (-np.log(0.9)) ** (1 / beta)),
    })

    # Weibull probability plot (Benard median ranks, failures only)
    tf = np.sort(fails)
    F = (np.arange(1, len(tf) + 1) - 0.3) / (len(tf) + 0.4)
    xx = np.linspace(np.log(tf.min() * 0.7), np.log(tf.max() * 1.3), 50)
    ax.scatter(np.log(tf), np.log(-np.log(1 - F)), color=BLUE, s=36, zorder=3,
               edgecolor="white", linewidth=0.8, label="observed")
    ax.plot(xx, beta * (xx - np.log(eta)), color=ORANGE,
            label=f"fit  β={beta:.2f}, η={eta:,.0f} h")
    ax.set(title=gname, xlabel="ln(running hours between failures)", ylabel="ln(-ln(1-F))")
    ax.legend()
fig.suptitle("Weibull probability plots", x=0.02, ha="left")
fig.tight_layout()
save(fig, C.FIG / "fig04_weibull.png")

wb = pd.DataFrame(wb_rows)
wb.to_csv(C.PROC / "weibull_results.csv", index=False)
print(wb.to_string(index=False))


# ---------------------------------------------------------------------------
# 3.5 Warning time: true P-F interval vs. first DCS alarm
#     (business case for predictive maintenance)
# ---------------------------------------------------------------------------
prog = ev[(ev.has_degradation_onset == 1) & (ev.is_failure == 1)].copy()


def first_alarm(r):
    a = al[(al.asset_id == r.asset_id) & al.AlarmStart.between(r.DegradationOnset, r.EventStart)]
    return a.AlarmStart.min()


prog["first_alarm"] = pd.to_datetime(prog.apply(first_alarm, axis=1))
prog["alarm_lead_days"] = (prog.EventStart - prog.first_alarm).dt.total_seconds() / 86400
prog["label"] = prog.EventID + " " + prog.asset_id + " " + prog.FailureModeCode
prog = prog.sort_values("pf_interval_days")
prog[["EventID", "asset_id", "FailureModeCode", "pf_interval_days", "alarm_lead_days"]] \
    .to_csv(C.PROC / "warning_time_analysis.csv", index=False)

med_pf, med_al = prog.pf_interval_days.median(), prog.alarm_lead_days.median()
y = np.arange(len(prog))
alarm_x = prog.alarm_lead_days.fillna(0)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hlines(y, alarm_x, prog.pf_interval_days, color=LIGHT, lw=3)
ax.scatter(prog.pf_interval_days, y, color=BLUE, s=60, zorder=3,
           label="degradation detectable (P-F interval)")
ax.scatter(alarm_x, y, color=CRITICAL, s=60, zorder=3, label="first DCS alarm")
ax.set_yticks(y, prog.label, fontsize=8)
ax.set_xlabel("days before the failure / trip")
ax.set_title(f"Alarms warned a median {med_al:.1f} days ahead; "
             f"degradation was visible {med_pf:.0f} days ahead")
ax.legend(loc="lower right")
save(fig, C.FIG / "fig05_warning_time_gap.png")

print(f"  median P-F interval: {med_pf:.1f} d | median alarm lead: {med_al:.1f} d")
print("  figures saved to", C.FIG)