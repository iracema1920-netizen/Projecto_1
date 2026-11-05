"""
STEP 5 - CONDITION MONITORING EDA (what does degradation look like in the data?)
================================================================================
Input : data/processed/sensor_hourly_clean.csv, run_status_hourly.csv,
        events_clean.csv, tag_dictionary.csv
Output: reports/figures/fig08_degradation_signatures.png
        reports/figures/fig09_k101_correlation.png
        reports/figures/fig10_ambient_effect.png

Before building a model, LOOK at the data. Here we plot the sensor that should
react to each failure mode (your maintenance experience is the advantage here)
and check whether the degradation is visible before the failure date.

Run:  python src/05_condition_monitoring_eda.py
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as C
from style import (apply_style, save, BLUE, ORANGE, INK, DIVERGING,
                   WARNING, SERIOUS, CRITICAL)

apply_style()
print("STEP 5 - Condition monitoring EDA")

X = pd.read_csv(C.PROC / "sensor_hourly_clean.csv", parse_dates=["Timestamp"], index_col="Timestamp")
S = pd.read_csv(C.PROC / "run_status_hourly.csv", parse_dates=["Timestamp"], index_col="Timestamp")
ev = pd.read_csv(C.PROC / "events_clean.csv",
                 parse_dates=["DegradationOnset", "EventStart", "ReturnToService"])
tags = pd.read_csv(C.PROC / "tag_dictionary.csv").set_index("tag")


def running(tag, status_col):
    """Tag values only for the hours the machine was running (NaN otherwise)."""
    return X[tag].where(S[status_col].eq("Running"))


def shade_events(ax, codes, end=None):
    """Yellow band = degradation period, red line = failure / repair."""
    e = ev[ev.FailureModeCode.isin(codes)]
    if end:
        e = e[e.EventStart <= pd.Timestamp(end)]
    for r in e.itertuples():
        if pd.notna(r.DegradationOnset):
            ax.axvspan(r.DegradationOnset, r.EventStart, color=WARNING, alpha=0.22, lw=0)
        ax.axvline(r.EventStart, color=CRITICAL, lw=1.2)


def limit_lines(ax, tag):
    """Dashed lines for the DCS alarm limits of a tag (L, H, HH)."""
    for col, name, colr in (("limit_L", "L", SERIOUS), ("limit_H", "H", SERIOUS),
                            ("limit_HH", "HH", CRITICAL)):
        v = tags.loc[tag, col]
        if pd.notna(v):
            ax.axhline(v, ls="--", color=colr, lw=1)
            ax.annotate(f"{name} {v:g} ", (1, v), xycoords=("axes fraction", "data"),
                        color=colr, fontsize=7.5, ha="right", va="bottom")


def daily_panel(ax, tag, status_col, codes, title, smooth_days=None):
    """Daily mean of a tag while running, with events and alarm limits."""
    s = running(tag, status_col).resample("D").mean()
    ax.plot(s.index, s, color=BLUE, lw=1.2, label="daily mean")
    if smooth_days:
        ax.plot(s.index, s.rolling(smooth_days, min_periods=smooth_days // 3).mean(),
                color=INK, lw=2, label=f"{smooth_days}-day mean")
        ax.legend(loc="upper left")
    shade_events(ax, codes)
    limit_lines(ax, tag)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))


# ---------------------------------------------------------------------------
# Figure 8 - four typical degradation signatures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# (a) Pump bearing failure - hourly vibration in the weeks before the trip
ax = axes[0, 0]
s = running("OGP1.P201A_VT5010", "OGP1.P201A_RUN_STATUS")["2023-04-01":"2023-05-20"]
ax.plot(s.index, s, lw=0.7, color=BLUE, alpha=0.7, label="hourly")
ax.plot(s.index, s.rolling(24, min_periods=6).mean(), lw=2, color=INK, label="24 h mean")
shade_events(ax, ["P201A_BRG"], end="2023-06-01")
limit_lines(ax, "OGP1.P201A_VT5010")
ax.set_title("P-201A DE bearing vibration (mm/s) - bearing failure May-2023")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax.legend(loc="upper left")

# (b)-(d) daily trends over three years
daily_panel(axes[0, 1], "OGP1.K101_EFF_POLY", "OGP1.K101_RUN_STATUS", ["K101_FOUL"],
            "K-101 polytropic efficiency (%) - fouling & washes")
daily_panel(axes[1, 0], "OGP1.GT101_PDT2007", "OGP1.GT101_RUN_STATUS", ["GT101_INLETF"],
            "GT-101 air inlet filter dP (mbar) - saw-tooth = clog & replace")
daily_panel(axes[1, 1], "OGP1.E401_TT4002", "OGP1.K101_RUN_STATUS", ["E401_FOUL"],
            "E-401 process outlet temperature (°C) - fouling", smooth_days=30)

fig.suptitle("Degradation signatures (yellow band = degradation period, red line = failure / repair)",
             x=0.01, ha="left")
fig.tight_layout()
save(fig, C.FIG / "fig08_degradation_signatures.png")


# ---------------------------------------------------------------------------
# Figure 9 - correlation between K-101 signals (daily means while running)
#            Helps to choose model features and avoid redundant ones.
# ---------------------------------------------------------------------------
k_cols = [c for c in X.columns if c.startswith("OGP1.K101")]
k = (X[k_cols].where(S["OGP1.K101_RUN_STATUS"].eq("Running"), axis=0)
       .resample("D").mean()
       .rename(columns=lambda c: c.replace("OGP1.K101_", "")))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(k.corr(), cmap=DIVERGING, center=0, vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, linecolor="white", cbar_kws={"label": "Pearson r", "shrink": 0.8})
ax.set_title("K-101 sensor correlation (daily means while running)")
save(fig, C.FIG / "fig09_k101_correlation.png")


# ---------------------------------------------------------------------------
# Figure 10 - operating context: hot days reduce gas-turbine power (air density).
#   A model must not confuse "hot weather" with "degradation" -> ambient
#   temperature is used as a context input in step 6.
# ---------------------------------------------------------------------------
d = pd.DataFrame({"amb": X["OGP1.SITE_TT0001"],
                  "power": running("OGP1.GT101_JT2002", "OGP1.GT101_RUN_STATUS"),
                  "egt": running("OGP1.GT101_TT2003", "OGP1.GT101_RUN_STATUS")}).dropna()
d = d[d.power > 5]
d = d.sample(min(5000, len(d)), random_state=1)

slope, intercept = np.polyfit(d.amb, d.power, 1)
xx = np.linspace(d.amb.min(), d.amb.max(), 10)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(d.amb, d.power, s=4, alpha=0.25, color=BLUE, lw=0)
ax[0].plot(xx, slope * xx + intercept, color=ORANGE, lw=2.2, label=f"slope {slope:.2f} MW/°C")
ax[0].set(xlabel="ambient temperature (°C)", ylabel="GT-101 power (MW)",
          title="Hot weather reduces GT power")
ax[0].legend()
ax[1].scatter(d.amb, d.egt, s=4, alpha=0.25, color=BLUE, lw=0)
ax[1].set(xlabel="ambient temperature (°C)", ylabel="exhaust gas temperature (°C)",
          title="…and raises exhaust temperature")
fig.tight_layout()
save(fig, C.FIG / "fig10_ambient_effect.png")
print("  figures saved")