"""
STEP 6 - FAILURE PREDICTION (PREDICTIVE MAINTENANCE)
=====================================================
Input : data/processed/sensor_hourly_clean.csv, run_status_hourly.csv,
        events_clean.csv, alarms_clean.csv
Output: data/processed/predictions_hourly.csv     health index + ML risk per asset-hour
        data/processed/model_metrics.csv           hour-level metrics on the 2025 test year
        data/processed/model_event_detection.csv   warning time for every 2025 failure
        data/processed/feature_importance.csv
        reports/figures/fig11..fig14
        models/*.joblib

THE BUSINESS QUESTION
  "Can the sensor data tell us that a machine has started to degrade - early
   enough to plan the repair instead of suffering a trip?"

THE APPROACH (two layers - the way industrial APM tools such as AVEVA PRiSM or
GE SmartSignal work)

  Layer A - NORMAL-BEHAVIOUR MODEL  (unsupervised: no failure labels needed)
      For each condition sensor, learn what it SHOULD read given the operating
      context (ambient temperature, flow, speed, suction pressure), using only
      HEALTHY hours from 2023-2024. The residual (actual - expected) tells us
      how abnormal the machine is, independently of load or weather.
      Residuals -> robust z-scores -> HEALTH INDEX (= largest smoothed |z|).
      Alert above the 99th percentile of healthy training hours.

  Layer B - SUPERVISED CLASSIFIER  (uses the failure labels)
      Target y = 1 inside the P-F interval [DegradationOnset, EventStart).
      Features = rolling statistics of the residual z-scores.
      Gradient boosting with class weights (degrading hours are rare).

  Baseline - "is a DCS alarm active on this asset?" (what operators have today)

  Validation: train 2023-2024, test on 2025 (unseen future). A random split
  would leak the future into the past and give unrealistically good scores.

Run:  python src/06_failure_prediction.py      (~2 minutes)
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

import config as C
from style import (apply_style, save, BLUE, ORANGE, AQUA, INK, LIGHT,
                   WARNING, CRITICAL)

apply_style()
print("STEP 6 - Failure prediction")

MODELS = C.ROOT / "models"
MODELS.mkdir(exist_ok=True)
TRAIN_END = pd.Timestamp(C.TRAIN_END)

X = pd.read_csv(C.PROC / "sensor_hourly_clean.csv", parse_dates=["Timestamp"], index_col="Timestamp")
S = pd.read_csv(C.PROC / "run_status_hourly.csv", parse_dates=["Timestamp"], index_col="Timestamp")
ev = pd.read_csv(C.PROC / "events_clean.csv",
                 parse_dates=["DegradationOnset", "EventStart", "ReturnToService"])
al = pd.read_csv(C.PROC / "alarms_clean.csv", parse_dates=["AlarmStart", "AlarmEnd"])

# ---------------------------------------------------------------------------
# What we model: one "unit" = one machine with its own normal-behaviour model.
#   context = operating-condition tags (model inputs)
#   every other tag of the unit = condition indicator (model target)
# ---------------------------------------------------------------------------
UNITS = {
    "P-201A": dict(group="pumps", prefix=("OGP1.P201A_",), status="OGP1.P201A_RUN_STATUS",
                   assets=["P-201A"],
                   context=["OGP1.SITE_TT0001", "OGP1.P201A_FT5003", "OGP1.P201A_PT5001"]),
    "P-201B": dict(group="pumps", prefix=("OGP1.P201B_",), status="OGP1.P201B_RUN_STATUS",
                   assets=["P-201B"],
                   context=["OGP1.SITE_TT0001", "OGP1.P201B_FT5003", "OGP1.P201B_PT5001"]),
    "K-101/GT-101": dict(group="compressor", prefix=("OGP1.K101_", "OGP1.GT101_"),
                         status="OGP1.K101_RUN_STATUS", assets=["K-101", "GT-101"],
                         context=["OGP1.SITE_TT0001", "OGP1.K101_FT1003", "OGP1.K101_PT1001",
                                  "OGP1.K101_ST1004"]),
}
progressive = ev[(ev.is_unplanned == 1) & ev.DegradationOnset.notna()]   # failures with a P-F interval


def make_label(assets, index):
    """y = 1 inside [DegradationOnset, EventStart); NaN during the outage (excluded)."""
    y = pd.Series(0.0, index=index)
    for r in progressive[progressive.asset_id.isin(assets)].itertuples():
        y[(index >= r.DegradationOnset) & (index < r.EventStart)] = 1
    for r in ev[ev.asset_id.isin(assets)].itertuples():
        y[(index >= r.EventStart) & (index < r.ReturnToService)] = np.nan
    return y


def short(tag):
    """'OGP1.P201A_VT5010' -> 'PUMP_VT5010' (both pumps share one feature name)."""
    return tag.replace("OGP1.", "").replace("P201A_", "PUMP_").replace("P201B_", "PUMP_")


def days_between(later, earlier):
    return round((later - earlier).total_seconds() / 86400, 1)


# ---------------------------------------------------------------------------
# 6.1 Layer A - normal-behaviour models and residual z-scores
# ---------------------------------------------------------------------------
unit_data, nb_models = {}, {}
for unit, cfg in UNITS.items():
    ctx = cfg["context"]
    targets = [c for c in X.columns if c.startswith(cfg["prefix"]) and c not in ctx]

    running = S[cfg["status"]].eq("Running")
    hours_since_start = running.groupby((~running).cumsum()).cumsum()
    usable = running & (hours_since_start > 6)                # skip start-up transients
    ctx_ok = X[ctx].notna().all(axis=1)
    y = make_label(cfg["assets"], X.index)
    healthy_train = usable & y.eq(0) & (X.index <= TRAIN_END)
    predict_rows = usable & ctx_ok

    Z, expected, nb_models[unit] = pd.DataFrame(index=X.index), pd.DataFrame(index=X.index), {}
    for t in targets:
        fit_rows = healthy_train & ctx_ok & X[t].notna()
        reg = HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=100, random_state=0)
        reg.fit(X.loc[fit_rows, ctx], X.loc[fit_rows, t])

        pred = pd.Series(np.nan, index=X.index)
        pred[predict_rows] = reg.predict(X.loc[predict_rows, ctx])
        resid = X[t] - pred
        centre = resid[fit_rows].median()
        scale = 1.4826 * (resid[fit_rows] - centre).abs().median() + 1e-9   # robust std (MAD)

        Z[short(t)] = (resid - centre) / scale
        expected[t] = pred
        nb_models[unit][t] = {"model": reg, "centre": centre, "scale": scale}

    # Features for Layer B: 24 h mean, 7 day mean and 3 day trend of each z-score
    z24 = Z.rolling(24, min_periods=6).mean()
    F = pd.concat([z24.add_suffix("__z24h"),
                   Z.rolling(168, min_periods=24).mean().add_suffix("__z7d"),
                   (z24 - z24.shift(72)).add_suffix("__ztrend3d")], axis=1)
    F = F[[f"{c}__{s}" for c in Z.columns for s in ("z24h", "z7d", "ztrend3d")]]  # sensor by sensor

    # Health index = largest |z24h| of any sensor; driver = the sensor responsible
    abs_z = z24.abs()
    has_data = abs_z.notna().any(axis=1)
    F["health_index"] = abs_z.max(axis=1).fillna(0)             # no data -> no evidence
    F["hi_driver"] = None
    F.loc[has_data, "hi_driver"] = abs_z[has_data].idxmax(axis=1)
    F["y"], F["unit"], F["group"] = y, unit, cfg["group"]

    unit_data[unit] = {"F": F[usable], "expected": expected, "Z": Z}
    print(f"  {unit:13s} normal-behaviour models: {len(targets)} sensors | "
          f"healthy training hours: {int(healthy_train.sum()):,}")
joblib.dump(nb_models, MODELS / "normal_behaviour_models.joblib")


# ---------------------------------------------------------------------------
# 6.2 Layer B (classifier) + health-index threshold, per equipment group
# ---------------------------------------------------------------------------
def active_alarm(assets, index):
    """1 for every hour a DCS alarm was active on the assets."""
    flag = pd.Series(0, index=index)
    for r in al[al.asset_id.isin(assets)].itertuples():
        flag[(index >= r.AlarmStart) & (index <= r.AlarmEnd)] = 1
    return flag


def sustained(alert, units, hours=6):
    """Alert only after N consecutive hours above threshold (avoids blips)."""
    return alert.groupby(units).transform(
        lambda s: s.astype(int).rolling(hours, min_periods=hours).sum().eq(hours)).astype(bool)


ML_THRESHOLD = 0.5
metrics, preds, detection, importance = [], [], [], []

for group in ("pumps", "compressor"):
    group_units = [u for u in UNITS if UNITS[u]["group"] == group]
    d = pd.concat([unit_data[u]["F"] for u in group_units]).dropna(subset=["y"]).sort_index()
    feat_cols = [c for c in d.columns if "__" in c]
    tr = d[d.index <= TRAIN_END]

    hi_thr = float(tr.loc[tr.y == 0, "health_index"].quantile(0.99))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                                         l2_regularization=1.0, class_weight="balanced",
                                         random_state=42).fit(tr[feat_cols], tr.y)
    joblib.dump({"model": clf, "features": feat_cols, "ml_threshold": ML_THRESHOLD,
                 "hi_threshold": hi_thr}, MODELS / f"{group}_classifier.joblib")

    # Scores and alerts for every hour (train + test)
    raw_risk = pd.Series(clf.predict_proba(d[feat_cols])[:, 1], index=d.index)
    d["ml_risk"] = raw_risk.groupby(d.unit).transform(lambda s: s.rolling(12, min_periods=1).mean())
    d["ml_alert"] = sustained(d.ml_risk >= ML_THRESHOLD, d.unit)
    d["hi_alert"] = sustained(d.health_index >= hi_thr, d.unit)
    d["dcs_alarm"] = 0
    for u in group_units:
        m = d.unit == u
        d.loc[m, "dcs_alarm"] = active_alarm(UNITS[u]["assets"], d.index[m]).values
    te = d[d.index > TRAIN_END]

    # Hour-level metrics on 2025
    methods = {"Health index (unsupervised)": (te.hi_alert, te.health_index),
               "ML classifier (supervised)": (te.ml_alert, te.ml_risk),
               "DCS alarm (current practice)": (te.dcs_alarm.astype(bool), te.dcs_alarm)}
    for method, (alert, score) in methods.items():
        episode_start = alert & ~alert.groupby(te.unit).shift(1, fill_value=False)
        metrics.append({
            "group": group, "method": method,
            "precision": round(precision_score(te.y, alert, zero_division=0), 3),
            "recall": round(recall_score(te.y, alert, zero_division=0), 3),
            "f1": round(f1_score(te.y, alert, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(te.y, score), 3),
            "pr_auc": round(average_precision_score(te.y, score), 3),
            "healthy_hours_in_alert_pct": round(100 * alert[te.y == 0].mean(), 2),
            "false_alert_episodes": int((episode_start & te.y.eq(0)).sum()),
        })

    # Event level: warning time for each 2025 progressive failure
    assets = [a for u in group_units for a in UNITS[u]["assets"]]
    for r in progressive[progressive.asset_id.isin(assets) & (progressive.EventStart > TRAIN_END)].itertuples():
        unit = r.asset_id if group == "pumps" else "K-101/GT-101"
        w = te[(te.unit == unit) & (te.index >= r.DegradationOnset) & (te.index < r.EventStart)]
        row = {"EventID": r.EventID, "asset_id": r.asset_id, "FailureModeCode": r.FailureModeCode,
               "FailureMode": r.FailureMode, "pf_interval_days": r.pf_interval_days}
        for col, name in (("hi_alert", "hi"), ("ml_alert", "ml")):
            row[f"{name}_lead_days"] = days_between(r.EventStart, w.index[w[col]].min()) \
                if w[col].any() else np.nan
        if w.hi_alert.any():
            row["hi_main_driver"] = w.loc[w.hi_alert, "hi_driver"].mode().iloc[0]
        a = al[(al.asset_id == r.asset_id) & al.AlarmStart.between(r.DegradationOnset, r.EventStart)]
        row["alarm_lead_days"] = days_between(r.EventStart, a.AlarmStart.min()) if len(a) else np.nan
        detection.append(row)

    # Which features drive the classifier (permutation importance on 2025)
    sample = te.sample(min(len(te), 4000), random_state=0)
    pi = permutation_importance(clf, sample[feat_cols], sample.y, scoring="average_precision",
                                n_repeats=3, random_state=0, n_jobs=-1)
    importance.append(pd.DataFrame({"group": group, "feature": feat_cols,
                                    "importance": pi.importances_mean}))

    d["hi_threshold"], d["ml_threshold"] = hi_thr, ML_THRESHOLD
    d["dataset"] = np.where(d.index > TRAIN_END, "test", "train")
    preds.append(d[["unit", "group", "health_index", "hi_driver", "hi_threshold", "hi_alert",
                    "ml_risk", "ml_threshold", "ml_alert", "dcs_alarm", "y", "dataset"]])


# ---------------------------------------------------------------------------
# Results tables
# Recommended method per group (decided from the 2025 test results):
#   pumps      -> ML classifier (fewest false alerts, all failures caught)
#   compressor -> health index  (classifier is almost always "on" = useless in practice)
# ---------------------------------------------------------------------------
met = pd.DataFrame(metrics)
imp = pd.concat(importance).sort_values(["group", "importance"], ascending=[True, False])

det = pd.DataFrame(detection)
is_pump = det.asset_id.str.startswith("P-")
det["recommended_method"] = np.where(is_pump, "ML classifier", "Health index")
det["model_lead_days"] = np.where(is_pump, det.ml_lead_days, det.hi_lead_days)

pr = pd.concat(preds).reset_index().rename(columns={"y": "actual_degrading"})
pumps = pr.group == "pumps"
pr["recommended_alert"] = np.where(pumps, pr.ml_alert, pr.hi_alert).astype(int)
pr["risk_ratio"] = np.where(pumps, pr.ml_risk / pr.ml_threshold,
                            pr.health_index / pr.hi_threshold).round(3)    # >= 1 means alert level
pr["health_index"] = pr.health_index.round(3)
pr["ml_risk"] = pr.ml_risk.round(4)

met.to_csv(C.PROC / "model_metrics.csv", index=False)
det.to_csv(C.PROC / "model_event_detection.csv", index=False)
imp.to_csv(C.PROC / "feature_importance.csv", index=False)
pr.to_csv(C.PROC / "predictions_hourly.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("\n  TEST YEAR 2025 - hour-level performance")
print(met.to_string(index=False))
print("\n  TEST YEAR 2025 - warning time per failure (days before the trip)")
print(det.drop(columns=["FailureMode"]).to_string(index=False))


# ---------------------------------------------------------------------------
# 6.3 Figures
# ---------------------------------------------------------------------------
# fig11 - 2025 timeline: both scores scaled so that 1.0 = alert threshold
hours_2025 = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, unit in zip(axes, UNITS):
    p = pr[(pr.unit == unit) & (pr.dataset == "test")].set_index("Timestamp").reindex(hours_2025)
    ax.fill_between(hours_2025, 0, 3, where=p.actual_degrading.eq(1), color=WARNING, alpha=0.25,
                    step="mid", lw=0, label="true degradation period (P-F)")
    ax.plot(hours_2025, (p.health_index / p.hi_threshold).clip(upper=3), color=BLUE, lw=0.9,
            label="health index / threshold")
    ax.plot(hours_2025, (p.ml_risk / p.ml_threshold).clip(upper=3), color=AQUA, lw=0.9,
            label="ML risk / threshold")
    ax.axhline(1, ls="--", color=INK, lw=1, label="alert level")
    fails_2025 = ev[ev.asset_id.isin(UNITS[unit]["assets"]) & (ev.is_failure == 1)
                    & (ev.EventStart.dt.year == 2025)]
    for t in fails_2025.EventStart:
        ax.axvline(t, color=CRITICAL, lw=1.6)
    ax.set(ylim=(0, 3.05), ylabel=unit)
axes[0].set_title("2025 test year (never seen in training) - red line = failure / trip")
fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 0.02))
save(fig, C.FIG / "fig11_risk_timeline_2025.png")

# fig12 - warning time per 2025 failure, using the recommended method per group
d2 = det.sort_values("pf_interval_days")
labels = d2.asset_id + "  " + d2.FailureMode.str.slice(0, 30) + "  [" + d2.recommended_method + "]"
y, h = np.arange(len(d2)), 0.27
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.barh(y + h, d2.pf_interval_days, h, color=LIGHT, label="P-F interval (maximum possible warning)")
ax.barh(y, d2.model_lead_days.fillna(0), h, color=BLUE, label="model alert")
ax.barh(y - h, d2.alarm_lead_days.fillna(0), h, color=ORANGE, label="first DCS alarm")
ax.set_yticks(y, labels, fontsize=8)
ax.set_xlabel("days of warning before the failure")
ax.set_title(f"2025 failures: model caught {d2.model_lead_days.notna().sum()}/{len(d2)}, "
             f"median warning {d2.model_lead_days.median():.0f} days vs "
             f"{d2.alarm_lead_days.median():.0f} days for DCS alarms")
ax.legend(loc="lower right")
save(fig, C.FIG / "fig12_lead_time_model_vs_alarm.png")

# fig13 - what drives the classifier
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, group in zip(axes, ("pumps", "compressor")):
    top = imp[imp.group == group].head(12).iloc[::-1]
    ax.barh(top.feature, top.importance, color=BLUE, height=0.65)
    ax.set(title=f"Top features - {group} classifier",
           xlabel="permutation importance (drop in PR-AUC)")
    ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
save(fig, C.FIG / "fig13_feature_importance.png")

# fig14 - how the normal-behaviour model "sees" a failure: last P-201B bearing failure in 2025
brg = progressive[(progressive.asset_id == "P-201B") & (progressive.EventStart > TRAIN_END)
                  & progressive.FailureModeCode.str.contains("BRG")].sort_values("EventStart")
if brg.empty:
    print("  fig14 skipped: no P-201B bearing failure in the test year")
else:
    onset, fail = brg.iloc[-1].DegradationOnset, brg.iloc[-1].EventStart
    w = slice(onset - pd.Timedelta("22D"), fail + pd.Timedelta("1D"))
    tag, u = "OGP1.P201B_VT5010", unit_data["P-201B"]
    run = S["OGP1.P201B_RUN_STATUS"].eq("Running")
    z = u["Z"]["PUMP_VT5010"].rolling(24, min_periods=6).mean()[w]
    hi_thr_p = pr.loc[pumps, "hi_threshold"].iloc[0]

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(X[tag].where(run)[w], color=BLUE, lw=0.8, label="measured")
    ax[0].plot(u["expected"][tag][w], color=ORANGE, lw=1.6, label="expected (normal-behaviour model)")
    ax[0].set(ylabel="vibration mm/s",
              title=f"P-201B DE bearing: measured vs expected vibration before the "
                    f"{fail:%b-%Y} failure")
    ax[1].plot(z, color=AQUA)
    ax[1].axhline(hi_thr_p, ls="--", color=INK, lw=1, label=f"health-index alert level ({hi_thr_p:.1f})")
    ax[1].set_ylabel("residual z-score (24 h)")
    for a_ in ax:
        a_.axvspan(onset, fail, color=WARNING, alpha=0.2, lw=0)
        a_.axvline(fail, color=CRITICAL, lw=1.5)
        a_.legend(loc="upper left")
    save(fig, C.FIG / "fig14_normal_behaviour_example.png")

print("\n  models saved to", MODELS, "| figures saved")