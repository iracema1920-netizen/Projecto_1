"""
STEP 1 - DATA CLEANING & DATA DICTIONARY
=========================================
Input : data/raw/*.xlsx  (PI historian export, event log, alarm log)
Output: data/processed/
          sensor_hourly_clean.csv   wide table, 1 row per hour, numeric only
          sensor_quality_flags.csv  what was removed and why (audit trail)
          run_status_hourly.csv     equipment running state per hour
          tag_dictionary.csv        English data dictionary (tag -> meaning, unit, alarm limits)
          events_clean.csv          event / failure log with derived columns
          alarms_clean.csv          alarm log with derived columns
          data_quality_summary.csv  % good data per tag

Why this step matters (say this in interviews):
  Real historian data is never clean. The PI system writes TEXT ('I/O Timeout',
  'Bad Input', 'Calc Failed'...) into numeric columns, instruments spike or
  freeze, and the tag legend is not always right. Every later result depends
  on handling this transparently, so we FLAG problems instead of silently
  deleting rows.

Run:  python src/01_data_cleaning.py
"""
import numpy as np
import pandas as pd

import config as C

print("STEP 1 - Data cleaning")

# ---------------------------------------------------------------------------
# 1.1 Load the raw data
# ---------------------------------------------------------------------------
raw = pd.read_excel(C.SENSOR_FILE, sheet_name="MASTER")
events = pd.read_excel(C.EVENT_FILE)
alarms = pd.read_excel(C.ALARM_FILE)
print(f"  sensor rows: {len(raw):,} | columns: {raw.shape[1]} | "
      f"{raw.Timestamp.min()} -> {raw.Timestamp.max()}")
print(f"  events: {len(events)} | alarms: {len(alarms)}")

# The time index must be one row per hour, with no gaps and no duplicates
assert raw.Timestamp.is_monotonic_increasing, "timestamps are not sorted"
assert not raw.Timestamp.duplicated().any(), "duplicate timestamps"
assert raw.Timestamp.diff().dropna().eq(pd.Timedelta("1h")).all(), "gaps in the hourly index"
raw = raw.set_index("Timestamp")

status_cols = [c for c in raw.columns if "RUN_STATUS" in c or "DUTY_PUMP" in c]
num_cols = [c for c in raw.columns if c not in status_cols]

flags = []   # audit trail: one small table (Timestamp, tag, flag) per problem found


def add_flags(mask, tag, label):
    """Record every hour where `mask` is True for `tag`."""
    if mask.any():
        flags.append(pd.DataFrame({"Timestamp": mask.index[mask], "tag": tag, "flag": label}))


# ---------------------------------------------------------------------------
# 1.2 Text quality codes -> NaN (keep the original text in the audit trail)
# ---------------------------------------------------------------------------
num = pd.DataFrame(index=raw.index)
for c in num_cols:
    as_num = pd.to_numeric(raw[c], errors="coerce")
    is_text = as_num.isna() & raw[c].notna()
    if is_text.any():
        flags.append(pd.DataFrame({"Timestamp": raw.index[is_text], "tag": c,
                                   "flag": raw.loc[is_text, c].astype(str).values}))
    num[c] = as_num

# Status tags: quality codes become 'Unknown'
status = raw[status_cols].mask(raw[status_cols].isin(C.QUALITY_CODES), "Unknown")

# ---------------------------------------------------------------------------
# 1.3 Physical range checks (values that cannot be true)
# ---------------------------------------------------------------------------
percent_tags = [c for c in num_cols if any(k in c for k in ("_MT", "_LT", "_LCV", "_FV", "EFF_POLY"))]
for c in percent_tags:
    bad = (num[c] < 0) | (num[c] > 100)
    add_flags(bad, c, "Out of physical range")
    num.loc[bad, c] = np.nan

# Small negative flows are transmitter zero-drift -> clip to 0 (not a data error)
flow_tags = [c for c in num_cols if "_FT" in c]
num[flow_tags] = num[flow_tags].clip(lower=0)

# ---------------------------------------------------------------------------
# 1.4 Spike detection (Hampel filter: robust rolling median +/- MAD)
#     A spike is an isolated value far from its neighbours. Ambient tags are
#     excluded because weather changes fast and is not an instrument fault;
#     control valves, flare flow and tank level are excluded because they
#     legitimately move fast (surge protection, upsets, tank draw-down).
# ---------------------------------------------------------------------------
FAST_TAGS = ("SITE_", "_FV", "_LCV", "FLR_", "TK501_LT")
spike_cols = [c for c in num_cols if not any(k in c for k in FAST_TAGS)]
x = num[spike_cols]
med = x.rolling(13, center=True, min_periods=5).median()
mad = (x - med).abs().rolling(13, center=True, min_periods=5).median()
spike = ((x - med).abs() > np.maximum(8 * 1.4826 * mad, 0.30 * med.abs())) & (med.abs() > 0.5)


# Domain rule: a sudden change during a trip, start-up or duty swap is REAL
# process behaviour. Only call it a spike when the equipment state was stable
# for +/- 6 h around that hour.
def stable(status_series):
    codes = pd.Series(pd.factorize(status_series)[0], index=status_series.index)
    window = codes.rolling(13, center=True, min_periods=1)
    return window.max() == window.min()


stable_p201a = stable(status["OGP1.P201A_RUN_STATUS"])
stable_p201b = stable(status["OGP1.P201B_RUN_STATUS"])
stable_plant = stable(status["OGP1.K101_RUN_STATUS"])
for c in spike_cols:
    machine_stable = (stable_p201a if "P201A" in c else
                      stable_p201b if "P201B" in c else stable_plant)
    spike[c] &= machine_stable
    add_flags(spike[c], c, "Spike")
num[spike_cols] = num[spike_cols].mask(spike)

# ---------------------------------------------------------------------------
# 1.5 Frozen ("stuck") instruments: identical value for >= 24 h while the
#     process is running. V-301 had two 'INST_STUCK' events in the log.
# ---------------------------------------------------------------------------
plant_running = status["OGP1.K101_RUN_STATUS"].eq("Running")
for c in [c for c in num_cols if c.startswith(("OGP1.V301", "OGP1.K101", "OGP1.GT101", "OGP1.E401"))]:
    s = num[c]
    run_length = s.groupby((s != s.shift()).cumsum()).transform("size")
    stuck = (run_length >= 24) & s.notna() & plant_running
    add_flags(stuck, c, "Frozen value")
    num.loc[stuck, c] = np.nan

flags = (pd.concat(flags, ignore_index=True) if flags
         else pd.DataFrame(columns=["Timestamp", "tag", "flag"]))

# Short gaps (<= 3 h) are filled by time interpolation; longer gaps stay NaN
num_filled = num.interpolate(method="time", limit=3, limit_area="inside")

# ---------------------------------------------------------------------------
# 1.6 Data-quality summary per tag
# ---------------------------------------------------------------------------
dq = flags.pivot_table(index="tag", columns="flag", values="Timestamp", aggfunc="count", fill_value=0)
dq["total_flagged"] = dq.sum(axis=1)
dq = dq.reindex(num_cols, fill_value=0)
dq["pct_good"] = (100 * (1 - dq["total_flagged"] / len(num))).round(2)
dq = dq.rename_axis("tag").reset_index()
dq.columns.name = None
print(f"  flagged values: {len(flags):,} of {num.size:,} ({100 * len(flags) / num.size:.2f} %)")
print("  flags by type:", flags.flag.value_counts().to_dict())

# ---------------------------------------------------------------------------
# 1.7 English tag dictionary (legend sheet translated + alarm-log overrides)
# ---------------------------------------------------------------------------
legend = pd.read_excel(C.SENSOR_FILE, sheet_name="Legenda", header=3)
legend = legend[legend["Tag"].astype(str).str.startswith("OGP1.")]
legend = legend[["Tag", "Tipo instr.", "Unidade (provável)"]]
legend.columns = ["tag", "instrument_type", "unit"]

DESC_EN = {
    "SITE_TT0001": "Ambient temperature", "SITE_MT0002": "Ambient relative humidity",
    "V301_PT3001": "HP separator pressure", "V301_TT3002": "Separator temperature",
    "V301_LT3003": "Separator liquid level", "V301_LT3004": "Separator interface level",
    "V301_LCV3007": "Level control valve opening", "V301_FT3008": "Separator inlet flow",
    "V301_FT3005": "Separator gas outlet flow (to K-101)", "FLR_FT6001": "Flare gas flow",
    "K101_PT1001": "Compressor suction pressure", "K101_PT1002": "Compressor discharge pressure",
    "K101_TT1001": "Compressor suction temperature", "K101_TT1002": "Compressor discharge temperature",
    "K101_FT1003": "Compressed gas flow", "K101_ST1004": "Compressor speed",
    "K101_EFF_POLY": "Polytropic efficiency (calculated)",
    "K101_VT1010A": "Radial vibration DE journal bearing X",
    "K101_VT1010B": "Radial vibration DE journal bearing Y",
    "K101_VT1011A": "Radial vibration NDE journal bearing X",
    "K101_ZT1012": "Axial (thrust) displacement", "K101_TT1013": "Thrust bearing temperature",
    "K101_TT1014A": "DE journal bearing temperature", "K101_TT1014B": "NDE journal bearing temperature",
    "K101_PDT1015": "Lube oil filter differential pressure", "K101_PT1016": "Lube oil header pressure",
    "K101_TT1017": "Lube oil temperature", "K101_FT1020": "Dry gas seal primary vent flow",
    "K101_SURGE_MARGIN": "Surge margin (calculated)", "K101_FV1030": "Anti-surge valve position",
    "GT101_ST2001": "Turbine speed", "GT101_JT2002": "Turbine power",
    "GT101_TT2003": "Exhaust gas temperature", "GT101_TT2004": "Exhaust temperature spread (max-min TC)",
    "GT101_FT2005": "Fuel gas flow", "GT101_PT2006": "Air compressor discharge pressure",
    "GT101_PDT2007": "Air inlet filter differential pressure", "GT101_TT2008": "Lube oil header temperature",
    "GT101_VT2009": "Turbine vibration",
    "E401_TT4001": "Process inlet temperature", "E401_TT4002": "Process outlet temperature",
    "E401_TT4003": "Cooling medium inlet temperature", "E401_TT4004": "Cooling medium outlet temperature",
    "E401_PDT4005": "Process side differential pressure", "E401_FT4006": "Cooling medium flow",
    "TK501_LT5101": "Storage tank level", "TK501_TT5102": "Storage tank temperature",
    "P201_DUTY_PUMP": "Pump on duty (A or B)",
}
PUMP_DESC = {
    "PT5001": "Pump suction pressure", "PT5002": "Pump discharge pressure", "FT5003": "Pump flow",
    "IT5004": "Motor current", "VT5010": "Pump DE bearing vibration (velocity RMS)",
    "VT5011": "Pump NDE bearing vibration", "TT5012": "Pump DE bearing temperature",
    "PT5015": "Mechanical seal pot pressure", "LT5016": "Mechanical seal pot level",
    "TT5018": "Motor stator winding temperature",
}
DESC_EN.update({f"{p}_{k}": v for p in ("P201A", "P201B") for k, v in PUMP_DESC.items()})
DESC_EN.update({f"{k}_RUN_STATUS": "Running status (text)" for k in ("K101", "GT101", "P201A", "P201B")})

legend["short_tag"] = legend["tag"].str.replace("OGP1.", "", regex=False)
legend["description_en"] = legend["short_tag"].map(DESC_EN)
legend["asset_id"] = (legend["short_tag"].str.split("_").str[0]
                      .map(C.PREFIX_TO_ASSET).fillna("P-201 (A/B system)"))
legend["unit"] = legend["unit"].replace({"texto": "text", "µm": "um pk-pk"})

# Alarm limits per tag (from the alarm log = the DCS configuration)
LIMIT_COLS = ["limit_LL", "limit_L", "limit_H", "limit_HH"]
lim = alarms.pivot_table(index="Tag", columns="AlarmType", values="Limit", aggfunc="first")
lim.columns = [f"limit_{c}" for c in lim.columns]
lim = lim.reindex(columns=LIMIT_COLS)            # all four columns, even if a type is unused

tag_dict = (legend.merge(lim, left_on="tag", right_index=True, how="left")
                  .merge(dq[["tag", "pct_good"]], on="tag", how="left"))
tag_dict = tag_dict[["tag", "short_tag", "asset_id", "instrument_type", "description_en", "unit",
                     *LIMIT_COLS, "pct_good"]]

# Note for the README: the original legend was 'inferred from tag names'.
# Two tags were wrong when compared with the alarm configuration:
#   GT101_TT2004 legend = inlet air temperature  -> alarm log = exhaust temp SPREAD
#   K101_FT1020  legend = anti-surge recycle flow -> alarm log = dry gas seal vent flow
# The alarm-log meaning is used above (validated by the failure log 'DetectedBy').

# ---------------------------------------------------------------------------
# 1.8 Clean the event log
# ---------------------------------------------------------------------------
ev = events.rename(columns={"DegradationOnset_GroundTruth": "DegradationOnset"})
ev["asset_id"] = ev["Asset"].map(C.EVENT_ASSET_MAP).fillna(ev["Asset"])
ev["is_failure"] = ev["EventType"].isin(C.FAILURE_TYPES).astype(int)
ev["is_unplanned"] = ev["EventType"].isin(C.UNPLANNED_TYPES).astype(int)
ev["has_degradation_onset"] = ev["DegradationOnset"].notna().astype(int)
# P-F interval: time from first detectable degradation to functional failure
ev["pf_interval_days"] = ((ev["EventStart"] - ev["DegradationOnset"]).dt.total_seconds() / 86400).round(1)

is_sudden = ev["FailureModeCode"].str.contains("SPUR|FLAMEOUT|ESD|SURGE|CAV", na=False)
ev["failure_type"] = np.select([is_sudden, ev["has_degradation_onset"].eq(1)],
                               ["Sudden / spurious", "Progressive degradation"], default="Other")
ev.loc[ev["is_unplanned"].eq(0), "failure_type"] = "Not a failure"
ev["year"] = ev["EventStart"].dt.year
ev["month"] = ev["EventStart"].dt.strftime("%Y-%m")

# ---------------------------------------------------------------------------
# 1.9 Clean the alarm log
# ---------------------------------------------------------------------------
al = alarms.copy()
al["asset_id"] = al["Asset"]
al["is_standing"] = (al["Duration_h"] >= 24).astype(int)          # EEMUA 191: stale alarm
above_limit = al["AlarmType"].isin(["H", "HH"])                   # high alarms: peak above limit
al["exceedance_pct"] = (100 * np.where(above_limit, al["PeakValue"] - al["Limit"],
                                       al["Limit"] - al["PeakValue"]) / al["Limit"].abs()).round(1)
al["date"] = al["AlarmStart"].dt.date

# ---------------------------------------------------------------------------
# 1.10 Save everything
# ---------------------------------------------------------------------------
num_filled.round(4).to_csv(C.PROC / "sensor_hourly_clean.csv")    # Timestamp index kept
status.to_csv(C.PROC / "run_status_hourly.csv")
for name, df in {"sensor_quality_flags": flags, "data_quality_summary": dq,
                 "tag_dictionary": tag_dict, "events_clean": ev, "alarms_clean": al}.items():
    df.to_csv(C.PROC / f"{name}.csv", index=False)
print("  saved cleaned files to", C.PROC)