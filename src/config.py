"""
config.py - central settings for the OGP1 Equipment Reliability project.

Every script imports this file, so folder paths, file names and asset
definitions live in ONE place. If you rename a file, change it here only.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders (all relative to the project root, so the repo works on any PC)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"              # original Excel files (never edited)
PROC = ROOT / "data" / "processed"       # cleaned CSVs + SQLite database
FIG = ROOT / "reports" / "figures"       # PNG charts for README / LinkedIn
REPORTS = ROOT / "reports"
PBI = ROOT / "powerbi" / "data"          # CSVs that Power BI connects to
SQL_DIR = ROOT / "sql"
DB_PATH = PROC / "ogp1_reliability.db"

for p in (PROC, FIG, PBI):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
SENSOR_FILE = RAW / "Facility_equipment_data.xlsx"
EVENT_FILE = RAW / "equipment_event_failure_event.xlsx"
ALARM_FILE = RAW / "Alarm_log.xlsx"

# Text values the PI historian writes instead of a number
QUALITY_CODES = ["Bad Input", "I/O Timeout", "Shutdown", "Calc Failed", "No Data"]

# ---------------------------------------------------------------------------
# Asset master data (what a CMMS / SAP PM functional-location list holds)
# tag_prefix = the part of the PI tag that identifies the equipment
# ---------------------------------------------------------------------------
ASSETS = [
    # asset_id, tag_prefix, asset_name, equipment_class, criticality, has_run_status
    ("K-101",  "K101",  "Gas compressor",                 "Centrifugal compressor", "A", True),
    ("GT-101", "GT101", "Gas turbine (K-101 driver)",     "Gas turbine",            "A", True),
    ("P-201A", "P201A", "Export pump A",                  "Centrifugal pump",       "B", True),
    ("P-201B", "P201B", "Export pump B",                  "Centrifugal pump",       "B", True),
    ("V-301",  "V301",  "HP separator",                   "Pressure vessel",        "A", False),
    ("E-401",  "E401",  "Gas cooler / heat exchanger",    "Heat exchanger",         "B", False),
    ("TK-501", "TK501", "Storage tank",                   "Atmospheric tank",       "C", False),
    ("FLARE",  "FLR",   "Flare system",                   "Flare",                  "A", False),
    ("SITE",   "SITE",  "Site ambient conditions",        "Ambient",                "-", False),
]
PREFIX_TO_ASSET = {a[1]: a[0] for a in ASSETS}

# Assets with a RUN_STATUS tag -> we can compute operating hours, MTBF, availability
ROTATING = ["K-101", "GT-101", "P-201A", "P-201B"]

# How the free-text 'Asset' column of the event log maps to a clean asset_id
EVENT_ASSET_MAP = {
    "K-101": "K-101", "GT-101": "GT-101", "P-201A": "P-201A", "P-201B": "P-201B",
    "V-301": "V-301", "E-401": "E-401",
    "P-201A/B": "P-201 (A/B system)", "TK-501 / P-201": "TK-501",
    "PLANT": "PLANT", "PI SYSTEM": "PI SYSTEM",
}

# Event types that count as an equipment FAILURE for reliability KPIs
FAILURE_TYPES = ["Failure / Trip"]
UNPLANNED_TYPES = ["Failure / Trip", "Corrective Maintenance"]

# Time split for the machine-learning model (train on the past, test on the future)
TRAIN_END = "2024-12-31 23:00"
