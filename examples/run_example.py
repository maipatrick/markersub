"""Analyse the example subject and print a short summary.

Run from the repository root:  python examples/run_example.py
"""
from pathlib import Path

from markersub import Settings, analyse_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "example_data"          # one folder per subject
OUT = ROOT / "results"

# ---- settings (edit here) -------------------------------------------------------------------
settings = Settings(
    static_token="static",       # token that marks the static trial in the file name
    mb_prefix="MB_",             # prefix of the marker-based marker labels
    marker_cutoff_hz=20.0,       # low-pass of marker-based markers [Hz]
    ml_marker_cutoff_hz=None,    # low-pass of markerless markers [Hz], None = as delivered
    force_cutoff_hz=20.0,        # low-pass of force, centre of pressure, plate moment [Hz]
    force_threshold_n=20.0,      # contact threshold on the raw vertical force [N]
    min_contact_s=0.100,         # minimum contact duration [s]
    analysis_window_s=0.100,     # early-stance window after initial contact [s]
    massless_foot=True,          # as in the companion article
    substitution=True,           # marker substitution when both sources are present
)
# -----------------------------------------------------------------------------------------------

table = analyse_dataset(DATA, OUT, settings)
ok = table[table["error"].isna()] if "error" in table else table
if "peak_kam_nmkg" in ok and len(ok):
    summary = (ok.groupby(["variant", "stage"], sort=False)[["peak_kam_nmkg", "knee_frontal_ic_deg"]]
                 .mean().round(3))
    print(summary.to_string())
else:
    print("No trial could be analysed.")
print(f"\n{len(table) - len(ok)} rows with errors; see {OUT / 'log.txt'}")
print(f"Results written to {OUT}")
