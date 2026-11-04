"""The example subject must reproduce the stored reference outcomes."""
from pathlib import Path

import numpy as np
import pandas as pd

from markersub import Settings, analyse_subject

ROOT = Path(__file__).resolve().parents[1]
COLS = ["peak_kam_nmkg", "peak_kfm_nmkg", "knee_frontal_ic_deg", "knee_frontal_min_deg",
        "knee_frontal_mean_deg", "knee_flexion_ic_deg"]


def test_example_subject_matches_reference():
    ref = pd.read_csv(ROOT / "tests" / "reference" / "S01_outcomes.csv")
    res = analyse_subject(ROOT / "example_data" / "S01", Settings(massless_foot=True), curve_variants=())
    out = res.outcomes
    assert "error" not in out or out["error"].isna().all()
    m = ref.merge(out, on=["trial", "variant", "leg"], suffixes=("_ref", ""))
    assert len(m) == len(ref) == len(out)
    for c in COLS:
        np.testing.assert_allclose(m[c], m[c + "_ref"], rtol=1e-7, atol=1e-9, err_msg=c)
