"""Time-normalised curves: 101 points from IC to toe-off, all axes."""
from pathlib import Path

import numpy as np

from markersub import Settings, analyse_subject, standard_variants

ROOT = Path(__file__).resolve().parents[1]


def test_normalised_curves_shape_and_endpoints():
    variants = [v for v in standard_variants() if v.id == "MB"]
    res = analyse_subject(ROOT / "example_data" / "S01", Settings(), variants=variants)
    n = res.curves_normalised
    raw = res.curves
    groups = n.groupby(["trial", "joint", "quantity", "component"])
    raw_idx = raw.set_index(["trial", "joint", "quantity", "component"]).sort_index()
    assert len(groups) == 3 * 3 * 2 * 3                    # trials x joints x (angle, moment) x axes
    assert all(len(g) == 101 for _, g in groups)
    for key, g in groups:
        r = raw_idx.loc[key]
        assert np.isclose(g["value"].iloc[0], r["value"].iloc[0])      # IC
        assert np.isclose(g["value"].iloc[-1], r["value"].iloc[-1])    # toe-off
