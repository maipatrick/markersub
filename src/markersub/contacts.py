"""Contact detection on the raw vertical ground reaction force.

A contact is a run of raw (unfiltered) vertical force above ``force_threshold_n`` that lasts at
least ``min_contact_s``. Initial contact (IC) is the first and toe-off the last supra-threshold
sample of that run. If a plate carries more than one contact the analysed one is chosen as:

1. plate already loaded at the start of the recording (standing on it, e.g. a countermovement
   jump): the last contact, i.e. the landing. If there is no later contact the rule is reported
   as ``loaded_at_start`` (initial contact not observed; check such trials);
2. one contact only: that contact;
3. several contacts: the one with heel and forefoot of the analysed leg over the plate at peak
   force (marker-based markers only, 10 mm tolerance); if this is not unique or no marker-based
   foot markers exist, the contact with the largest vertical impulse.

The vertical force must be positive upward (loading).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

PRELOAD_TOLERANCE_SAMPLES = 5
PLATE_MARGIN_MM = 10.0


@dataclass
class Contact:
    td_s: float            # initial contact [s] from the first sample of the file
    to_s: float            # toe-off [s]
    rule: str              # single | preloaded_landing | loaded_at_start | foot_on_plate | max_impulse
    n_contacts: int


def sustained_runs(above: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) of every run of True lasting at least *min_samples*."""
    runs, i, n = [], 0, len(above)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if j - i >= min_samples:
            runs.append((i, j - 1))
        i = j
    return runs


def _point_in_polygon(pt, poly) -> bool:
    x, y = float(pt[0]), float(pt[1])
    inside, n = False, len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _dist_to_segment(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = 0.0 if not np.any(ab) else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def point_on_plate(pt, corners_xy, margin_mm: float = PLATE_MARGIN_MM) -> bool:
    """True if *pt* (x, y) lies inside the plate outline or within *margin_mm* of its edge."""
    if _point_in_polygon(pt, corners_xy):
        return True
    n = len(corners_xy)
    return any(_dist_to_segment(pt, corners_xy[i], corners_xy[(i + 1) % n]) <= margin_mm for i in range(n))


def select_contact(fz: np.ndarray, analog_rate: float, threshold_n: float, min_contact_s: float,
                   foot_on_plate=None) -> Contact | None:
    """Analysed contact on one plate.

    fz            raw vertical force (N), positive = loading
    foot_on_plate optional callable(force_sample_index) -> True | False | None used by rule 3
    References: Tirosh O, Sparrow WA (2003) J Appl Biomech 19:178-184.
    """
    min_samples = int(round(min_contact_s * analog_rate))
    runs = sustained_runs(np.asarray(fz) > threshold_n, min_samples)
    if not runs:
        return None
    preloaded = runs[0][0] <= PRELOAD_TOLERANCE_SAMPLES
    if preloaded:
        k, rule = len(runs) - 1, ("loaded_at_start" if len(runs) == 1 else "preloaded_landing")
    elif len(runs) == 1:
        k, rule = 0, "single"
    else:
        hits = []
        if foot_on_plate is not None:
            for k_, (i, j) in enumerate(runs):
                pk = i + int(np.argmax(fz[i:j + 1]))
                if foot_on_plate(pk) is True:
                    hits.append(k_)
        if len(hits) == 1:
            k, rule = hits[0], "foot_on_plate"
        else:
            t = np.arange(len(fz)) / analog_rate
            imp = [float(_trapezoid(fz[i:j + 1], t[i:j + 1])) if j > i else 0.0 for i, j in runs]
            k, rule = int(np.argmax(imp)), "max_impulse"
    i, j = runs[k]
    return Contact(td_s=i / analog_rate, to_s=j / analog_rate, rule=rule, n_contacts=len(runs))
