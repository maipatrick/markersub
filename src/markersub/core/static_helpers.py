"""Helpers for static-trial preprocessing.

Heading alignment of the static trial to +X and foot length from the heel and
toe (or forefoot) markers. Marker arrays are (n_markers, 3, n_frames)
in mm.
"""

from __future__ import annotations
import numpy as np


def _rz_align_to_plus_x(markers_m3n: np.ndarray, labels: list[str]) -> np.ndarray:
    """Rotation about vertical Z that turns the static pelvis-forward direction (mid_ASIS - mid_PSIS,
    horizontal) onto +X. Identity if the pelvis markers are missing.
    """
    def _mean(name: str) -> np.ndarray:
        i = labels.index(name)
        return np.nanmean(markers_m3n[i, :3, :], axis=1)
    try:
        fwd = (_mean("SIAS_right") + _mean("SIAS_left")) / 2.0 \
            - (_mean("SIPS_right") + _mean("SIPS_left")) / 2.0
    except (ValueError, KeyError):
        return np.eye(3)
    fwd[2] = 0.0
    nrm = np.linalg.norm(fwd)
    if nrm < 1e-9:
        return np.eye(3)
    ang = -np.arctan2(fwd[1], fwd[0])          # rotate current facing -> +X
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotate_markers_m3n(R: np.ndarray, markers_m3n: np.ndarray) -> np.ndarray:
    """Apply R to every 3-vector of a (n_markers, 3, n_frames) array."""
    return np.einsum("ij,mjn->min", R, markers_m3n)




def _foot_length_mm(markers_raw: np.ndarray, labels: list[str]) -> float:
    """Mean foot length calc_back <-> toe (both feet).
    Marker sets without a dedicated toe marker fall back to the forefoot markers
    (midpoint of the medial and lateral MTP markers), so the length is still measured.
    """
    def _pos(name: str) -> np.ndarray:
        return markers_raw[labels.index(name), :3, :].mean(axis=1)

    def _toe(side: str) -> np.ndarray:
        for cand in (f"toe_{side}",):
            if cand in labels:
                return _pos(cand)
        # forefoot fallback: midpoint of the med/lat MTP markers (RIGHT: forfoot, LEFT: forefoot)
        med = "forfoot_med_right" if side == "right" else "forefoot_med_left"
        lat = "forfoot_lat_right" if side == "right" else "forefoot_lat_left"
        if med in labels and lat in labels:
            return (_pos(med) + _pos(lat)) / 2.0
        if med in labels:
            return _pos(med)
        raise KeyError(f"no toe/forefoot marker for {side}")

    lengths = []
    for side in ("right", "left"):
        try:
            lengths.append(float(np.linalg.norm(_toe(side) - _pos(f"calc_back_{side}"))))
        except (ValueError, KeyError):
            pass
    return float(np.mean(lengths)) if lengths else 250.0
