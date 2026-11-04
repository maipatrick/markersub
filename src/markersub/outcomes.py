"""Discrete outcomes and curves.

Curves: ``stance_curves`` (per marker frame, IC to toe-off) and ``normalised_curves`` (IC = 0 %
to toe-off = 100 %, 101 points). Angle components: flexion, adduction, internal rotation (Cardan,
distal relative to proximal segment). Moment components: external flexion, abduction and
internal rotation moments, projected onto the joint coordinate system axes.

Knee outcomes in the early-stance window [IC, IC + analysis_window_s] (inclusive):

* ``peak_kam_nmkg``         peak external knee abduction moment [Nm/kg]
* ``peak_kfm_nmkg``         peak external knee flexion moment [Nm/kg]
* ``knee_frontal_ic_deg``   knee frontal-plane angle at IC [deg] (Cardan X; adduction positive,
                            abduction, i.e. valgus, negative)
* ``knee_frontal_min_deg``  minimum of the frontal-plane angle in the window [deg]
* ``knee_frontal_mean_deg`` mean of the frontal-plane angle in the window [deg]
* ``knee_flexion_ic_deg``   knee flexion angle at IC [deg]
"""
from __future__ import annotations

import numpy as np

from .preprocessing import LegEvents

JOINTS = ("Hip", "Knee", "Ankle")


def knee_outcomes(angles: dict, moments: dict, ev: LegEvents, rate: float, window_s: float) -> dict:
    side = "Left" if ev.leg == "left" else "Right"
    td = ev.td_frame
    w = slice(td, td + int(round(window_s * rate)) + 1)
    kam = np.asarray(moments["Knee"]["Abduction"], float)
    kfm = np.asarray(moments["Knee"]["Flexion"], float)
    x = np.asarray(angles["JOINT"]["deg"][f"{side}_Knee"]["X"], float)
    y = np.asarray(angles["JOINT"]["deg"][f"{side}_Knee"]["Y"], float)
    return {"peak_kam_nmkg": float(np.nanmax(kam[w])), "peak_kfm_nmkg": float(np.nanmax(kfm[w])),
            "knee_frontal_ic_deg": float(x[td]), "knee_frontal_min_deg": float(np.nanmin(x[w])),
            "knee_frontal_mean_deg": float(np.nanmean(x[w])), "knee_flexion_ic_deg": float(y[td])}


_ANGLE_AXES = (("Y", "flexion"), ("X", "adduction"), ("Z", "internal_rotation"))
_MOMENT_AXES = (("Flexion", "flexion"), ("Abduction", "abduction"), ("InternalRotation", "internal_rotation"))


def normalised_curves(angles: dict, moments: dict, ev: LegEvents, n_points: int = 101) -> list[dict]:
    """Joint angles [deg] and external moments [Nm/kg] of hip, knee and ankle, all three axes,
    time-normalised from IC (0 %) to toe-off (100 %) on *n_points* points (default 101).

    The signals are linearly interpolated between the marker frames of IC and toe-off."""
    side = "Left" if ev.leg == "left" else "Right"
    x_new = np.linspace(ev.td_frame, ev.to_frame, n_points)
    pct = np.linspace(0.0, 100.0, n_points)
    rows = []

    def add(joint, quantity, component, sig):
        sig = np.asarray(sig, float)
        if ev.to_frame >= len(sig):
            return
        frames = np.arange(len(sig), dtype=float)
        vals = np.interp(x_new, frames, sig)
        rows.extend({"joint": joint.lower(), "quantity": quantity, "component": component,
                     "percent_stance": float(p), "value": float(v)} for p, v in zip(pct, vals))

    for joint in JOINTS:
        ang = angles["JOINT"]["deg"].get(f"{side}_{joint}", {})
        for axis, name in _ANGLE_AXES:
            if axis in ang:
                add(joint, "angle_deg", name, ang[axis])
        mom = moments.get(joint, {})
        for key, name in _MOMENT_AXES:
            if key in mom:
                add(joint, "external_moment_nmkg", name, mom[key])
    return rows


def stance_curves(angles: dict, moments: dict, ev: LegEvents, rate: float) -> list[dict]:
    """Joint angles [deg] and external moments [Nm/kg] from IC to toe-off, long format."""
    side = "Left" if ev.leg == "left" else "Right"
    rows = []
    frames = range(ev.td_frame, ev.to_frame + 1)
    for joint in JOINTS:
        ang = angles["JOINT"]["deg"].get(f"{side}_{joint}", {})
        for axis, name in (("Y", "flexion"), ("X", "adduction"), ("Z", "internal_rotation")):
            if axis in ang:
                a = np.asarray(ang[axis], float)
                rows += [{"joint": joint.lower(), "quantity": "angle_deg", "component": name,
                          "frame_from_ic": k - ev.td_frame, "time_from_ic_s": (k - ev.td_frame) / rate,
                          "value": float(a[k])} for k in frames if k < len(a)]
        mom = moments.get(joint, {})
        for key, name in (("Flexion", "flexion"), ("Abduction", "abduction"),
                          ("InternalRotation", "internal_rotation")):
            if key in mom:
                m = np.asarray(mom[key], float)
                rows += [{"joint": joint.lower(), "quantity": "external_moment_nmkg", "component": name,
                          "frame_from_ic": k - ev.td_frame, "time_from_ic_s": (k - ev.td_frame) / rate,
                          "value": float(m[k])} for k in frames if k < len(m)]
    return rows
