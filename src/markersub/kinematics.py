"""Static calibration and dynamic reconstruction (segment-wise kinematics).

Static: technical-to-anatomical calibration of every segment from the time-averaged standing
posture. Dynamic: SVD rigid-body fit of the technical frames (Soederkvist and Wedin 1993), the
calibrated anatomical frames and Cardan joint angles (flexion Y, adduction X, rotation Z).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

from .core import dynamic_reconstruction as _dr
from .core import joint_centers as _jc
from .core import static_calibration as _sc
from .core.static_helpers import _foot_length_mm, _rotate_markers_m3n, _rz_align_to_plus_x
from .markers import joint_pairs, segments_config

STATIC_FRAMES = 64      # the mean posture is repeated to a short constant trajectory ...
STATIC_RATE_HZ = 100.0  # ... with a nominal rate (a constant signal is unaffected by filtering)


def calibrate(pose: np.ndarray, labels: list[str], sex: str, cutoff_hz: float) -> dict:
    """Static calibration from a time-averaged posture.

    pose    (n_markers, 3) [mm], internal model names in *labels*
    sex     "female" or "male" (sex-specific segment parameters, de Leva 1996)
    Returns the calibration dictionary used by :func:`reconstruct` and the inverse dynamics.
    References: Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Clin Biomech 10:171-178.; Harrington ME et al. (2007) J Biomech 40:595-602.; Wu G et al. (2002) J Biomech 35:543-548.; de Leva P (1996) J Biomech 29:1223-1230.
    """
    markers = np.repeat(np.asarray(pose, float)[:, :, None], STATIC_FRAMES, axis=2)   # (m, 3, n)
    labels = list(labels)
    align = _rz_align_to_plus_x(markers, labels)     # subject faces +X in the calibration frame
    markers = _rotate_markers_m3n(align, markers)
    foot_len = _foot_length_mm(markers, labels)
    jc = _jc.compute_all_joint_centers(markers, labels, frame=None)
    with contextlib.redirect_stdout(io.StringIO()):
        refframe, filt, _tech, _anat = _sc.calibrate_static_lower_limb(
            markers_raw=markers, labels=labels, freq_kinematics=STATIC_RATE_HZ,
            cut_off_markers=float(cutoff_hz),
            anthro={"FootLength": foot_len, "gender": "female" if sex == "female" else "male"},
            static_joint_center_source="filtered", static_hip_center_source="harrington")
    return {"refframe": refframe, "jc": jc, "foot_length_mm": foot_len, "labels": labels,
            "static_markers_filt": filt.get("Filt"), "align_R": align}


def reconstruct(markers: np.ndarray, labels: list[str], rate: float, calib: dict,
                cutoff_hz: float) -> tuple[dict, dict]:
    """Segment frames and joint angles of one dynamic trial.

    markers   (n_frames, n_markers, 3) [mm], lab frame, internal model names
    cutoff_hz low-pass applied to the segment linear velocities before differentiation
    Returns (frames, angles); angles["JOINT"]["deg"]["Left_Knee"]["X"|"Y"|"Z"].
    ``frames["warnings"]`` lists markers that were missing and reconstructed from the static
    geometry (rigid-body assumption, Cappozzo et al. 1995) and segments that could not be built.
    References: Soederkvist I, Wedin PA (1993) J Biomech 26:1473-1477.; Arun KS, Huang TS, Blostein SD (1987) IEEE Trans Pattern Anal Mach Intell 9:698-700.; Grood ES, Suntay WJ (1983) J Biomech Eng 105:136-144.
    """
    trial = {"marker_labels": list(labels),
             "marker_data": np.transpose(np.asarray(markers, float), (1, 2, 0)),
             "marker_rate_hz": float(rate)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        frames, angles = _dr.reconstruct_kinematics(
            trial, calib["refframe"], segments_config=segments_config(), joint_pairs=joint_pairs(),
            marker_cutoff_hz=float(cutoff_hz), static_markers_filt=calib.get("static_markers_filt"),
            ref_to_static=False)
    # clinical sign convention: hip flexion positive
    for jn in ("Right_Hip", "Left_Hip"):
        for unit in ("deg", "rad"):
            blk = angles.get("JOINT", {}).get(unit, {}).get(jn)
            if isinstance(blk, dict) and "Y" in blk:
                blk["Y"] = -np.asarray(blk["Y"])
    frames["warnings"] = core_warnings(buf.getvalue())
    return frames, angles


def core_warnings(text: str) -> list[str]:
    """Lines of the core output that report missing or reconstructed markers or failed segments."""
    keys = ("WARNING", "FAILED", "rebuilt", "missing", "filled", "skipped")
    return [ln.strip() for ln in text.splitlines() if any(k in ln for k in keys)]
