"""Inverse dynamics and external joint moments in the joint coordinate system.

Newton-Euler inverse dynamics of foot, shank and thigh (Hof 1992) with sex-specific segment
parameters (de Leva 1996). The net joint moment is re-expressed as the EXTERNAL moment on the
joint coordinate system axes (Grood and Suntay 1983): flexion, abduction, internal rotation.
Moments are normalised to body mass [Nm/kg] and returned per marker frame.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

from .core import inverse_dynamics as _idn
from .core import jcs_moments as _jm
from .preprocessing import PreparedTrial


def _anthropometry(mass_kg, height_mm, calib, leg, sex, massless_foot):
    """References: de Leva P (1996) J Biomech 29:1223-1230."""
    Leg = leg.capitalize()
    seg_lengths = {}
    for seg in ("Foot", "Shank", "Thigh"):
        sl = calib["refframe"]["Anatomical"].get(f"{Leg}{seg}", {}).get("SLength", 0)
        if sl and sl > 0:
            seg_lengths[seg] = float(sl)
    anthro = _idn.compute_anthropometry(mass_kg, height_mm, seg_lengths, 0 if sex == "female" else 1)
    if massless_foot:
        anthro["SegmentMass"]["Foot"] = 0.0
        anthro["MoI"]["Foot"] = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    return anthro


def external_jcs_moments(frames: dict, calib: dict, trial: PreparedTrial, leg: str,
                         mass_kg: float, height_m: float, sex: str,
                         massless_foot: bool = False) -> dict:
    """External hip, knee and ankle moments of one leg [Nm/kg] per marker frame.

    Returns ``{"Hip"|"Knee"|"Ankle": {"Flexion", "Abduction", "InternalRotation"}}``. The ground
    reaction force acts only between initial contact and toe-off of this leg.
    References: Hof AL (1992) J Biomech 25:1209-1211.; de Leva P (1996) J Biomech 29:1223-1230.; Grood ES, Suntay WJ (1983) J Biomech Eng 105:136-144.; Desroches G, Cheze L, Dumas R (2010) J Biomech Eng 132:114503.
    """
    ev = trial.events[leg]
    anat = frames["Anatomical"]
    anthro = _anthropometry(mass_kg, height_m * 1000.0, calib, leg, sex, massless_foot)
    jc_dyn = _idn.compute_dynamic_jc(anat, calib["refframe"], calib["jc"], leg=leg)
    ff = anat.get(f"{leg.capitalize()}Forefoot")
    if ff is not None:
        jc_dyn["MPJC"] = np.asarray(ff["O"], dtype=float)
    plate = trial.plates[ev.plate - 1]
    ratio = trial.analog_rate / trial.rate
    fp = {"force": np.asarray(plate["force"], float).copy(),
          "cop": np.asarray(plate["cop"], float).copy(),
          "moment": np.asarray(plate["moment"], float).copy(),
          "sampling_factor": int(round(ratio))}
    td_f, to_f = int(round(ev.td_frame * ratio)), int(round(ev.to_frame * ratio))
    n = fp["force"].shape[1]
    stance = np.zeros(n, dtype=bool)
    stance[max(0, td_f):min(n, to_f + 1)] = True
    fp["force"][:, ~stance] = 0.0
    fp["moment"][:, ~stance] = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        kinetics, _ = _idn.inverse_dynamics_hof(anat, fp, jc_dyn, anthro, leg=leg, fp_moment_unit="Nmm")
    jcs = _jm.external_jcs_from_kinetics(kinetics, anat, leg, n, _idn.resample_signal)
    n_kin = max(2, int(round(n / ratio)))
    x_force = np.arange(n, dtype=float)
    x_kin = np.minimum(np.arange(n_kin, dtype=float) * ratio, n - 1.0)   # marker frame k = sample k*ratio
    return {j: {k: np.interp(x_kin, x_force, np.asarray(v, float)) / mass_kg for k, v in c["proj"].items()}
            for j, c in jcs.items()}
