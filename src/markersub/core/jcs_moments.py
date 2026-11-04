"""External joint moments expressed in the joint coordinate system (JCS).

JCS axes after Grood and Suntay (1983, J Biomech Eng): e1 is the proximal
segment's mediolateral axis, e3 the distal segment's long axis and e2 = e1 x e3
(floating axis, normalised). Moments are given both as orthogonal projections onto the
three axes and as the non-orthogonal decomposition M = c1 e1 + c2 e2 + c3 e3.
"""

from __future__ import annotations
import numpy as np


JCS_JOINTS = {            # joint: (proximal segment, distal segment, sagittal sign of e1 component)
    "Hip":   ("Pelvis",   "Thigh",    -1.0),
    "Knee":  ("Thigh",    "Shank",    +1.0),
    "Ankle": ("Shank",    "Foot",     -1.0),
    "MPJ":   ("Rearfoot", "Forefoot", -1.0),
}


def jcs_axes(R_prox: np.ndarray, R_dist: np.ndarray):
    """(3,3,n) proximal and distal anatomical rotations -> e1, e2, e3 as (3,n) unit vectors.
    References: Grood ES, Suntay WJ (1983) J Biomech Eng 105:136-144.
    """
    e1 = R_prox[:, 1, :]
    e3 = R_dist[:, 2, :]
    e2 = np.cross(e1.T, e3.T).T
    e2 = e2 / np.linalg.norm(e2, axis=0, keepdims=True)
    return e1, e2, e3


def project(M: np.ndarray, e1, e2, e3) -> np.ndarray:
    """Orthogonal projections onto the JCS axes. M (3,n) -> (3,n) rows = (e1, e2, e3).
    References: Desroches G, Cheze L, Dumas R (2010) J Biomech Eng 132:114503.
    """
    return np.stack([np.sum(M * e1, 0), np.sum(M * e2, 0), np.sum(M * e3, 0)])


def decompose(M: np.ndarray, e1, e2, e3) -> np.ndarray:
    """Non-orthogonal decomposition M = c1 e1 + c2 e2 + c3 e3 (mixed product / Cramer).
    References: Desroches G, Cheze L, Dumas R (2010) J Biomech Eng 132:114503.
    """
    vol = np.sum(e1 * np.cross(e2.T, e3.T).T, 0)
    c1 = np.sum(M * np.cross(e2.T, e3.T).T, 0) / vol
    c2 = np.sum(e1 * np.cross(M.T, e3.T).T, 0) / vol
    c3 = np.sum(e1 * np.cross(e2.T, M.T).T, 0) / vol
    return np.stack([c1, c2, c3])


def to_clinical(raw: np.ndarray, joint: str, leg: str) -> dict:
    """Raw JCS components (rows e1, e2, e3) of the EXTERNAL moment -> clinical named components."""
    s_flex = JCS_JOINTS[joint][2]
    s_lr = -1.0 if leg.lower().startswith("r") else 1.0
    out = {"Flexion": s_flex * raw[0], "Abduction": s_lr * raw[1], "InternalRotation": -s_lr * raw[2]}
    if joint == "MPJ":                                 # single-DOF joint: only the flexion component is kept
        out["Abduction"] = np.zeros_like(raw[1]); out["InternalRotation"] = np.zeros_like(raw[2])
    return out


def external_jcs_from_kinetics(kinetics: dict, frames_anat: dict, leg: str, n_fp: int, resample_signal):
    """Per joint {'proj': {comp: (n_fp,)}, 'decomp': {...}} in N*m (external, clinical signs).

    `kinetics` is the KINETICS dict returned by inverse_dynamics_hof (its global 'Moment' is
    the internal moment). `resample_signal` is inverse_dynamics.resample_signal, i.e. the
    same resampling as in inverse_dynamics_hof.
    References: Grood ES, Suntay WJ (1983) J Biomech Eng 105:136-144.; Desroches G, Cheze L, Dumas R (2010) J Biomech Eng 132:114503.
    """
    S = "Right" if leg.lower().startswith("r") else "Left"
    res = {}
    for joint, (prox, dist, _s) in JCS_JOINTS.items():
        k = kinetics.get(joint)
        if not isinstance(k, dict) or "Moment" not in k:
            continue
        if f"{S}{prox}" not in frames_anat or f"{S}{dist}" not in frames_anat:
            continue
        Rp = resample_signal(frames_anat[f"{S}{prox}"]["R"], n_fp, method="polyphase", project_rotation=True)
        Rd = resample_signal(frames_anat[f"{S}{dist}"]["R"], n_fp, method="polyphase", project_rotation=True)
        e1, e2, e3 = jcs_axes(Rp, Rd)
        M_ext = -np.asarray(k["Moment"], dtype=float)
        res[joint] = {"proj":   to_clinical(project(M_ext, e1, e2, e3), joint, leg),
                      "decomp": to_clinical(decompose(M_ext, e1, e2, e3), joint, leg)}
    return res
