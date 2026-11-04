"""Marker naming and the internal lower-limb segment model.

Markerless (ML) markers carry the names produced by the OpenCap marker augmenter
("study" markers, Uhlrich et al. 2023, PLoS Comput Biol), e.g. ``L_knee_study``.
Marker-based (MB) markers carry the same names with a prefix (default ``MB_``), e.g.
``MB_L_knee_study``. Both sources are mapped onto the same internal segment model, so each ML
marker and its MB counterpart describe the same landmark and can be exchanged one by one.

Segment model (both legs): pelvis (left and right ASIS and PSIS), thigh (3 tracking markers),
shank (3 tracking markers), foot (heel, 5th metatarsal head, big toe).

Foot: the marker augmenter has no marker that matches a marker on the big toe or on the 1st
metatarsal head, so the front point of the foot is the raw big-toe keypoint of the pose estimator
(``LBigToe`` / ``RBigToe``) and its marker-based counterpart is a marker on the big toe
(``MB_LBigToe`` / ``MB_RBigToe``). Heel (``*_calc_study``) and 5th metatarsal head
(``*_5meta_study``) come from the augmenter. Internally the front point occupies the slot named
``forefoot_med_*`` of the segment model. Anatomical landmarks:
lateral and medial femoral epicondyles and malleoli. Hip joint centre after Harrington et al.
(2007), knee and ankle joint centres at the epicondyle and malleolus midpoints.
"""
from __future__ import annotations

# marker name (markerless convention) -> internal model name. Tracking markers of thigh and shank
# (thigh1..3, sh1..3) are paired by their role in the cluster, not by location: marker-based
# tracking markers sit wherever they were attached. Anatomical landmarks are paired by location.
LSTM_TO_MODEL: dict[str, str] = {
    # pelvis
    "L.ASIS_study": "SIAS_left", "r.ASIS_study": "SIAS_right",
    "L.PSIS_study": "SIPS_left", "r.PSIS_study": "SIPS_right",
    # right leg
    "r_thigh1_study": "cluster_femur_right_1", "r_thigh2_study": "cluster_femur_right_2",
    "r_thigh3_study": "cluster_femur_right_3",
    "r_knee_study": "epi_lat_right", "r_mknee_study": "epi_med_right",
    "r_sh1_study": "cluster_tibia_right_1", "r_sh2_study": "cluster_tibia_right_2",
    "r_sh3_study": "cluster_tibia_right_3",
    "r_ankle_study": "mal_lat_right", "r_mankle_study": "mal_med_right",
    "r_calc_study": "calc_back_right", "r_5meta_study": "forfoot_lat_right",
    "RBigToe": "forfoot_med_right",     # big toe, raw keypoint (front point of the foot frame)
    # left leg
    "L_thigh1_study": "cluster_femur_left_1", "L_thigh2_study": "cluster_femur_left_2",
    "L_thigh3_study": "cluster_femur_left_3",
    "L_knee_study": "epi_lat_left", "L_mknee_study": "epi_med_left",
    "L_sh1_study": "cluster_tibia_left_1", "L_sh2_study": "cluster_tibia_left_2",
    "L_sh3_study": "cluster_tibia_left_3",
    "L_ankle_study": "mal_lat_left", "L_mankle_study": "mal_med_left",
    "L_calc_study": "calc_back_left", "L_5meta_study": "forefoot_lat_left",
    "LBigToe": "forefoot_med_left",     # big toe, raw keypoint (front point of the foot frame)
}
MODEL_TO_LSTM: dict[str, str] = {v: k for k, v in LSTM_TO_MODEL.items()}
REQUIRED_MARKERS: tuple[str, ...] = tuple(LSTM_TO_MODEL)

_SIDES = ("left", "right")

# substitution groups (internal names, both legs)
GROUPS: dict[str, tuple[str, ...]] = {
    "pelvis": ("SIAS_left", "SIAS_right", "SIPS_left", "SIPS_right"),
    "thigh_cluster": tuple(f"cluster_femur_{s}_{i}" for s in _SIDES for i in (1, 2, 3)),
    "shank_cluster": tuple(f"cluster_tibia_{s}_{i}" for s in _SIDES for i in (1, 2, 3)),
    "epicondyles": tuple(f"epi_{k}_{s}" for s in _SIDES for k in ("lat", "med")),
    "malleoli": tuple(f"mal_{k}_{s}" for s in _SIDES for k in ("lat", "med")),
    "foot": ("calc_back_left", "forefoot_lat_left", "forefoot_med_left",
             "calc_back_right", "forfoot_lat_right", "forfoot_med_right"),
}


def segments_config() -> dict:
    """Technical-frame markers per segment (input of the dynamic reconstruction)."""
    cfg = {}
    for side, Side in (("right", "Right"), ("left", "Left")):
        ff = "forfoot" if side == "right" else "forefoot"
        cfg[f"{Side}Pelvis"] = {"markers": ["SIAS_right", "SIAS_left", "SIPS_right", "SIPS_left"],
                                "n_markers": 4, "ref_key": f"{Side}Pelvis"}
        cfg[f"{Side}Thigh"] = {"markers": [f"cluster_femur_{side}_{i}" for i in (1, 2, 3)],
                               "n_markers": 3, "ref_key": f"{Side}Thigh"}
        cfg[f"{Side}Shank"] = {"markers": [f"cluster_tibia_{side}_{i}" for i in (1, 2, 3)],
                               "n_markers": 3, "ref_key": f"{Side}Shank"}
        cfg[f"{Side}Foot"] = {"markers": [f"calc_back_{side}", f"{ff}_lat_{side}", f"{ff}_med_{side}"],
                              "n_markers": 3, "ref_key": f"{Side}Foot"}
    return cfg


def joint_pairs() -> dict:
    """Joint = (proximal segment, distal segment)."""
    return {
        "Right_Hip": ("RightPelvis", "RightThigh"), "Right_Knee": ("RightThigh", "RightShank"),
        "Right_Ankle": ("RightShank", "RightFoot"),
        "Left_Hip": ("LeftPelvis", "LeftThigh"), "Left_Knee": ("LeftThigh", "LeftShank"),
        "Left_Ankle": ("LeftShank", "LeftFoot"),
    }


def split_sources(labels: list[str], mb_prefix: str = "MB_") -> dict[str, list[tuple[int, str]]]:
    """Sort c3d labels into the two marker sources.

    Returns ``{"MB": [(column, model_name), ...], "ML": [...]}``. Labels that are not part of
    the segment model (e.g. raw keypoints, upper-body markers) are ignored."""
    out: dict[str, list[tuple[int, str]]] = {"MB": [], "ML": []}
    for i, raw in enumerate(labels):
        lbl = raw.strip()
        if lbl.startswith(mb_prefix) and lbl[len(mb_prefix):] in LSTM_TO_MODEL:
            out["MB"].append((i, LSTM_TO_MODEL[lbl[len(mb_prefix):]]))
        elif lbl in LSTM_TO_MODEL:
            out["ML"].append((i, LSTM_TO_MODEL[lbl]))
    return out
