"""Marker substitution.

Starting from the markerless (ML) marker set, groups of ML markers are replaced by their
marker-based (MB) counterparts. The change of an outcome towards the MB value shows how much
of the ML-MB difference the substituted markers explain. Substitutions apply to both legs.

Stage 1, anatomical landmarks (static trial only; the dynamic trials stay ML): malleoli (M),
femoral epicondyles (E), pelvis (P) and their combinations.
Stage 2, tracking markers (static and dynamic): shank cluster (T), thigh cluster (F), pelvis (P)
and combinations.
Stage 3, segment-wise (static and dynamic): proximal (epicondyles and thigh cluster), distal
(malleoli and shank cluster), both, both and pelvis.
Stage 4, foot (static and dynamic): foot markers (heel, 5th metatarsal head, big toe), and the
distal chain with the foot. With a massless foot (default) the knee outcomes do not depend on the
foot markers; with full inverse dynamics (``Settings.massless_foot = False``) they do.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .markers import GROUPS, LSTM_TO_MODEL
from .preprocessing import MarkerSet

ALL = frozenset(LSTM_TO_MODEL.values())


@dataclass(frozen=True)
class Variant:
    id: str
    stage: int                  # 0 = reference set (pure ML or pure MB)
    static: frozenset           # internal names taken from MB in the static trial
    dynamic: frozenset          # internal names taken from MB in the dynamic trials
    description: str


def _g(*names) -> frozenset:
    return frozenset().union(*(GROUPS[n] for n in names))


def standard_variants() -> list[Variant]:
    """Reference sets and the substitutions of the three stages."""
    v = [Variant("ML", 0, frozenset(), frozenset(), "all markers markerless"),
         Variant("MB", 0, ALL, ALL, "all markers marker-based")]
    land = {"M": "malleoli", "E": "epicondyles", "P": "pelvis"}
    for r in (1, 2, 3):
        for combo in itertools.combinations("MEP", r):
            names = tuple(land[c] for c in combo)
            v.append(Variant("S1_" + "".join(combo), 1, _g(*names), frozenset(),
                             "static landmarks: " + ", ".join(names)))
    for vid, names in (("S2_T", ("shank_cluster",)), ("S2_F", ("thigh_cluster",)),
                       ("S2_FT", ("thigh_cluster", "shank_cluster")),
                       ("S2_PT", ("pelvis", "shank_cluster")), ("S2_PF", ("pelvis", "thigh_cluster")),
                       ("S2_PFT", ("pelvis", "thigh_cluster", "shank_cluster"))):
        v.append(Variant(vid, 2, _g(*names), _g(*names), "tracking markers: " + ", ".join(names)))
    for vid, names in (("S3_prox", ("epicondyles", "thigh_cluster")),
                       ("S3_dist", ("malleoli", "shank_cluster")),
                       ("S3_both", ("epicondyles", "thigh_cluster", "malleoli", "shank_cluster")),
                       ("S3_both_P", ("epicondyles", "thigh_cluster", "malleoli", "shank_cluster", "pelvis"))):
        v.append(Variant(vid, 3, _g(*names), _g(*names), "segment-wise: " + ", ".join(names)))
    for vid, names in (("S4_foot", ("foot",)), ("S4_dist_foot", ("malleoli", "shank_cluster", "foot"))):
        v.append(Variant(vid, 4, _g(*names), _g(*names), "foot: " + ", ".join(names)))
    return v


def hybrid(ml: MarkerSet, mb: MarkerSet, take_from_mb: frozenset) -> MarkerSet:
    """ML marker set with the markers in *take_from_mb* replaced by their MB counterparts.

    Works for dynamic trajectories (n_frames, n_markers, 3) and static poses (n_markers, 3).
    A selected marker that the MB set lacks is dropped (as in a pure MB analysis)."""
    axis = ml.data.ndim - 2                    # marker axis
    labels, cols = [], []
    for j, lbl in enumerate(ml.labels):
        if lbl in take_from_mb:
            if lbl in mb.labels:
                labels.append(lbl)
                cols.append(np.take(mb.data, mb.labels.index(lbl), axis=axis))
        else:
            labels.append(lbl)
            cols.append(np.take(ml.data, j, axis=axis))
    for lbl in sorted(take_from_mb):           # MB markers without an ML counterpart
        if lbl not in labels and lbl in mb.labels:
            labels.append(lbl)
            cols.append(np.take(mb.data, mb.labels.index(lbl), axis=axis))
    return MarkerSet(data=np.stack(cols, axis=axis), labels=labels)


def build(variant: Variant, sets: dict[str, MarkerSet], static: bool) -> MarkerSet:
    """Marker set of *variant* from ``{"ML": ..., "MB": ...}`` (static or dynamic trial)."""
    if variant.id in ("ML", "MB"):
        return sets[variant.id]
    return hybrid(sets["ML"], sets["MB"], variant.static if static else variant.dynamic)
