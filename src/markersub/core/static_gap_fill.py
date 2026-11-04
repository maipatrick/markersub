"""Reconstruction of missing cluster markers from the static reference geometry.

Missing markers are placed rigidly from the visible, geometry-consistent markers of
the same cluster (calibrated anatomical systems technique, CAST; Cappozzo et al.
1995, Clin Biomech). The rigid fit follows Soederkvist and Wedin (1993, J Biomech)
and Arun et al. (1987, IEEE Trans PAMI). Positions are in mm.
"""

import numpy as np


# --- Static-referenced CAST reconstruction ---
def _soderkvist_fit(A, B):
    """Least-squares rigid transform mapping static points A (n,3) -> observed B (n,3).
    Soederkvist and Wedin (1993), Arun et al. (1987). Returns (R, t) with B ~ A @ R.T + t.
    """
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _static_consistent_subset(vis, obs, ref, tol):
    """Keep visible markers whose pairwise distances agree with the static rigid geometry
    (each must agree with >=2 others within `tol` mm). This rejects swapped or ghost-labelled
    markers before the rigid fit, because a swap produces inconsistent inter-marker distances.
    """
    good = []
    for i in vis:
        agree = sum(1 for j in vis if j != i and
                    abs(np.linalg.norm(obs[i] - obs[j]) - np.linalg.norm(ref[i] - ref[j])) < tol)
        if agree >= 2:
            good.append(i)
    return good


def fill_missing_from_static(marker_data, marker_labels, segments_config, static_ref, tol=12.0):
    """Reconstruct missing cluster markers from the static reference geometry (CAST).

    For each segment and frame, a rigid transform is fitted to the >=3 visible cluster markers
    that are consistent with the static geometry (swapped or ghost frames are gated out), and
    each missing marker is placed at its static-relative position. The technical frame is then
    fitted over the full, over-determined cluster instead of the minimum of 3 markers. Only
    missing frames are written back into `marker_data` (observed values are never changed),
    so the function is a no-op on fully tracked data.

    marker_data  : (n_markers, 3, n_frames) lab-frame mm, modified IN PLACE.
    static_ref   : {label: (3,)} static reference position per marker (e.g. static-trial mean).
    Returns provenance {segment: {marker: {'frames_filled': k, 'conf_mm': median residual}}}.
    References: Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Clin Biomech 10:171-178.; Soederkvist I, Wedin PA (1993) J Biomech 26:1473-1477.
    """
    n_frames = marker_data.shape[2]
    label_to_idx = {str(l): i for i, l in enumerate(marker_labels)}
    prov = {}
    for seg_name, cfg in segments_config.items():
        names = [m for m in cfg.get('gap_fill_markers', cfg.get('markers', []))
                 if m in label_to_idx and m in static_ref]
        if len(names) < 4:          # need >=3 references to reconstruct a 4th
            continue
        idx = {m: label_to_idx[m] for m in names}
        fin = {m: np.all(np.isfinite(marker_data[idx[m], :, :]), axis=0) for m in names}  # (n,)
        need = [m for m in names if not fin[m].all()]
        if not need:
            continue
        for tgt in need:
            filled, resid = 0, []
            for f in range(n_frames):
                if fin[tgt][f]:
                    continue
                vis = [m for m in names if m != tgt and fin[m][f]]
                if len(vis) < 3:
                    continue
                obs = {m: marker_data[idx[m], :, f] for m in vis}
                use = _static_consistent_subset(vis, obs, static_ref, tol)
                if len(use) < 3:
                    continue
                A = np.array([static_ref[m] for m in use])
                B = np.array([obs[m] for m in use])
                R, t = _soderkvist_fit(A, B)
                marker_data[idx[tgt], :, f] = static_ref[tgt] @ R.T + t
                filled += 1
                resid.append(np.median([np.linalg.norm((static_ref[m] @ R.T + t) - obs[m]) for m in use]))
            if filled:
                prov.setdefault(seg_name, {})[tgt] = {
                    'frames_filled': int(filled),
                    'conf_mm': round(float(np.median(resid)), 1) if resid else None,
                }
    return prov
