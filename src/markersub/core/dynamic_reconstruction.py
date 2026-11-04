"""Dynamic segment kinematics from marker trajectories.

Per segment, the static cluster geometry is fitted to the dynamic markers by SVD
rigid-body matching (Arun et al. 1987, IEEE Trans PAMI; Soederkvist and Wedin 1993,
J Biomech), the technical frame is built from the fitted markers and the
anatomical frame follows from the static technical-to-anatomical calibration.
Segment and joint angles are YXZ Cardan angles; angular velocity is taken from
the skew-symmetric matrix R_dot R^T and derivatives use 3-point central differences.
Marker positions are in mm, linear velocities in m/s, angles in rad and deg.
"""

import numpy as np
from scipy.linalg import svd
from .filters import butterworth_lowpass
from .segment_config import (
    get_default_joint_pairs as _get_default_joint_pairs_shared,
    get_default_segments_config as _get_default_segments_config_shared,
)


# Unwrap the Cardan angle series so that a joint near the arctan2 +/-180 deg branch
# (e.g. a composite-foot ankle angle in a jump) does not show spurious ~360 deg jumps.
# No-op for signals that never wrap (knee and hip angles stay far from +/-180 deg).
# Set False to obtain the raw wrapped angles.
_UNWRAP = True


# Optional low-pass of the angular velocity before differentiating to angular acceleration,
# analogous to the linear velocity being filtered before LinACC. Off by default; switching it
# on reduces ringing in the angular acceleration on low-frame-rate data. Uses the same
# cutoff as the linear path (linv_cutoff_hz).
_FILTER_ANGVEL = False


# A jump larger than this between two consecutive frames (after np.unwrap, which leaves
# jumps of at most pi) starts a new segment for the branch check in _recentre_segments.
# 90 deg/frame is 18000 deg/s at 200 Hz and 2700 deg/s at 30 Hz, far beyond any joint
# or segment angular velocity.
_SEGMENT_JUMP_RAD = 0.5 * np.pi


def _recentre_segments(v, flip_guard=True):
    """Re-centre one unwrapped Cardan series onto the branch around zero.

    1. |median| of the whole series <= 1.5*pi: returned unchanged.
    2. Otherwise the whole series is shifted by 2*pi*round(median/2*pi).
    3. Segment guard (only with flip_guard, i.e. joint angles): the series is split at
       frame-to-frame jumps > _SEGMENT_JUMP_RAD. A segment that the shift of step 2 pushes
       beyond +/-1.5*pi was already on the right branch (typically the valid frames before
       a partial frame flip); it is re-centred on its own median instead
       (2*pi*round(own median/2*pi) applied to the unshifted values, usually a shift of 0).
    Segments that step 2 leaves within +/-1.5*pi keep the step 2 result.
    """
    v = np.asarray(v)
    if v.shape[0] == 0 or not np.isfinite(v).any():
        return v
    med = np.nanmedian(v)
    if abs(med) <= 1.5 * np.pi:
        return v
    out = v - 2.0 * np.pi * np.round(med / (2.0 * np.pi))
    if not flip_guard:
        return out
    with np.errstate(invalid="ignore"):
        jump = np.abs(np.diff(v)) > _SEGMENT_JUMP_RAD          # NaN differences never cut
    bounds = [0] + (np.flatnonzero(jump) + 1).tolist() + [v.shape[0]]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = out[a:b]
        if not np.isfinite(seg).any() or abs(np.nanmedian(seg)) <= 1.5 * np.pi:
            continue
        raw = v[a:b]
        out[a:b] = raw - 2.0 * np.pi * np.round(np.nanmedian(raw) / (2.0 * np.pi))
    return out


def svdopti(trial_raw, ref_static):
    """Optimize marker trajectories to comply with the rigid-body assumption.

    Uses SVD-based point cloud matching (Arun et al. 1987, Soederkvist and Wedin 1993)
    to find the optimal rotation and translation for each frame.

    Parameters
    ----------
    trial_raw : ndarray
        Dynamic trial markers, shape (3, n_markers, n_frames)
    ref_static : ndarray
        Static reference markers, shape (3, n_markers) or (3, n_markers, n_frames)
        If 2D, replicated to n_frames

    Returns
    -------
    trial_opti : ndarray
        Optimized markers, shape (3, n_markers, n_frames)
    """
    
    if ref_static.ndim == 2:
        ref_static = np.tile(ref_static[:, :, np.newaxis], (1, 1, trial_raw.shape[2]))
    
    n_frames = trial_raw.shape[2]
    n_markers = trial_raw.shape[1]
    trial_opti = np.zeros_like(trial_raw)
    
    # Compute barycenters for trial
    o_trial = np.mean(trial_raw, axis=1, keepdims=True)  # shape (3, 1, n_frames)
    tech_trial = trial_raw - o_trial
    
    # Compute barycenters and mean for ref
    o_ref = np.mean(ref_static, axis=1, keepdims=True)  # shape (3, 1, n_frames)
    tech_ref_h = ref_static - o_ref
    o_ref_mean = np.mean(o_ref, axis=2)  # shape (3, 1)
    tech_ref_mean = np.mean(tech_ref_h, axis=2)  # shape (3, n_markers)
    
    # Cross-dispersion matrix and SVD per frame
    R_frames = np.zeros((3, 3, n_frames))
    d_frames = np.zeros((3, n_frames))
    
    for i in range(n_frames):
        # G = tech_trial * tech_ref^T
        G = tech_trial[:, :, i] @ tech_ref_mean.T
        
        # SVD decomposition
        U, _, Vt = svd(G)
        V = Vt.T
        
        # Ensure proper rotation (det(R) = +1, not -1)
        v_diag = np.array([1, 1, np.sign(np.linalg.det(U @ V.T))])
        R_frames[:, :, i] = U @ np.diag(v_diag) @ V.T
        
        # Translation
        d_frames[:, i] = o_trial[:, 0, i] - R_frames[:, :, i] @ o_ref_mean[:, 0]
        
        # Reconstruct optimized markers
        for j in range(n_markers):
            trial_opti[:, j, i] = R_frames[:, :, i] @ ref_static[:, j, i] + d_frames[:, i]
    
    return trial_opti


def stack_markers(*args):
    """Stack marker trajectories into (3, n_markers, n_frames) format.

    Parameters
    ----------
    *args : ndarray
        Variable number of marker arrays, each shape (3, n_frames)
        Can be 3 markers or 4 markers

    Returns
    -------
    Y : ndarray
        Stacked array, shape (3, len(args), n_frames)
    """
    n_markers = len(args)
    n_frames = args[0].shape[1]
    Y = np.zeros((3, n_markers, n_frames))
    
    for j, marker in enumerate(args):
        Y[:, j, :] = marker
    
    return Y


def unstack_markers(V):
    """Unpack stacked markers from (3, n_markers, n_frames) format.

    Parameters
    ----------
    V : ndarray
        Stacked array, shape (3, n_markers, n_frames)

    Returns
    -------
    Y1, Y2, Y3, Y4 : ndarray
        Unpacked marker arrays, shape (3, n_frames)
        (Y4 only if n_markers >= 4, otherwise zero-filled)
    """
    n_markers = V.shape[1]
    n_frames = V.shape[2]
    
    Y1 = V[:, 0, :] if n_markers >= 1 else np.zeros((3, n_frames))
    Y2 = V[:, 1, :] if n_markers >= 2 else np.zeros((3, n_frames))
    Y3 = V[:, 2, :] if n_markers >= 3 else np.zeros((3, n_frames))
    Y4 = V[:, 3, :] if n_markers >= 4 else np.zeros((3, n_frames))
    
    return Y1, Y2, Y3, Y4


def technical_frame_from_points(P1, P2, P3, P4=None):
    """Determine the technical coordinate system from 3 or 4 markers.

    Parameters
    ----------
    P1, P2, P3 : ndarray
        Marker positions, shape (3, n_frames)
    P4 : ndarray, optional
        Fourth marker (for 4-marker clusters)

    Returns
    -------
    O : ndarray
        Frame origin, shape (3, n_frames)
    R : ndarray
        Rotation matrices, shape (3, 3, n_frames)
    References: Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Clin Biomech 10:171-178.
    """
    n_frames = P1.shape[1]
    O = np.zeros((3, n_frames))
    R = np.zeros((3, 3, n_frames))
    
    if P4 is None:
        # 3-marker case
        for i in range(n_frames):
            O[:, i] = (P1[:, i] + P2[:, i] + P3[:, i]) / 3.0
        
        for i in range(n_frames):
            X = P2[:, i] - P1[:, i]
            Help1 = P3[:, i] - P1[:, i]
            Z = np.cross(X, Help1)
            Y = np.cross(Z, X)
            
            X = X / np.linalg.norm(X)
            Y = Y / np.linalg.norm(Y)
            Z = Z / np.linalg.norm(Z)
            
            R[:, 0, i] = X
            R[:, 1, i] = Y
            R[:, 2, i] = Z
    else:
        # 4-marker case
        for i in range(n_frames):
            O[:, i] = (P1[:, i] + P2[:, i] + P3[:, i] + P4[:, i]) / 4.0
        
        for i in range(n_frames):
            Z = ((P1[:, i] + P2[:, i]) / 2.0) - O[:, i]
            Help1 = ((P2[:, i] + P4[:, i]) / 2.0) - O[:, i]
            X = np.cross(Help1, Z)
            Y = np.cross(Z, X)
            
            X = X / np.linalg.norm(X)
            Y = Y / np.linalg.norm(Y)
            Z = Z / np.linalg.norm(Z)
            
            R[:, 0, i] = X
            R[:, 1, i] = Y
            R[:, 2, i] = Z
    
    return O, R


def getderivate_m3point(M):
    """Numerical differentiation with the 3-point method.

    Uses the per-sample central difference (f[i+1] - f[i-1]) / 2; divide by dt for a
    time derivative. The first and last samples copy their inner neighbours.

    Parameters
    ----------
    M : ndarray
        Input matrix, shape (n, m) or (n, m, k)

    Returns
    -------
    Matrix : ndarray
        Derivative, same shape as M
    References: Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    Matrix = np.zeros_like(M, dtype=float)
    
    if M.ndim == 2:
        # Vector case
        for j in range(1, M.shape[1] - 1):
            Matrix[:, j] = (M[:, j + 1] - M[:, j - 1]) / 2.0
        Matrix[:, 0] = Matrix[:, 1]    # boundary: first sample copies its neighbour (last sample likewise)
        Matrix[:, -1] = Matrix[:, -2]
    
    elif M.ndim == 3:
        # Matrix case (for rotation matrices)
        for i in range(1, M.shape[2] - 1):
            Matrix[:, :, i] = (M[:, :, i + 1] - M[:, :, i - 1]) / 2.0
        Matrix[:, :, 0] = Matrix[:, :, 1]    # boundary: first sample copies its neighbour (last sample likewise)
        Matrix[:, :, -1] = Matrix[:, :, -2]
    
    return Matrix


def gettheta(R_dot, R):
    """Determine the skew-symmetric matrix Theta from the rotation derivative.

    Theta = R_dot * R^T
    From Theta, the angular velocity components are:
        omega_x = Theta[2, 1]
        omega_y = Theta[0, 2]
        omega_z = Theta[1, 0]

    Parameters
    ----------
    R_dot : ndarray
        Rotation matrix derivative, shape (3, 3, n_frames)
    R : ndarray
        Rotation matrices, shape (3, 3, n_frames)

    Returns
    -------
    Theta : ndarray
        Skew-symmetric matrix, shape (3, 3, n_frames)
    """
    n_frames = R_dot.shape[2]
    Theta = np.zeros((3, 3, n_frames))
    
    for i in range(n_frames):
        Theta[:, :, i] = R_dot[:, :, i] @ R[:, :, i].T
    
    return Theta


def _correct_left_side_angles(angles_xyz):
    """Correct joint angles for the left-side coordinate system reflection.

    Because the left side is mirrored to match the right side's coordinate system,
    the X (adduction/abduction) and Z (rotation) angles require a sign correction for
    the left side to be directly comparable with the right side.

    Y (flexion/extension) remains unchanged because flexion is symmetric about
    the sagittal plane.

    Parameters
    ----------
    angles_xyz : dict
        Dictionary with keys 'X', 'Y', 'Z' containing angle arrays

    Returns
    -------
    dict
        Corrected angles with X and Z negated
    """
    return {
        'X': -angles_xyz['X'],
        'Y': angles_xyz['Y'],
        'Z': -angles_xyz['Z']
    }


def cardan_angles(R, flip_guard=False):
    """Extract Cardan angles from rotation matrices (YXZ sequence).

    Assumes the rotation order Ry(gamma) * Rx(beta) * Rz(alpha).

    Parameters
    ----------
    R : ndarray
        Rotation matrices, shape (3, 3, n_frames)
    flip_guard : bool
        True for joint angles: segment guard of the branch re-centring (see _recentre_segments).
        Segment angles (lab frame) use the plain whole-series rule (default False), because a
        segment may rotate through more than +/-180 deg in the lab (180 deg pivot turns) and
        Cardan singularities of the arm segments produce large Y/Z jumps that are not frame flips.

    Returns
    -------
    angles_xyz : dict
        Dictionary with keys 'X', 'Y', 'Z' (each shape (n_frames,))
        Angles in radians.
        For lower-limb joints:  X = abduction/adduction
                                Y = flexion/extension
                                Z = internal/external rotation
    References: Grood ES, Suntay WJ (1983) J Biomech Eng 105:136-144.; Wu G et al. (2002) J Biomech 35:543-548.
    """
    n_frames = R.shape[2]
    Y_yxz = np.zeros(n_frames)
    X_yxz = np.zeros(n_frames)
    Z_yxz = np.zeros(n_frames)
    
    for i in range(n_frames):
        # YXZ Cardan angles
        Y_yxz[i] = np.arctan2(R[0, 2, i], R[2, 2, i])
        X_yxz[i] = np.arctan2(-R[1, 2, i], 
                               np.sqrt(R[0, 2, i]**2 + R[2, 2, i]**2))
        Z_yxz[i] = np.arctan2(R[1, 0, i], R[1, 1, i])

    out = {'X': X_yxz, 'Y': Y_yxz, 'Z': Z_yxz}
    if _UNWRAP and n_frames > 1:
        # Remove arctan2 branch-cut jumps. The middle axis X is bounded to
        # [-pi/2, pi/2], so unwrap is a no-op there; Y and Z can wrap at +/-pi.
        out = {k: np.unwrap(v) for k, v in out.items()}
        # np.unwrap keeps the 2*pi branch of the first frame. If the first frames lie on the
        # other side of the +/-pi cut (e.g. a poorly tracked edge frame), the whole series ends
        # up exactly 2*pi off. A series whose median lies beyond +/-1.5*pi (270 deg) is therefore
        # re-centred by the nearest multiple of 2*pi; series within +/-270 deg are unchanged.
        # For joint angles, a jump-delimited segment that this shift would push beyond +/-270 deg
        # keeps its own branch, so a partial frame flip does not move the valid frames 360 deg.
        out = {k: _recentre_segments(v, flip_guard) for k, v in out.items()}
    return out


def getderivatepom_vel(V):
    """Numerical differentiation of a velocity signal (3-point method).

    Parameters
    ----------
    V : ndarray
        Velocity, shape (n_frames,)

    Returns
    -------
    dV : ndarray
        Per-sample central difference (divide by dt for acceleration), same shape
    References: Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    dV = np.zeros_like(V)
    
    for i in range(1, len(V) - 1):
        dV[i] = (V[i + 1] - V[i - 1]) / 2.0
    
    dV[0]  = dV[1]    # boundary: first sample copies its neighbour (last sample likewise)
    dV[-1] = dV[-2]
    return dV


def get_r_ref_norm(R1, R2, R_ref, ref_to_static=False):
    """Determine the joint rotation matrix between two segments.

    Optionally referenced to the static trial.

    Parameters
    ----------
    R1 : ndarray
        Proximal segment rotation, shape (3, 3, n_frames)
    R2 : ndarray
        Distal segment rotation, shape (3, 3, n_frames)
    R_ref : ndarray
        Static reference rotation, shape (3, 3)
    ref_to_static : bool
        If True: R_joint = (R1^T * R2) * R_ref^T
        If False: R_joint = R1^T * R2

    Returns
    -------
    R : ndarray
        Joint rotation matrices, shape (3, 3, n_frames)
    """
    n_frames = R1.shape[2]
    R = np.zeros((3, 3, n_frames))
    
    if ref_to_static:
        for i in range(n_frames):
            R[:, :, i] = (R1[:, :, i].T @ R2[:, :, i]) @ R_ref.T
    else:
        for i in range(n_frames):
            R[:, :, i] = R1[:, :, i].T @ R2[:, :, i]
    
    return R


def getsegmentderivatives(R, O, freq, linv_cutoff_hz=None):
    """Determine angular and linear velocities and accelerations of a segment.

    Parameters
    ----------
    R : ndarray
        Rotation matrices, shape (3, 3, n_frames)
    O : ndarray
        Segment origin, shape (3, n_frames), mm
    freq : float
        Sampling frequency (Hz)
    linv_cutoff_hz : float or None
        Low-pass cutoff for the linear velocity before differentiation to acceleration.
        Should match the marker filter cutoff (e.g. cut_off_marker). None disables filtering.

    Returns
    -------
    V : ndarray
        Angular velocity, shape (3, n_frames) [X, Y, Z components], rad/s
    ACC : ndarray
        Angular acceleration, shape (3, n_frames), rad/s^2
    LinV : ndarray
        Linear velocity, shape (3, n_frames), m/s
    LinACC : ndarray
        Linear acceleration, shape (3, n_frames), m/s^2
    Theta : ndarray
        Skew-symmetric matrix, shape (3, 3, n_frames)
    References: Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    dt = 1.0 / freq
    
    # Angular kinematics
    R_dot = getderivate_m3point(R) / dt
    Theta = gettheta(R_dot, R)
    
    # Extract angular velocity components from Theta
    V = np.zeros((3, R.shape[2]))
    V[0, :] = Theta[2, 1, :]  # omega_x
    V[1, :] = Theta[0, 2, :]  # omega_y
    V[2, :] = Theta[1, 0, :]  # omega_z

    # Optional low-pass of omega before differentiating to alpha (same band as the linear
    # path). Smooths both alpha and the gyroscopic term omega x (I omega).
    if _FILTER_ANGVEL and linv_cutoff_hz is not None and R.shape[2] > 15:
        V = butterworth_lowpass(V, cutoff_hz=linv_cutoff_hz, sample_rate_hz=freq, order=2, axis=1)

    # Angular acceleration
    ACC = np.zeros((3, R.shape[2]))
    for j in range(3):
        ACC[j, :] = getderivatepom_vel(V[j, :]) / dt
    
    # Linear kinematics
    LinV = getderivate_m3point(O) / dt / 1000.0  # Convert mm to m
    # Low-pass filter LinV before differentiation to reduce acceleration noise; the cutoff
    # matches the marker filter (linv_cutoff_hz) so both share a consistent band.
    if linv_cutoff_hz is not None:
        LinV = butterworth_lowpass(LinV, cutoff_hz=linv_cutoff_hz, sample_rate_hz=freq, order=2, axis=1)
    
    LinACC = np.zeros_like(LinV)
    for j in range(3):
        LinACC[j, :] = getderivatepom_vel(LinV[j, :]) / dt
    
    return V, ACC, LinV, LinACC, Theta


def get_default_segments_config():
    return _get_default_segments_config_shared()


def get_default_joint_pairs():
    return _get_default_joint_pairs_shared()


def reconstruct_kinematics(trial_dynamic, REFFRAME, segments_config=None,
                            joint_pairs=None, verbose=False, marker_cutoff_hz=None,
                            static_markers_filt=None, ref_to_static=False):
    """Full kinematic reconstruction: SVD optimisation -> anatomical frames -> joint angles.

    Parameters
    ----------
    trial_dynamic       : dict   Pre-processed dynamic trial (see markersub.kinematics.reconstruct).
    REFFRAME            : dict   Static calibration reference frames.
    segments_config     : dict   Segment definitions; defaults to get_default_segments_config().
    joint_pairs         : dict   Joint pairs; defaults to get_default_joint_pairs().
    verbose             : bool   Print per-segment status lines.
    static_markers_filt : dict   Filtered static trial marker dict {label: (3, n_static_frames)}.
                                 When provided, the static-trial mean is used as the svdopti
                                 reference. If None, the mean of the dynamic trial is used.

    Returns
    -------
    frames_dynamic : dict   {'Technical': {seg: {'O','R'}}, 'Anatomical': {seg: dict}}
    angles_dynamic : dict   {'SEGMENT': {'rad': ..., 'deg': ...}, 'JOINT': {'rad': ..., 'deg': ...}}
    References: Soederkvist I, Wedin PA (1993) J Biomech 26:1473-1477.; Arun KS, Huang TS, Blostein SD (1987) IEEE Trans Pattern Anal Mach Intell 9:698-700.; Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Clin Biomech 10:171-178.; Grood ES, Suntay WJ (1983) J Biomech Eng 105:136-144.
    """
    import traceback

    if segments_config is None:
        segments_config = get_default_segments_config()
    if joint_pairs is None:
        joint_pairs = get_default_joint_pairs()

    marker_labels    = trial_dynamic['marker_labels']
    n_dynamic_frames = trial_dynamic['marker_data'].shape[2]
    freq_kinematics  = float(trial_dynamic['marker_rate_hz'])

    # Reference marker positions for svdopti: the mean of the filtered static-trial
    # markers if given, otherwise the mean of the dynamic trial.
    if static_markers_filt is not None:
        ref_markers_dict = {
            label: np.asarray(static_markers_filt[label], dtype=float).mean(axis=1)
            for label in static_markers_filt
        }
    else:
        ref_markers_dict = {
            label: trial_dynamic['marker_data'][i, :3, :].mean(axis=1)
            for i, label in enumerate(marker_labels)
        }

    frames_dynamic = {'Technical': {}, 'Anatomical': {}}
    angles_dynamic = {
        'SEGMENT': {'rad': {}, 'deg': {}},
        'JOINT':   {'rad': {}, 'deg': {}},
    }

    # --- Static-referenced CAST reconstruction of missing cluster markers ---
    # Completes a cluster whose marker is fully occluded in the analysed window (all NaN, so
    # within-trial gap filling has no reference) by placing it from the static geometry via a
    # rigid fit on the >=3 visible, static-consistent cluster markers (swap frames gated out).
    # Only missing frames are filled, so the technical frame is fitted over the full
    # (over-determined) cluster. No-op on fully tracked data; runs only with a static reference.
    if static_markers_filt is not None:
        try:
            from .static_gap_fill import fill_missing_from_static
            frames_dynamic['static_recon'] = fill_missing_from_static(
                trial_dynamic['marker_data'], marker_labels, segments_config, ref_markers_dict)
        except Exception:
            frames_dynamic['static_recon'] = {}

    # --- Segment reconstruction ---
    for seg_name, seg_cfg in segments_config.items():
        ref_key = seg_cfg['ref_key']
        if ref_key not in REFFRAME['Technical_to_Anatomical']:
            continue
        try:
            marker_names = seg_cfg['markers']
            dynamic_markers, ref_markers_static, missing_dyn = [], [], []
            no_static_ref = []
            for m_name in marker_names:
                try:
                    m_idx = marker_labels.index(m_name)
                except ValueError:
                    missing_dyn.append(m_name)
                    continue
                # A marker whose label is present but which has any non-finite frame in the analysed
                # window is treated like an absent marker. Otherwise its NaN would reach svdopti (which
                # fits all frames jointly) and invalidate the whole segment. Dropping it lets the segment
                # build from the remaining fully present >=3 markers. A marker that the static
                # reconstruction completes is finite and therefore kept.
                m_col = trial_dynamic['marker_data'][m_idx, :3, :]
                # a frame at the lab origin (0, 0, 0) is a fill artefact, never a measurement
                if (not np.isfinite(m_col).all()) or np.any(np.all(np.abs(m_col) < 1e-9, axis=0)):
                    missing_dyn.append(m_name)
                    continue
                dynamic_markers.append(m_col)
                if m_name in ref_markers_dict:
                    ref_pos = ref_markers_dict[m_name].reshape(3, 1)
                else:
                    # Fallback: use the dynamic-trial mean for markers absent from the static trial
                    ref_pos = trial_dynamic['marker_data'][m_idx, :3, :].mean(axis=1).reshape(3, 1)
                    no_static_ref.append(m_name)
                ref_markers_static.append(ref_pos)

            if len(dynamic_markers) < 3:
                raise KeyError(f"only {len(dynamic_markers)}/{len(marker_names)} markers available (need >= 3)")
            if missing_dyn:
                print(f"  [WARN] {seg_name}: marker(s) missing in dynamic ({', '.join(missing_dyn)}), using {len(dynamic_markers)}/{len(marker_names)}")

            # The static calibration built this segment's technical frame from the first <=4
            # static markers of the candidate list (static_calibration._try_add_technical_frame)
            # and stored Technical_to_Anatomical relative to that frame. If a dynamic marker is
            # missing (absent or not fully finite) or has no static reference, a frame built from
            # the remaining markers would use a different definition (3- vs 4-marker formula, other
            # marker order) and R_tech @ R_ta would be wrong. In that case the static geometry is
            # rigidly fitted to the available markers, all static technical markers are rebuilt and
            # the frame is formed exactly as in the static trial. Complete data skip this branch.
            static_seq = None
            if static_markers_filt is not None and (missing_dyn or no_static_ref):
                static_seq = [m for m in marker_names if m in ref_markers_dict][:4]
                static_avail = [m for m in static_seq if m not in missing_dyn]
                dyn_used = [m for m in marker_names if m not in missing_dyn]
                if not no_static_ref and dyn_used == static_seq:
                    static_seq = None   # same markers in the same order as the static frame: the direct path is consistent
                elif len(static_seq) < 3 or len(static_avail) < 3:
                    raise KeyError(f"only {len(static_avail)}/{len(static_seq)} static-referenced markers available (need >= 3)")
                else:
                    # Geometry gate: nearly collinear remaining markers leave the rotation about their
                    # common line undetermined, so the segment is rejected instead of returning distorted angles.
                    _A = np.stack([ref_markers_dict[m] for m in static_avail], axis=0)
                    _sv = np.linalg.svd(_A - _A.mean(0), compute_uv=False)
                    if _sv[1] < 20.0:
                        raise KeyError(f"remaining markers {', '.join(static_avail)} nearly collinear "
                                       f"(2nd singular value {_sv[1]:.1f} mm < 20 mm)")
            if static_seq is not None:
                A_ref = np.stack([ref_markers_dict[m] for m in static_avail], axis=1)          # (3, k)
                B_dyn = np.stack([trial_dynamic['marker_data'][marker_labels.index(m), :3, :]
                                  for m in static_avail], axis=1)                                 # (3, k, n)
                a0 = A_ref.mean(axis=1, keepdims=True)
                Ys = [np.zeros((3, n_dynamic_frames)) for _ in static_seq]
                for i in range(n_dynamic_frames):
                    b0 = B_dyn[:, :, i].mean(axis=1, keepdims=True)
                    G = (B_dyn[:, :, i] - b0) @ (A_ref - a0).T
                    U, _, Vt = svd(G)
                    Rk = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
                    dk = b0[:, 0] - Rk @ a0[:, 0]
                    for q, m in enumerate(static_seq):
                        Ys[q][:, i] = Rk @ ref_markers_dict[m] + dk
                print(f"  {seg_name}: technical frame rebuilt from static geometry "
                      f"({', '.join(static_avail)} -> {', '.join(static_seq)})")
                n_available = len(static_seq)
                O_tech, R_tech = technical_frame_from_points(*Ys)
            else:
                n_available = len(dynamic_markers)
                dyn_stack = stack_markers(*dynamic_markers)
                ref_stack = stack_markers(*ref_markers_static)
                ref_stack = np.repeat(ref_stack, n_dynamic_frames, axis=2)

                optimized = svdopti(dyn_stack, ref_stack)
                if optimized is None:
                    raise ValueError("svdopti returned None")

                unstacked = unstack_markers(optimized)
                if not isinstance(unstacked, tuple) or len(unstacked) != 4:
                    raise ValueError("unstack_markers() returned unexpected type/length")
                Y1, Y2, Y3, Y4 = unstacked

                if n_available == 3:
                    O_tech, R_tech = technical_frame_from_points(Y1, Y2, Y3)
                else:
                    O_tech, R_tech = technical_frame_from_points(Y1, Y2, Y3, Y4)

            frames_dynamic['Technical'][seg_name] = {'O': O_tech, 'R': R_tech}

            R_anat = np.zeros_like(R_tech)
            O_anat = np.zeros_like(O_tech)
            R_ta = REFFRAME['Technical_to_Anatomical'][ref_key]['R']
            O_ta = REFFRAME['Technical_to_Anatomical'][ref_key]['O']
            for i in range(n_dynamic_frames):
                R_anat[:, :, i] = R_tech[:, :, i] @ R_ta
                O_anat[:, i]    = O_tech[:, i] + R_tech[:, :, i] @ O_ta

            frames_dynamic['Anatomical'][seg_name] = {'O': O_anat, 'R': R_anat}

            V, ACC, LinV, LinACC, Theta = getsegmentderivatives(R_anat, O_anat, freq_kinematics, linv_cutoff_hz=marker_cutoff_hz)
            frames_dynamic['Anatomical'][seg_name].update(
                {'V': V, 'ACC': ACC, 'LinV': LinV, 'LinACC': LinACC, 'Theta': Theta}
            )

            angles_xyz = cardan_angles(R_anat)
            # Correct left-side angles for the coordinate system mirror
            if 'Left' in seg_name:
                angles_xyz = _correct_left_side_angles(angles_xyz)
            angles_dynamic['SEGMENT']['rad'][seg_name] = angles_xyz
            angles_dynamic['SEGMENT']['deg'][seg_name] = {k: np.degrees(v) for k, v in angles_xyz.items()}

            if verbose:
                print(f"  OK {seg_name:15s}: O={O_anat.shape}  R={R_anat.shape}")

        except KeyError as e:
            print(f"  WARNING {seg_name:15s}: KeyError: {e}")
            if verbose:
                traceback.print_exc()
        except Exception as e:
            print(f"  FAILED {seg_name:15s}: {type(e).__name__}: {e}")
            if verbose:
                traceback.print_exc()

    # --- Joint angle computation ---
    joint_ref_map = {
        'Right_Hip': 'Right_Pelvis_to_Thigh',
        'Right_Knee': 'Right_Thigh_to_Shank',
        'Right_Ankle': 'Right_Shank_to_Rearfoot',
        'Right_MidFoot': 'Right_Rearfoot_to_Forefoot',
        'Left_Hip': 'Left_Pelvis_to_Thigh',
        'Left_Knee': 'Left_Thigh_to_Shank',
        'Left_Ankle': 'Left_Shank_to_Rearfoot',
        'Left_MidFoot': 'Left_Rearfoot_to_Forefoot',
        'Trunk_Pelvis': 'Pelvis_to_Torso',
    }

    for joint_name, (prox_seg, dist_seg) in joint_pairs.items():
        try:
            if prox_seg not in frames_dynamic['Anatomical'] or dist_seg not in frames_dynamic['Anatomical']:
                continue
            R_prox = frames_dynamic['Anatomical'][prox_seg]['R']
            R_dist = frames_dynamic['Anatomical'][dist_seg]['R']

            ref_key = joint_ref_map.get(joint_name)
            R_ref = None
            if ref_key is not None:
                R_ref = REFFRAME.get('Joint', {}).get(ref_key)

            if R_ref is not None and ref_to_static:
                R_joint = get_r_ref_norm(R_prox, R_dist, R_ref, ref_to_static=True)
            else:                                           # no static referencing: R_prox^T R_dist
                R_joint = get_r_ref_norm(R_prox, R_dist, np.eye(3), ref_to_static=False)

            frames_dynamic['Anatomical'][f"{joint_name}_Joint"] = {'R': R_joint}

            angles_xyz = cardan_angles(R_joint, flip_guard=True)   # branch segment guard for joint angles only
            # Correct left-side angles for the coordinate system mirror
            if 'Left' in joint_name:
                angles_xyz = _correct_left_side_angles(angles_xyz)
            angles_dynamic['JOINT']['rad'][joint_name] = angles_xyz
            angles_dynamic['JOINT']['deg'][joint_name] = {k: np.degrees(v) for k, v in angles_xyz.items()}

            if verbose:
                print(f"  OK {joint_name:15s}: {R_joint.shape}")

        except Exception as e:
            print(f"  WARNING {joint_name:15s}: {str(e)[:60]}")

    return frames_dynamic, angles_dynamic
