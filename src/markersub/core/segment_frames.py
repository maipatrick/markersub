"""Technical and anatomical segment frames of the lower-limb (and upper-body) model.

Marker trajectories are low-pass filtered with a zero-lag Butterworth filter.
Technical frames are built from 3- or 4-marker clusters, anatomical frames from
static anatomical landmarks following the ISB recommendations (Wu et al. 2002,
J Biomech 35:543-548; Wu et al. 2005, J Biomech 38:981-992). Rotation matrices
have the columns [X|Y|Z]; for the leg segments Z is the long axis and Y the
mediolateral (flexion) axis, for the foot segments X runs heel -> forefoot.
Static mean rotations are projected onto SO(3). Positions are in mm, segment
lengths in m.
"""

import numpy as np
from scipy.signal import filtfilt


# Debug switch: use the lab-axis seed [0,1,0] for the thigh frame instead of the
# anatomical epicondylar axis. Default False (epicondyles).
_FORCE_LAB_THIGH = False


# Debug switch: use the lab-axis seed [0,1,0] for the shank frame instead of the
# anatomical malleolar axis (ISB tibia, Wu et al. 2002). Default False (malleoli).
# See RightShank_regular for why knee angles are unaffected by this choice.
_FORCE_LAB_SHANK = False


def markersfilt(raw_data, b, a):
    """Filter raw 3D trajectory data with a zero-lag Butterworth filter.

    Parameters
    ----------
    raw_data : ndarray
        Raw marker trajectory, shape (3, n_frames)
    b, a : ndarray
        Butterworth filter coefficients

    Returns
    -------
    filt_data : ndarray
        Filtered trajectory, shape (3, n_frames)
    References: Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    filt_data = np.zeros_like(raw_data, dtype=float)
    for j in range(3):
        # Fill NaN by linear interpolation (constant at the ends)
        filled = raw_data[j, :].copy()
        nan_idx = np.isnan(filled)
        if nan_idx.any():
            valid_idx = np.where(~nan_idx)[0]
            if len(valid_idx) > 0:
                filled[nan_idx] = np.interp(np.where(nan_idx)[0], valid_idx, filled[valid_idx])
        
        # Apply zero-phase filter
        filt_data[j, :] = filtfilt(b, a, filled)
    
    return filt_data


def get_technical_frame(P1, P2, P3, P4=None):
    """Compute the technical coordinate system from a marker cluster (3 or 4 markers).

    Parameters
    ----------
    P1, P2, P3 : ndarray
        Marker positions, each shape (3, n_frames)
    P4 : ndarray, optional
        Fourth marker for the 4-marker case

    Returns
    -------
    O : ndarray
        Origin, shape (3, n_frames)
    R : ndarray
        Rotation matrices, shape (3, 3, n_frames)
    References: Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Clin Biomech 10:171-178.
    """
    _, n_frames = P1.shape
    O = np.zeros((3, n_frames))
    
    if P4 is None:
        # 3-marker case
        for i in range(n_frames):
            O[:, i] = (P1[:, i] + P2[:, i] + P3[:, i]) / 3.0
        
        X = np.zeros((3, n_frames))
        Y = np.zeros((3, n_frames))
        Z = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            X[:, i] = P2[:, i] - P1[:, i]
            Help1 = P3[:, i] - P1[:, i]
            Z[:, i] = np.cross(X[:, i], Help1)
            Y[:, i] = np.cross(Z[:, i], X[:, i])
            
            X[:, i] = _safe_normalize(X[:, i], (1.0, 0.0, 0.0))
            Y[:, i] = _safe_normalize(Y[:, i], (0.0, 1.0, 0.0))
            Z[:, i] = _safe_normalize(Z[:, i], (0.0, 0.0, 1.0))
    else:
        # 4-marker case
        for i in range(n_frames):
            O[:, i] = (P1[:, i] + P2[:, i] + P3[:, i] + P4[:, i]) / 4.0
        
        X = np.zeros((3, n_frames))
        Y = np.zeros((3, n_frames))
        Z = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            Z[:, i] = ((P1[:, i] + P2[:, i]) / 2.0) - O[:, i]
            Help1 = ((P2[:, i] + P4[:, i]) / 2.0) - O[:, i]
            X[:, i] = np.cross(Help1, Z[:, i])
            Y[:, i] = np.cross(Z[:, i], X[:, i])
            
            X[:, i] = _safe_normalize(X[:, i], (1.0, 0.0, 0.0))
            Y[:, i] = _safe_normalize(Y[:, i], (0.0, 1.0, 0.0))
            Z[:, i] = _safe_normalize(Z[:, i], (0.0, 0.0, 1.0))
    
    # Build rotation matrix [X | Y | Z] columnwise, shape (3, 3, n_frames)
    R = np.zeros((3, 3, n_frames))
    for i in range(n_frames):
        R[:, 0, i] = X[:, i]
        R[:, 1, i] = Y[:, i]
        R[:, 2, i] = Z[:, i]
    
    return O, R


def Rmean(R):
    """Mean of rotation matrices over frames, projected back onto SO(3).

    The elementwise mean of rotation matrices is not itself a rotation (SO(3) is not
    convex): it is neither orthonormal nor has det = +1. For a still static trial the
    error is tiny (<0.01 deg), but it is unbounded if the static contains motion or gaps.
    The chordal L2 mean is used, i.e. the closest rotation to the elementwise mean via
    the polar/SVD projection M = U S V^T -> R_mean = U diag(1, 1, det(U V^T)) V^T (the
    det term avoids a reflection). For an already orthonormal mean the input is returned.

    Parameters
    ----------
    R : ndarray
        Rotation matrices, shape (3, 3, n_frames)

    Returns
    -------
    Rmean_out : ndarray
        Nearest rotation matrix to the elementwise mean, shape (3, 3)
    References: Moakher M (2002) SIAM J Matrix Anal Appl 24:1-16.
    """
    M = np.mean(R, axis=2)
    try:
        U, _s, Vt = np.linalg.svd(M)
        D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
        return U @ D @ Vt
    except np.linalg.LinAlgError:
        return M


def _help_perp(z_unit, primary=(0.0, 1.0, 0.0), alt=(0.0, 0.0, 1.0)):
    """Return the transverse seed vector, swapped to `alt` when `primary` is nearly
    collinear with the (unit) segment long axis z_unit. Otherwise primary x z ~ 0 and the
    frame normalisation divides by ~0 (silent NaN). Guards the fixed lab seed [0,1,0]
    for data that are not Z-up and for near-horizontal segments.
    """
    p = np.asarray(primary, dtype=float)
    if abs(float(np.dot(p, z_unit))) > 0.98:
        return np.asarray(alt, dtype=float)
    return p


def Vmean(V):
    """Compute the mean of a vector over frames.

    Parameters
    ----------
    V : ndarray
        Vectors, shape (3, n_frames)

    Returns
    -------
    Vmean_out : ndarray
        Mean vector, shape (3,)
    """
    return np.mean(V, axis=1)


def _safe_normalize(v, fallback):
    n = np.linalg.norm(v)
    if n <= 1e-12 or not np.isfinite(n):
        return np.asarray(fallback, dtype=float)
    return v / n


def _build_frame_xyz_from_points(O_allframes, p_x_from, p_x_to, p_y_from, p_y_to):
    """Build the mean anatomical frame from marker-derived axis seeds across static frames."""
    _, n_frames = O_allframes.shape
    R_all = np.zeros((3, 3, n_frames), dtype=float)

    for i in range(n_frames):
        x_seed = p_x_to[:, i] - p_x_from[:, i]
        y_seed = p_y_to[:, i] - p_y_from[:, i]

        x = _safe_normalize(x_seed, np.array([1.0, 0.0, 0.0]))
        z = np.cross(x, y_seed)
        z = _safe_normalize(z, np.array([0.0, 0.0, 1.0]))
        y = np.cross(z, x)
        y = _safe_normalize(y, np.array([0.0, 1.0, 0.0]))

        # Re-orthogonalize z after potential y fallback.
        z = np.cross(x, y)
        z = _safe_normalize(z, np.array([0.0, 0.0, 1.0]))

        R_all[:, 0, i] = x
        R_all[:, 1, i] = y
        R_all[:, 2, i] = z

    O = np.mean(O_allframes, axis=1)
    R = Rmean(R_all)
    return O, R


def _marker_any(markers_filtered, *names):
    for name in names:
        value = markers_filtered.get(name)
        if value is not None:
            return value
    return None


def getanaframe(markers_filtered, segment_name, anthro=None):
    """Compute the anatomical reference frame of a segment (static trial).

    Parameters
    ----------
    markers_filtered : dict
        Dictionary of filtered marker trajectories {label: ndarray (3, n_frames)}
    segment_name : str
        Segment identifier (e.g., 'RightShank_regular', 'RightPelvis_regular')
    anthro : dict, optional
        Anthropometric data (e.g., {'FootLength': 250})

    Returns
    -------
    O : ndarray
        Origin, shape (3,)
    R : ndarray
        Rotation matrix, shape (3, 3)
    SLength : float
        Segment length in m (if computed, else 0)
    References: Wu G et al. (2002) J Biomech 35:543-548.; Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Clin Biomech 10:171-178.
    """
    
    if anthro is None:
        anthro = {}

    # Segment CoM location along the segment (de Leva 1996), gender dependent.
    # Male values are the default; female values are used when gender starts with 'f'.
    _female = str(anthro.get('gender', 'male')).strip().lower().startswith('f')
    _COM_THIGH = 0.3612 if _female else 0.4095    # from hip toward knee
    _COM_SHANK = 0.4352 if _female else 0.4395    # from knee toward ankle

    n_frames = next(iter(markers_filtered.values())).shape[1]
    
    # ===================== RIGHT LEG =====================
    
    if segment_name == 'RightShank_regular':
        mal_med = markers_filtered.get('mal_med_right')
        mal_lat = markers_filtered.get('mal_lat_right')
        epi_med = markers_filtered.get('epi_med_right')
        epi_lat = markers_filtered.get('epi_lat_right')
        
        O_allframes = np.zeros((3, n_frames))
        MidPointKnee = np.zeros((3, n_frames))
        MidPointAnkle = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            MidPointKnee[:, i] = (epi_med[:, i] + epi_lat[:, i]) / 2.0
            MidPointAnkle[:, i] = (mal_med[:, i] + mal_lat[:, i]) / 2.0
            O_allframes[:, i] = MidPointKnee[:, i] + (MidPointAnkle[:, i] - MidPointKnee[:, i]) * _COM_SHANK
        
        O = np.mean(O_allframes, axis=1)
        
        # Build rotation matrix
        X = np.zeros((3, n_frames))
        Y = np.zeros((3, n_frames))
        Z = np.zeros((3, n_frames))
        Zo = np.zeros((3, n_frames))
        
        # Frontal/transverse axes from the malleolar axis (mal_lat - mal_med, the ISB
        # intermalleolar axis; Wu et al. 2002) instead of the fixed lab seed [0,1,0], so the
        # shank ab/adduction and rotation reference is tied to the tibia. Falls back to the lab
        # seed if a malleolus is missing. The sign is aligned with the lab-seed axis so the axis
        # convention does not flip. For the knee (shank = distal segment) this is a pure rotation
        # about the shank long axis that cancels in the static-referenced joint angle, so knee
        # angles and knee moments are unaffected. It changes the ankle angle (shank = proximal
        # segment) and the shank segment angle, which become malleolus referenced.
        _use_mal = (mal_med is not None and mal_lat is not None) and not _FORCE_LAB_SHANK
        for i in range(n_frames):
            Z[:, i] = MidPointKnee[:, i] - MidPointAnkle[:, i]
            Zo[:, i] = Z[:, i].copy()
            if _use_mal:
                ml = mal_lat[:, i] - mal_med[:, i]          # malleolar (ML) axis
                Y_old = np.cross(Z[:, i], np.cross([0, 1, 0], Z[:, i]))
                if np.dot(ml, Y_old) < 0:
                    ml = -ml                                # keep the sign of the lab-seed ML axis
                Help1 = ml
            else:
                Help1 = _help_perp(_safe_normalize(Z[:, i], (0.0, 0.0, 1.0)))
            X[:, i] = np.cross(Help1, Z[:, i])
            Y[:, i] = np.cross(Z[:, i], X[:, i])
            X[:, i] = _safe_normalize(X[:, i], (1.0, 0.0, 0.0))
            Y[:, i] = _safe_normalize(Y[:, i], (0.0, 1.0, 0.0))
            Z[:, i] = _safe_normalize(Z[:, i], (0.0, 0.0, 1.0))

        R_all = np.zeros((3, 3, n_frames))
        for i in range(n_frames):
            R_all[:, 0, i] = X[:, i]
            R_all[:, 1, i] = Y[:, i]
            R_all[:, 2, i] = Z[:, i]

        R = Rmean(R_all)
        SLength = np.linalg.norm(np.mean(Zo, axis=1)) / 1000.0

        return O, R, SLength

    elif segment_name == 'RightThigh_regular':
        cluster_femur_1 = markers_filtered.get('cluster_femur_right_1')
        epi_med = markers_filtered.get('epi_med_right')
        epi_lat = markers_filtered.get('epi_lat_right')
        hip_jc = markers_filtered.get('hip_joint_center_right')
        
        O_allframes = np.zeros((3, n_frames))
        MidPointKnee = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            MidPointKnee[:, i] = (epi_med[:, i] + epi_lat[:, i]) / 2.0
            O_allframes[:, i] = hip_jc[:, i] + (MidPointKnee[:, i] - hip_jc[:, i]) * _COM_THIGH
        
        O = np.mean(O_allframes, axis=1)
        
        X = np.zeros((3, n_frames))
        Y = np.zeros((3, n_frames))
        Z = np.zeros((3, n_frames))
        Zo = np.zeros((3, n_frames))
        
        # Frontal/transverse axes from the knee epicondylar axis (anatomical, ISB) instead
        # of the fixed lab axis [0,1,0]. With the lab axis, thigh ab/adduction and rotation
        # are not tied to the femur and drift as the leg rotates transversally (e.g. in a
        # cutting manoeuvre). Falls back to the lab axis if an epicondyle marker is missing.
        # The sign is aligned with the lab-seed flexion axis so the flexion convention does not flip.
        _use_epi = (epi_med is not None and epi_lat is not None) and not _FORCE_LAB_THIGH
        for i in range(n_frames):
            Z[:, i] = hip_jc[:, i] - MidPointKnee[:, i]
            Zo[:, i] = Z[:, i].copy()
            if _use_epi:
                ml = epi_lat[:, i] - epi_med[:, i]          # epicondylar (flexion) axis
                Y_old = np.cross(Z[:, i], np.cross([0, 1, 0], Z[:, i]))
                if np.dot(ml, Y_old) < 0:
                    ml = -ml                                # keep the sign of the lab-seed flexion axis
                Help1 = ml
            else:
                Help1 = _help_perp(_safe_normalize(Z[:, i], (0.0, 0.0, 1.0)))
            X[:, i] = np.cross(Help1, Z[:, i])
            Y[:, i] = np.cross(Z[:, i], X[:, i])
            X[:, i] = _safe_normalize(X[:, i], (1.0, 0.0, 0.0))
            Y[:, i] = _safe_normalize(Y[:, i], (0.0, 1.0, 0.0))
            Z[:, i] = _safe_normalize(Z[:, i], (0.0, 0.0, 1.0))

        R_all = np.zeros((3, 3, n_frames))
        for i in range(n_frames):
            R_all[:, 0, i] = X[:, i]
            R_all[:, 1, i] = Y[:, i]
            R_all[:, 2, i] = Z[:, i]

        R = Rmean(R_all)
        SLength = np.linalg.norm(np.mean(Zo, axis=1)) / 1000.0

        return O, R, SLength

    elif segment_name == 'RightPelvis_regular':
        SIAS_left = markers_filtered.get('SIAS_left')
        SIAS_right = markers_filtered.get('SIAS_right')
        SIPS_left = markers_filtered.get('SIPS_left')
        SIPS_right = markers_filtered.get('SIPS_right')
        sacrum = markers_filtered.get('sacrum')   # if present (single sacrum marker), used as mid-PSIS

        X = np.zeros((3, n_frames))
        Y = np.zeros((3, n_frames))
        Z = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            MPASI = (SIAS_left[:, i] + SIAS_right[:, i]) / 2.0
            MPPSI = sacrum[:, i] if sacrum is not None else (SIPS_left[:, i] + SIPS_right[:, i]) / 2.0
            Y[:, i] = SIAS_left[:, i] - SIAS_right[:, i]
            Help1 = MPASI - MPPSI
            Z[:, i] = np.cross(Help1, Y[:, i])
            X[:, i] = np.cross(Y[:, i], Z[:, i])
            X[:, i] = _safe_normalize(X[:, i], (1.0, 0.0, 0.0))
            Y[:, i] = _safe_normalize(Y[:, i], (0.0, 1.0, 0.0))
            Z[:, i] = _safe_normalize(Z[:, i], (0.0, 0.0, 1.0))
        
        R_all = np.zeros((3, 3, n_frames))
        for i in range(n_frames):
            R_all[:, 0, i] = X[:, i]
            R_all[:, 1, i] = Y[:, i]
            R_all[:, 2, i] = Z[:, i]
        
        R = Rmean(R_all)
        
        # Hip center position calculation (Seidel/Bell regression)
        DIST_ASIS = np.mean(np.linalg.norm(SIAS_right - SIAS_left, axis=0))
        DIST_RASI_RPSI = np.mean(np.linalg.norm(SIAS_right - SIPS_right, axis=0))
        
        POS_hip_local = np.array([
            -DIST_RASI_RPSI * 0.34,
            DIST_ASIS * 0.14,
            -DIST_ASIS * 0.31
        ])
        
        O_allframes = np.zeros((3, n_frames))
        for i in range(n_frames):
            O_allframes[:, i] = R @ POS_hip_local + SIAS_right[:, i]
        
        O = np.mean(O_allframes, axis=1)
        SLength = 0.0
        
        return O, R, SLength
    
    # ===================== LEFT LEG =====================
    
    elif segment_name == 'LeftShank_regular':
        mal_med = markers_filtered.get('mal_med_left')
        mal_lat = markers_filtered.get('mal_lat_left')
        epi_med = markers_filtered.get('epi_med_left')
        epi_lat = markers_filtered.get('epi_lat_left')
        
        O_allframes = np.zeros((3, n_frames))
        MidPointKnee = np.zeros((3, n_frames))
        MidPointAnkle = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            MidPointKnee[:, i] = (epi_med[:, i] + epi_lat[:, i]) / 2.0
            MidPointAnkle[:, i] = (mal_med[:, i] + mal_lat[:, i]) / 2.0
            O_allframes[:, i] = MidPointKnee[:, i] + (MidPointAnkle[:, i] - MidPointKnee[:, i]) * _COM_SHANK
        
        O = np.mean(O_allframes, axis=1)
        
        X = np.zeros((3, n_frames))
        Y = np.zeros((3, n_frames))
        Z = np.zeros((3, n_frames))
        Zo = np.zeros((3, n_frames))
        
        # Malleolar ML axis (ISB tibia, Wu et al. 2002) instead of the lab seed [0,1,0].
        # See RightShank_regular for the rationale and why knee angles are unaffected.
        _use_mal = (mal_med is not None and mal_lat is not None) and not _FORCE_LAB_SHANK
        for i in range(n_frames):
            Z[:, i] = MidPointKnee[:, i] - MidPointAnkle[:, i]
            Zo[:, i] = Z[:, i].copy()
            if _use_mal:
                ml = mal_lat[:, i] - mal_med[:, i]          # malleolar (ML) axis
                Y_old = np.cross(Z[:, i], np.cross([0, 1, 0], Z[:, i]))
                if np.dot(ml, Y_old) < 0:
                    ml = -ml                                # keep the sign of the lab-seed ML axis
                Help1 = ml
            else:
                Help1 = _help_perp(_safe_normalize(Z[:, i], (0.0, 0.0, 1.0)))
            X[:, i] = np.cross(Help1, Z[:, i])
            Y[:, i] = np.cross(Z[:, i], X[:, i])
            X[:, i] = _safe_normalize(X[:, i], (1.0, 0.0, 0.0))
            Y[:, i] = _safe_normalize(Y[:, i], (0.0, 1.0, 0.0))
            Z[:, i] = _safe_normalize(Z[:, i], (0.0, 0.0, 1.0))

        R_all = np.zeros((3, 3, n_frames))
        for i in range(n_frames):
            R_all[:, 0, i] = X[:, i]
            R_all[:, 1, i] = Y[:, i]
            R_all[:, 2, i] = Z[:, i]

        R = Rmean(R_all)
        SLength = np.linalg.norm(np.mean(Zo, axis=1)) / 1000.0

        return O, R, SLength

    elif segment_name == 'LeftThigh_regular':
        cluster_femur_1 = markers_filtered.get('cluster_femur_left_1')
        epi_med = markers_filtered.get('epi_med_left')
        epi_lat = markers_filtered.get('epi_lat_left')
        hip_jc = markers_filtered.get('hip_joint_center_left')
        
        O_allframes = np.zeros((3, n_frames))
        MidPointKnee = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            MidPointKnee[:, i] = (epi_med[:, i] + epi_lat[:, i]) / 2.0
            O_allframes[:, i] = hip_jc[:, i] + (MidPointKnee[:, i] - hip_jc[:, i]) * _COM_THIGH
        
        O = np.mean(O_allframes, axis=1)
        
        X = np.zeros((3, n_frames))
        Y = np.zeros((3, n_frames))
        Z = np.zeros((3, n_frames))
        Zo = np.zeros((3, n_frames))
        
        # Frontal/transverse axes from the knee epicondylar axis (anatomical, ISB)
        # instead of the fixed lab axis [0,1,0]. See RightThigh_regular for the rationale.
        _use_epi = (epi_med is not None and epi_lat is not None) and not _FORCE_LAB_THIGH
        for i in range(n_frames):
            Z[:, i] = hip_jc[:, i] - MidPointKnee[:, i]
            Zo[:, i] = Z[:, i].copy()
            if _use_epi:
                ml = epi_lat[:, i] - epi_med[:, i]          # epicondylar (flexion) axis
                Y_old = np.cross(Z[:, i], np.cross([0, 1, 0], Z[:, i]))
                if np.dot(ml, Y_old) < 0:
                    ml = -ml                                # keep the sign of the lab-seed flexion axis
                Help1 = ml
            else:
                Help1 = _help_perp(_safe_normalize(Z[:, i], (0.0, 0.0, 1.0)))
            X[:, i] = np.cross(Help1, Z[:, i])
            Y[:, i] = np.cross(Z[:, i], X[:, i])
            X[:, i] = _safe_normalize(X[:, i], (1.0, 0.0, 0.0))
            Y[:, i] = _safe_normalize(Y[:, i], (0.0, 1.0, 0.0))
            Z[:, i] = _safe_normalize(Z[:, i], (0.0, 0.0, 1.0))

        R_all = np.zeros((3, 3, n_frames))
        for i in range(n_frames):
            R_all[:, 0, i] = X[:, i]
            R_all[:, 1, i] = Y[:, i]
            R_all[:, 2, i] = Z[:, i]

        R = Rmean(R_all)
        SLength = np.linalg.norm(np.mean(Zo, axis=1)) / 1000.0

        return O, R, SLength

    elif segment_name == 'LeftPelvis_regular':
        SIAS_left = markers_filtered.get('SIAS_left')
        SIAS_right = markers_filtered.get('SIAS_right')
        SIPS_left = markers_filtered.get('SIPS_left')
        SIPS_right = markers_filtered.get('SIPS_right')
        sacrum = markers_filtered.get('sacrum')   # if present (single sacrum marker), used as mid-PSIS

        X = np.zeros((3, n_frames))
        Y = np.zeros((3, n_frames))
        Z = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            MPASI = (SIAS_left[:, i] + SIAS_right[:, i]) / 2.0
            MPPSI = sacrum[:, i] if sacrum is not None else (SIPS_left[:, i] + SIPS_right[:, i]) / 2.0
            Y[:, i] = SIAS_left[:, i] - SIAS_right[:, i]
            Help1 = MPASI - MPPSI
            Z[:, i] = np.cross(Help1, Y[:, i])
            X[:, i] = np.cross(Y[:, i], Z[:, i])
            X[:, i] = _safe_normalize(X[:, i], (1.0, 0.0, 0.0))
            Y[:, i] = _safe_normalize(Y[:, i], (0.0, 1.0, 0.0))
            Z[:, i] = _safe_normalize(Z[:, i], (0.0, 0.0, 1.0))
        
        R_all = np.zeros((3, 3, n_frames))
        for i in range(n_frames):
            R_all[:, 0, i] = X[:, i]
            R_all[:, 1, i] = Y[:, i]
            R_all[:, 2, i] = Z[:, i]
        
        R = Rmean(R_all)
        
        # Hip center position calculation (Seidel/Bell regression, left side)
        DIST_ASIS = np.mean(np.linalg.norm(SIAS_right - SIAS_left, axis=0))
        DIST_LASI_LPSI = np.mean(np.linalg.norm(SIAS_left - SIPS_left, axis=0))
        
        POS_hip_local = np.array([
            -DIST_LASI_LPSI * 0.34,
            -DIST_ASIS * 0.14,  # Minus for left side
            -DIST_ASIS * 0.31
        ])
        
        O_allframes = np.zeros((3, n_frames))
        for i in range(n_frames):
            O_allframes[:, i] = R @ POS_hip_local + SIAS_left[:, i]
        
        O = np.mean(O_allframes, axis=1)
        SLength = 0.0
        
        return O, R, SLength

    # ===================== UPPER BODY =====================

    elif segment_name == 'Head_regular':
        head_front_right = markers_filtered.get('head_front_right')
        head_front_left = markers_filtered.get('head_front_left')
        head_back_right = markers_filtered.get('head_back_right')
        head_back_left = markers_filtered.get('head_back_left')
        # 3-marker head cluster (front + left/right, no back markers)
        head_front = markers_filtered.get('head_front')
        head_left = markers_filtered.get('head_left')
        head_right = markers_filtered.get('head_right')
        ear_right = markers_filtered.get('ear_right')
        ear_left = markers_filtered.get('ear_left')
        head_center = markers_filtered.get('head_center')   # markerless: single head-centre keypoint

        # Preference order (the head only feeds the whole-body CoM, so the origin O is what
        # matters; R is not used downstream):
        #   1) 4-marker head set (front/back, left/right);
        #   2) 3-marker head cluster -> O = centroid of the three markers;
        #   3) ear_left/ear_right;
        #   4) single head-centre keypoint (markerless).
        if all(m is not None for m in (head_front_right, head_front_left, head_back_right, head_back_left)):
            front_center = 0.5 * (head_front_right + head_front_left)
            back_center = 0.5 * (head_back_right + head_back_left)
            right_center = 0.5 * (head_front_right + head_back_right)
            left_center = 0.5 * (head_front_left + head_back_left)
            O_allframes = 0.5 * (front_center + back_center)
            O, R = _build_frame_xyz_from_points(
                O_allframes,
                p_x_from=back_center,
                p_x_to=front_center,
                p_y_from=right_center,
                p_y_to=left_center,
            )
        elif all(m is not None for m in (head_front, head_left, head_right)):
            lr_center = 0.5 * (head_left + head_right)
            O_allframes = (head_front + head_left + head_right) / 3.0   # head centroid ~ head centre
            O, R = _build_frame_xyz_from_points(
                O_allframes,
                p_x_from=lr_center,    # posterior (between the side markers) -> anterior (front marker)
                p_x_to=head_front,
                p_y_from=head_right,
                p_y_to=head_left,
            )
        elif ear_right is not None and ear_left is not None:
            O_allframes = 0.5 * (ear_right + ear_left)
            O = np.mean(O_allframes, axis=1)
            R = np.eye(3, dtype=float)
        elif head_center is not None:
            # Markerless: the head only feeds the whole-body CoM (O = head centre); R is unused downstream.
            O = np.mean(head_center, axis=1)
            R = np.eye(3, dtype=float)
        else:
            raise KeyError(
                "Head segment requires head_front_*/head_back_* (4), head_front/left/right (3), "
                "ear_right+ear_left (2), or head_center (1)"
            )

        height_mm = float(
            anthro.get('HeightMM', anthro.get('BodyHeightMM', anthro.get('Height', 0.0)))
        )
        SLength = (height_mm / 1000.0) * 0.1395 if height_mm > 0.0 else 0.0
        return O, R, SLength

    elif segment_name == 'RightUpperArm_regular':
        acrom_right = markers_filtered.get('acrom_right')
        shoulder_right = markers_filtered.get('shoulder_right')
        elbow_top_right = markers_filtered.get('elbow_top_right')

        if elbow_top_right is None:   # markerless: elbow centre = midpoint of lat+med epicondyle
            _el = markers_filtered.get('elbow_lat_right'); _em = markers_filtered.get('elbow_med_right')
            if _el is not None and _em is not None:
                elbow_top_right = 0.5 * (_el + _em)

        if acrom_right is None or elbow_top_right is None:
            raise KeyError(
                "RightUpperArm requires at least acrom_right and elbow_top_right"
            )

        O_allframes = np.zeros((3, n_frames))
        shoulder_center = np.zeros((3, n_frames))
        R_all = np.zeros((3, 3, n_frames))
        Zo = np.zeros((3, n_frames))

        for i in range(n_frames):
            shoulder_z = shoulder_right[2, i] if shoulder_right is not None else acrom_right[2, i]
            shoulder_center[:, i] = np.array([
                acrom_right[0, i],
                acrom_right[1, i],
                shoulder_z,
            ])
            O_allframes[:, i] = shoulder_center[:, i] + (elbow_top_right[:, i] - shoulder_center[:, i]) * 0.5772

            z = shoulder_center[:, i] - elbow_top_right[:, i]
            Zo[:, i] = z
            if z[2] > np.linalg.norm(z) * 0.5:
                help1 = np.array([0.0, 1.0, 0.0])
                x = np.cross(help1, z)
                y = np.cross(z, x)
            else:
                help1 = np.array([1.0, 0.0, 0.0])
                y = np.cross(z, help1)
                x = np.cross(y, z)

            x = _safe_normalize(x, np.array([1.0, 0.0, 0.0]))
            y = _safe_normalize(y, np.array([0.0, 1.0, 0.0]))
            z = _safe_normalize(z, np.array([0.0, 0.0, 1.0]))
            R_all[:, 0, i] = x
            R_all[:, 1, i] = y
            R_all[:, 2, i] = z

        O = np.mean(O_allframes, axis=1)
        R = Rmean(R_all)
        SLength = np.linalg.norm(np.mean(Zo, axis=1)) / 1000.0
        return O, R, SLength

    elif segment_name == 'LeftUpperArm_regular':
        acrom_left = markers_filtered.get('acrom_left')
        shoulder_left = markers_filtered.get('shoulder_left')
        elbow_top_left = markers_filtered.get('elbow_top_left')

        if elbow_top_left is None:   # markerless: elbow centre = midpoint of lat+med epicondyle
            _el = markers_filtered.get('elbow_lat_left'); _em = markers_filtered.get('elbow_med_left')
            if _el is not None and _em is not None:
                elbow_top_left = 0.5 * (_el + _em)

        if acrom_left is None or elbow_top_left is None:
            raise KeyError(
                "LeftUpperArm requires at least acrom_left and elbow_top_left"
            )

        O_allframes = np.zeros((3, n_frames))
        shoulder_center = np.zeros((3, n_frames))
        R_all = np.zeros((3, 3, n_frames))
        Zo = np.zeros((3, n_frames))

        for i in range(n_frames):
            shoulder_z = shoulder_left[2, i] if shoulder_left is not None else acrom_left[2, i]
            shoulder_center[:, i] = np.array([
                acrom_left[0, i],
                acrom_left[1, i],
                shoulder_z,
            ])
            O_allframes[:, i] = shoulder_center[:, i] + (elbow_top_left[:, i] - shoulder_center[:, i]) * 0.5772

            z = shoulder_center[:, i] - elbow_top_left[:, i]
            Zo[:, i] = z
            if z[2] > np.linalg.norm(z) * 0.5:
                help1 = np.array([0.0, 1.0, 0.0])
                x = np.cross(help1, z)
                y = np.cross(z, x)
            else:
                help1 = np.array([1.0, 0.0, 0.0])
                y = np.cross(z, help1)
                x = np.cross(y, z)

            x = _safe_normalize(x, np.array([1.0, 0.0, 0.0]))
            y = _safe_normalize(y, np.array([0.0, 1.0, 0.0]))
            z = _safe_normalize(z, np.array([0.0, 0.0, 1.0]))
            R_all[:, 0, i] = x
            R_all[:, 1, i] = y
            R_all[:, 2, i] = z

        O = np.mean(O_allframes, axis=1)
        R = Rmean(R_all)
        SLength = np.linalg.norm(np.mean(Zo, axis=1)) / 1000.0
        return O, R, SLength

    elif segment_name == 'RightLowerArm_regular':
        hand_med_right = markers_filtered.get('hand_med_right')
        elbow_top_right = markers_filtered.get('elbow_top_right')

        if elbow_top_right is None:   # markerless: elbow centre = midpoint of lat+med epicondyle
            _el = markers_filtered.get('elbow_lat_right'); _em = markers_filtered.get('elbow_med_right')
            if _el is not None and _em is not None:
                elbow_top_right = 0.5 * (_el + _em)
        if hand_med_right is None:    # markerless: wrist centre = midpoint of lat+med styloid
            _wl = markers_filtered.get('wrist_lat_right'); _wm = markers_filtered.get('wrist_med_right')
            if _wl is not None and _wm is not None:
                hand_med_right = 0.5 * (_wl + _wm)
            elif _wm is not None:
                hand_med_right = _wm
        if hand_med_right is None or elbow_top_right is None:
            raise KeyError("RightLowerArm requires at least hand_med_right and elbow_top_right")

        O_allframes = np.zeros((3, n_frames))
        R_all = np.zeros((3, 3, n_frames))
        Zo = np.zeros((3, n_frames))

        for i in range(n_frames):
            O_allframes[:, i] = elbow_top_right[:, i] + (hand_med_right[:, i] - elbow_top_right[:, i]) * 0.4574

            z = elbow_top_right[:, i] - hand_med_right[:, i]
            Zo[:, i] = z
            if z[2] > np.linalg.norm(z) * 0.5:
                help1 = np.array([0.0, 1.0, 0.0])
                x = np.cross(help1, z)
                y = np.cross(z, x)
            else:
                help1 = np.array([1.0, 0.0, 0.0])
                y = np.cross(z, help1)
                x = np.cross(y, z)

            x = _safe_normalize(x, np.array([1.0, 0.0, 0.0]))
            y = _safe_normalize(y, np.array([0.0, 1.0, 0.0]))
            z = _safe_normalize(z, np.array([0.0, 0.0, 1.0]))
            R_all[:, 0, i] = x
            R_all[:, 1, i] = y
            R_all[:, 2, i] = z

        O = np.mean(O_allframes, axis=1)
        R = Rmean(R_all)
        SLength = np.linalg.norm(np.mean(Zo, axis=1)) / 1000.0
        return O, R, SLength

    elif segment_name == 'LeftLowerArm_regular':
        hand_med_left = markers_filtered.get('hand_med_left')
        elbow_top_left = markers_filtered.get('elbow_top_left')

        if elbow_top_left is None:    # markerless: elbow centre = midpoint of lat+med epicondyle
            _el = markers_filtered.get('elbow_lat_left'); _em = markers_filtered.get('elbow_med_left')
            if _el is not None and _em is not None:
                elbow_top_left = 0.5 * (_el + _em)
        if hand_med_left is None:     # markerless: wrist centre = midpoint of lat+med styloid
            _wl = markers_filtered.get('wrist_lat_left'); _wm = markers_filtered.get('wrist_med_left')
            if _wl is not None and _wm is not None:
                hand_med_left = 0.5 * (_wl + _wm)
            elif _wm is not None:
                hand_med_left = _wm
        if hand_med_left is None or elbow_top_left is None:
            raise KeyError("LeftLowerArm requires at least hand_med_left and elbow_top_left")

        O_allframes = np.zeros((3, n_frames))
        R_all = np.zeros((3, 3, n_frames))
        Zo = np.zeros((3, n_frames))

        for i in range(n_frames):
            O_allframes[:, i] = elbow_top_left[:, i] + (hand_med_left[:, i] - elbow_top_left[:, i]) * 0.4574

            z = elbow_top_left[:, i] - hand_med_left[:, i]
            Zo[:, i] = z
            if z[2] > np.linalg.norm(z) * 0.5:
                help1 = np.array([0.0, 1.0, 0.0])
                x = np.cross(help1, z)
                y = np.cross(z, x)
            else:
                help1 = np.array([1.0, 0.0, 0.0])
                y = np.cross(z, help1)
                x = np.cross(y, z)

            x = _safe_normalize(x, np.array([1.0, 0.0, 0.0]))
            y = _safe_normalize(y, np.array([0.0, 1.0, 0.0]))
            z = _safe_normalize(z, np.array([0.0, 0.0, 1.0]))
            R_all[:, 0, i] = x
            R_all[:, 1, i] = y
            R_all[:, 2, i] = z

        O = np.mean(O_allframes, axis=1)
        R = Rmean(R_all)
        SLength = np.linalg.norm(np.mean(Zo, axis=1)) / 1000.0
        return O, R, SLength
    
    # ===================== FEET =====================
    
    elif segment_name == 'RightFoot_regular':
        calc_back = markers_filtered.get('calc_back_right')
        forfoot_lat = _marker_any(markers_filtered, 'forfoot_lat_right', 'forefoot_lat_right')
        forfoot_med = _marker_any(markers_filtered, 'forfoot_med_right', 'forefoot_med_right')

        # Whole-foot anatomical frame from calcaneus + metatarsal markers.
        # X: heel -> forefoot center (anterior), Y: lateral -> medial, Z: vertical.
        fore_center = 0.5 * (forfoot_med + forfoot_lat)
        O_allframes = (calc_back + forfoot_lat + forfoot_med) / 3.0
        O, R = _build_frame_xyz_from_points(
            O_allframes,
            p_x_from=calc_back,
            p_x_to=fore_center,
            p_y_from=forfoot_lat,
            p_y_to=forfoot_med,
        )

        foot_vec = fore_center - calc_back
        SLength = np.linalg.norm(np.mean(foot_vec, axis=1)) / 1000.0
        return O, R, SLength
    
    elif segment_name == 'RightForefoot_regular':
        toe = markers_filtered.get('toe_right')
        forfoot_lat = _marker_any(markers_filtered, 'forfoot_lat_right', 'forefoot_lat_right')
        forfoot_med = _marker_any(markers_filtered, 'forfoot_med_right', 'forefoot_med_right')

        # Forefoot frame from metatarsal pair + toe marker.
        fore_center = 0.5 * (forfoot_med + forfoot_lat)
        O_allframes = fore_center.copy()
        O, R = _build_frame_xyz_from_points(
            O_allframes,
            p_x_from=fore_center,
            p_x_to=toe,
            p_y_from=forfoot_lat,
            p_y_to=forfoot_med,
        )
        SLength = 0.0
        return O, R, SLength
    
    elif segment_name == 'RightRearfoot_regular':
        calc_back = markers_filtered.get('calc_back_right')
        calc_med = markers_filtered.get('calc_med_right')
        calc_lat = markers_filtered.get('calc_lat_right')

        # Rearfoot frame from calcaneus triad.
        calc_mid = 0.5 * (calc_med + calc_lat)
        O_allframes = (calc_back + calc_med + calc_lat) / 3.0
        O, R = _build_frame_xyz_from_points(
            O_allframes,
            p_x_from=calc_back,
            p_x_to=calc_mid,
            p_y_from=calc_lat,
            p_y_to=calc_med,
        )
        SLength = 0.0
        return O, R, SLength
    
    elif segment_name == 'LeftFoot_regular':
        calc_back = markers_filtered.get('calc_back_left')
        forefoot_lat = markers_filtered.get('forefoot_lat_left')
        forefoot_med = markers_filtered.get('forefoot_med_left')

        fore_center = 0.5 * (forefoot_med + forefoot_lat)
        O_allframes = (calc_back + forefoot_lat + forefoot_med) / 3.0
        # Left foot: the ML seed runs med -> lat (mirror of the right foot's lat -> med) so that
        # the derived vertical z = x_(heel->forefoot) x y_seed points up. With lat -> med on the
        # left side, y_seed would point to -Y, z would point down and the whole left foot frame
        # would be inverted by ~180 deg (ankle flexion and rotation offset by ~180 deg, inversion/
        # eversion sign-flipped). The mirrored seed keeps the left frame upright and consistent
        # with the right foot.
        O, R = _build_frame_xyz_from_points(
            O_allframes,
            p_x_from=calc_back,
            p_x_to=fore_center,
            p_y_from=forefoot_med,
            p_y_to=forefoot_lat,
        )

        foot_vec = fore_center - calc_back
        SLength = np.linalg.norm(np.mean(foot_vec, axis=1)) / 1000.0
        return O, R, SLength

    elif segment_name == 'LeftForefoot_regular':
        toe = markers_filtered.get('toe_left')
        forefoot_lat = markers_filtered.get('forefoot_lat_left')
        forefoot_med = markers_filtered.get('forefoot_med_left')

        fore_center = 0.5 * (forefoot_med + forefoot_lat)
        O_allframes = fore_center.copy()
        # Left foot: ML seed med -> lat (mirror of the right foot's lat -> med) so that
        # z = x x y_seed points up, as for RightForefoot and LeftFoot (see LeftFoot_regular).
        O, R = _build_frame_xyz_from_points(
            O_allframes,
            p_x_from=fore_center,
            p_x_to=toe,
            p_y_from=forefoot_med,
            p_y_to=forefoot_lat,
        )
        SLength = 0.0
        return O, R, SLength

    elif segment_name == 'LeftRearfoot_regular':
        calc_back = markers_filtered.get('calc_back_left')
        calc_med = markers_filtered.get('calc_med_left')
        calc_lat = markers_filtered.get('calc_lat_left')

        calc_mid = 0.5 * (calc_med + calc_lat)
        O_allframes = (calc_back + calc_med + calc_lat) / 3.0
        # Left foot: ML seed med -> lat (mirror of the right foot's lat -> med) so that
        # z = x x y_seed points up, as for RightRearfoot and LeftFoot (see LeftFoot_regular).
        O, R = _build_frame_xyz_from_points(
            O_allframes,
            p_x_from=calc_back,
            p_x_to=calc_mid,
            p_y_from=calc_med,
            p_y_to=calc_lat,
        )
        SLength = 0.0
        return O, R, SLength
    
    # ===================== TORSO (ISB Wu et al. 2005 adapted) =====================

    elif segment_name == 'Torso_regular':
        # Mandatory markers
        C_7       = markers_filtered.get('C_7')
        B_10      = markers_filtered.get('B_10')
        acrom_r   = markers_filtered.get('acrom_right')
        acrom_l   = markers_filtered.get('acrom_left')

        if any(m is None for m in [C_7, acrom_r, acrom_l]):
            raise KeyError(
                "Torso segment requires markers: C_7, acrom_right, acrom_left"
            )

        # Marker sets without a T10 marker (B_10), e.g. markerless keypoint sets, use a reduced
        # trunk frame whose long axis runs from the pelvis centre (inferior) to the shoulder-girdle
        # centre (superior). This gives trunk lean (Trunk_Pelvis) and a trunk length (for the CoM).
        # The axis roles (X anterior, Y left, Z superior) are identical to the full ISB frame
        # below, so Trunk_Pelvis stays consistent. Without pelvis markers, the shoulder-plane
        # normal is used (no trunk length).
        reduced = B_10 is None
        pelvis_c = None
        if reduced:
            _pm = [markers_filtered.get(k) for k in ('SIAS_left', 'SIAS_right', 'SIPS_left', 'SIPS_right')]
            _pm = [m for m in _pm if m is not None]
            if _pm:
                pelvis_c = sum(_pm) / len(_pm)        # (3, n) pelvis centre = inferior trunk landmark

        O_allframes = np.zeros((3, n_frames))
        R_all       = np.zeros((3, 3, n_frames))
        Zo_all      = np.zeros((3, n_frames))  # longitudinal vector for segment length

        for i in range(n_frames):
            c7 = C_7[:, i]
            ar  = acrom_r[:, i]
            al  = acrom_l[:, i]

            if not reduced:
                b10 = B_10[:, i]
                # Origin: midpoint of superior (C7) and inferior (T10/B10) spine landmarks
                O_allframes[:, i] = (c7 + b10) / 2.0
                # Z-axis: longitudinal/superior, from T10 toward C7
                z = c7 - b10
            else:
                sh_mid = (ar + al) / 2.0
                if pelvis_c is not None:
                    bottom = pelvis_c[:, i]
                    O_allframes[:, i] = (sh_mid + bottom) / 2.0
                    z = sh_mid - bottom               # superior = pelvis -> shoulder girdle (trunk long axis)
                else:
                    O_allframes[:, i] = sh_mid
                    z = np.cross(sh_mid - c7, al - ar)  # shoulder-plane normal (no trunk length available)
            Zo_all[:, i] = z
            z_n = np.linalg.norm(z)
            z = z / z_n if z_n > 1e-12 else np.array([0.0, 0.0, 1.0])

            # Mediolateral seed: left acromion minus right acromion (pointing left)
            y_approx = al - ar

            # X-axis: anterior = cross(y_approx, z), then normalize
            x = np.cross(y_approx, z)
            x_n = np.linalg.norm(x)
            x = x / x_n if x_n > 1e-12 else np.array([1.0, 0.0, 0.0])

            # Y-axis: corrected mediolateral (left) = cross(z, x)
            y = np.cross(z, x)
            y_n = np.linalg.norm(y)
            y = y / y_n if y_n > 1e-12 else np.array([0.0, 1.0, 0.0])

            # Re-orthogonalize z in case of numerical drift
            z = np.cross(x, y)
            z_n = np.linalg.norm(z)
            z = z / z_n if z_n > 1e-12 else z

            R_all[:, 0, i] = x
            R_all[:, 1, i] = y
            R_all[:, 2, i] = z

        O = np.mean(O_allframes, axis=1)
        R = Rmean(R_all)
        SLength = np.linalg.norm(np.mean(Zo_all, axis=1)) / 1000.0  # mm -> m

        return O, R, SLength

    else:
        raise ValueError(f"Unknown segment: {segment_name}")
