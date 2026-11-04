"""Newton-Euler inverse dynamics of the lower limb after Hof (1992, J Biomech 25:1209-1211).

Joint moments are computed distal to proximal (MPJ, ankle, knee, hip) from the
ground reaction force, the centre of pressure, the dynamic joint centres and the
segment kinematics resampled to the force-plate rate. Segment masses and radii of
gyration follow de Leva (1996, J Biomech 29:1223-1230). Positions are in mm, forces
in N, moments in N*m; the lab frame is Z-up.
"""

import numpy as np
from fractions import Fraction
from scipy.interpolate import interp1d
from scipy.signal import resample_poly


def _resample_1d_linear(row, n_target):
    n_src = row.shape[0]
    if n_src == n_target:
        return row.copy()
    x_src = np.linspace(0.0, 1.0, n_src)
    x_tgt = np.linspace(0.0, 1.0, n_target)
    return interp1d(
        x_src,
        row,
        kind='linear',
        bounds_error=False,
        fill_value=(row[0], row[-1]),
    )(x_tgt)


def _resample_1d_polyphase(row, n_target):
    row = np.asarray(row, dtype=float)
    n_src = row.shape[0]
    if n_src == n_target:
        return row.copy()

    finite = np.isfinite(row)
    if not np.any(finite):
        return np.full(n_target, np.nan, dtype=float)

    # Bridge NaN gaps by linear interpolation so that the FIR filter stays stable.
    filled = row.copy()
    if not np.all(finite):
        x = np.arange(n_src, dtype=float)
        filled[~finite] = np.interp(x[~finite], x[finite], row[finite])

    frac = Fraction(int(n_target), int(n_src))
    up, down = frac.numerator, frac.denominator
    out = resample_poly(filled, up=up, down=down, padtype='line')

    if out.shape[0] == n_target:
        return out

    # Rare length mismatch from FIR edge handling: map to the exact target grid.
    x_out = np.linspace(0.0, 1.0, out.shape[0])
    x_tgt = np.linspace(0.0, 1.0, n_target)
    return np.interp(x_tgt, x_out, out)


def _project_rotation_series_to_so3(rot_series):
    """Project each 3x3 frame to the closest proper rotation matrix.
    References: Moakher M (2002) SIAM J Matrix Anal Appl 24:1-16.
    """
    R = np.asarray(rot_series, dtype=float).copy()
    if R.ndim != 3 or R.shape[0] != 3 or R.shape[1] != 3:
        return R

    n = R.shape[2]
    for k in range(n):
        M = R[:, :, k]
        if not np.isfinite(M).all():
            continue
        u, _, vt = np.linalg.svd(M, full_matrices=False)
        Rk = u @ vt
        if np.linalg.det(Rk) < 0:
            u[:, -1] *= -1.0
            Rk = u @ vt
        R[:, :, k] = Rk
    return R


def resample_signal(signal, n_target, method='polyphase', project_rotation=False):
    """Resample to n_target samples.

    The default is polyphase FIR resampling (anti-aliasing). If that fails numerically,
    linear interpolation is used instead.

    Parameters
    ----------
    signal   : ndarray  shape (..., n_src), 1-D, 2-D or 3-D
    n_target : int

    Returns
    -------
    ndarray  same ndim, last dimension = n_target
    """
    signal = np.asarray(signal, dtype=float)
    n_src = signal.shape[-1]
    if n_src == n_target:
        out = signal.copy()
        return _project_rotation_series_to_so3(out) if project_rotation else out

    moved = np.moveaxis(signal, -1, 0)  # (n_src, ...)
    flat = moved.reshape(n_src, -1)
    out_flat = np.empty((n_target, flat.shape[1]), dtype=float)

    for col in range(flat.shape[1]):
        row = flat[:, col]
        try:
            if method == 'polyphase':
                out_flat[:, col] = _resample_1d_polyphase(row, n_target)
            else:
                out_flat[:, col] = _resample_1d_linear(row, n_target)
        except Exception:
            out_flat[:, col] = _resample_1d_linear(row, n_target)

    out = out_flat.reshape((n_target,) + moved.shape[1:])
    out = np.moveaxis(out, 0, -1)

    if project_rotation:
        out = _project_rotation_series_to_so3(out)

    if signal.ndim == 1:
        return out
    if signal.ndim == 2:
        return out
    if signal.ndim == 3:
        return out

    raise ValueError(f"resample_signal: ndim={signal.ndim} is not supported")


def _cross_cols(a, b):
    """Column-wise cross product: a (3, n) x b (3, n) -> (3, n)."""
    c = np.empty_like(a)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


def _batch_inertia_term(R, I_diag_vec, ACC, Theta, V):
    """Vectorized L = R I R^T and Term4 = L alpha + [omega]x L omega for all frames at once.

    Parameters
    ----------
    R          : (3, 3, n)  rotation matrices
    I_diag_vec : (3,)       principal MoI [Ix, Iy, Iz] in kg*m^2
    ACC        : (3, n)     angular acceleration in rad/s^2
    Theta      : (3, 3, n)  skew-symmetric [omega]x matrix
    V          : (3, n)     angular velocity in rad/s

    Returns
    -------
    LL    : (3, 3, n)  inertia tensor in the global frame, kg*m^2
    Term4 : (3, n)     rotational inertia contribution, N*m
    References: Hof AL (1992) J Biomech 25:1209-1211.; Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    # L_ijn = sum_k R_ikn * I_k * R_jkn   (R I R^T, batch over frame index n)
    LL = np.einsum('ikn,k,jkn->ijn', R, I_diag_vec, R)
    # L alpha
    La = np.einsum('ijn,jn->in', LL, ACC)
    # [omega]x (L omega)
    LV  = np.einsum('ijn,jn->in', LL, V)
    TLV = np.einsum('ijn,jn->in', Theta, LV)
    return LL, La + TLV


def _batch_project_moment(R_seg, Moment):
    """Project a moment from the global frame into the proximal segment frame for all frames.

    Parameters
    ----------
    R_seg  : (3, 3, n)  proximal segment rotation matrices
    Moment : (3, n)     moment in the global frame

    Returns
    -------
    (3, n)  moment in the proximal segment frame
    """
    # result_jn = sum_i R_seg[i,j,n] * Moment[i,n]  (equivalent to R^T m per frame)
    return np.einsum('ijn,in->jn', R_seg, Moment)


def compute_anthropometry(body_mass_kg, body_height_mm, segment_lengths=None, gender=1):
    """Compute segment masses and mass moments of inertia for the lower-limb segments.

    Mass fractions and radii of gyration after de Leva (1996).

    Parameters
    ----------
    body_mass_kg     : float   body mass [kg]
    body_height_mm   : float   body height [mm]
    segment_lengths  : dict, optional
        {'Foot': m, 'Shank': m, 'Thigh': m} in metres.
        If None or a value is 0: estimated from body height (Dempster fractions).
    gender           : int, optional
        0 = female, 1 = male (default: male)

    Returns
    -------
    anthro : dict
        {
          'SegmentMass': {'Foot': kg, 'Shank': kg, 'Thigh': kg},
          'MoI':  {
              'Foot':  {'X': kg*m^2, 'Y': kg*m^2, 'Z': kg*m^2},
              'Shank': {...},
              'Thigh': {...},
          }
        }
    """
    # de Leva (1996) segment mass fractions, by gender
    # Male parameters (default)
    MASS_FRAC_MALE = {'Foot': 0.0137, 'Shank': 0.0433, 'Thigh': 0.1416}
    # Female parameters (de Leva 1996)
    MASS_FRAC_FEMALE = {'Foot': 0.0129, 'Shank': 0.0481, 'Thigh': 0.1478}
    
    # Select gender-specific mass fractions
    MASS_FRAC = MASS_FRAC_FEMALE if gender == 0 else MASS_FRAC_MALE

    # de Leva (1996) radius-of-gyration factors per axis
    # Male parameters
    ROG_MALE = {
        'Foot':  [0.124, 0.245, 0.257],
        'Shank': [0.251, 0.246, 0.102],
        'Thigh': [0.329, 0.329, 0.149],
    }
    # Female parameters (de Leva 1996).
    # Axis order matches ROG_MALE: Thigh/Shank = [IXX, IYY, IZZ]; Foot is permuted to
    # [IZZ, IYY, IXX] because the foot segment frame's long axis is X (heel -> toe), not Z.
    ROG_FEMALE = {
        'Foot':  [0.139, 0.279, 0.299],   # de Leva IZZ, IYY, IXX = 0.139, 0.279, 0.299
        'Shank': [0.267, 0.263, 0.092],   # IXX, IYY, IZZ
        'Thigh': [0.369, 0.364, 0.162],   # IXX, IYY, IZZ
    }
    
    # Select gender-specific ROG
    ROG = ROG_FEMALE if gender == 0 else ROG_MALE

    # Segment length as a fraction of body height (Dempster estimate, fallback)
    LEN_FRAC = {'Foot': 0.152, 'Shank': 0.246, 'Thigh': 0.245}

    h_m = body_height_mm / 1000.0
    anthro = {'SegmentMass': {}, 'MoI': {}}

    for seg in ('Foot', 'Shank', 'Thigh'):
        mass = MASS_FRAC[seg] * body_mass_kg
        anthro['SegmentMass'][seg] = mass

        if segment_lengths and seg in segment_lengths and segment_lengths[seg] > 0:
            L = float(segment_lengths[seg])
        else:
            L = h_m * LEN_FRAC[seg]

        anthro['MoI'][seg] = {
            'X': mass * (ROG[seg][0] * L) ** 2,
            'Y': mass * (ROG[seg][1] * L) ** 2,
            'Z': mass * (ROG[seg][2] * L) ** 2,
        }

    return anthro


def _virtual_marker(jc_static_global, R_seg_static, O_seg_static,
                    R_seg_dyn, O_seg_dyn):
    """Reconstruct the dynamic trajectory of an anatomical landmark (virtual marker).

    Step 1: express jc_static in the static segment frame (local coordinates).
    Step 2: transform back with the dynamic segment frame for each frame.

    Parameters
    ----------
    jc_static_global : (3,)      static JC position in the global lab frame [mm]
    R_seg_static     : (3, 3)    static segment rotation
    O_seg_static     : (3,)      static segment origin [mm]
    R_seg_dyn        : (3, 3, n) dynamic segment rotations
    O_seg_dyn        : (3, n)    dynamic segment origins [mm]

    Returns
    -------
    jc_dyn : (3, n)   dynamic JC trajectory [mm]
    """
    jc_local = R_seg_static.T @ (jc_static_global - O_seg_static)
    # Vectorized: jc_dyn[:, n] = O[:, n] + R[:, :, n] @ jc_local  for all n at once.
    jc_dyn = O_seg_dyn + np.einsum('ijn,j->in', R_seg_dyn, jc_local)
    return jc_dyn


def compute_dynamic_jc(frames_anat_dyn, REFFRAME, static_jc, leg='right'):
    """Compute dynamic joint-centre trajectories (virtual marker technique).

    For each joint, the JC expressed in the static segment frame is transformed back
    into the global frame with the dynamic segment pose.

    Parameters
    ----------
    frames_anat_dyn : dict
        Dynamic anatomical frames per segment:
          {seg: {'O': (3,n) mm,  'R': (3,3,n)}}
    REFFRAME : dict
        Result of calibrate_static_lower_limb().
        Requires REFFRAME['Anatomical'][seg]['R'] (3,3) and ['O'] (3,).
    static_jc : dict
        From compute_all_joint_centers():
          {'AJC_right': (3,), 'KJC_right': (3,), 'HJC_right': (3,),
           'AJC_left':  (3,), 'KJC_left':  (3,), 'HJC_left':  (3,)}
    leg : str   'right' or 'left'

    Returns
    -------
    dict
        {'AnkleJC': (3,n_kin) mm,
         'KneeJC':  (3,n_kin) mm,
         'HipJC':   (3,n_kin) mm}
    References: Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Clin Biomech 10:171-178.
    """
    s = 'right' if leg.lower() == 'right' else 'left'
    S = s.capitalize()
    anat = REFFRAME.get('Anatomical', {})

    # (JC output key, static_jc key, segment in which the JC is expressed)
    defs = [
        ('AnkleJC', f'AJC_{s}', f'{S}Shank'),
        ('KneeJC',  f'KJC_{s}', f'{S}Thigh'),
        ('HipJC',   f'HJC_{s}', f'{S}Pelvis'),
    ]

    jc_out = {}
    for jc_key_out, jc_key_in, seg in defs:
        if jc_key_in not in static_jc:
            print(f"  compute_dynamic_jc: '{jc_key_in}' missing in static_jc, skipped")
            continue
        if seg not in anat:
            print(f"  compute_dynamic_jc: Segment '{seg}' missing in REFFRAME['Anatomical'], skipped")
            continue
        if seg not in frames_anat_dyn:
            print(f"  compute_dynamic_jc: Segment '{seg}' missing in frames_anat_dyn, skipped")
            continue

        R_st = anat[seg]['R']
        O_st = anat[seg]['O']

        # If R/O carry several frames (3-D / 2-D), use the mean
        if R_st.ndim == 3:
            R_st = R_st.mean(axis=2)
        if O_st.ndim == 2:
            O_st = O_st.mean(axis=1)

        jc_out[jc_key_out] = _virtual_marker(
            static_jc[jc_key_in], R_st, O_st,
            frames_anat_dyn[seg]['R'],
            frames_anat_dyn[seg]['O'],
        )

    return jc_out


def inverse_dynamics_hof(frames_anat_dyn, fp_data, jc_dyn_kin, anthro, leg='right', fp_moment_unit='Nmm'):
    """Newton-Euler inverse dynamics after Hof (1992) for one lower limb.

    Computes joint moments at the MPJ, ankle, knee and hip (distal -> proximal).

    Parameters
    ----------
    frames_anat_dyn : dict
        Dynamic anatomical frames (e.g. frames_dynamic['Anatomical'] from
        reconstruct_kinematics). Expected per segment:
          {seg: {'O':      (3, n_kin)    mm,
                 'R':      (3, 3, n_kin),
                 'V':      (3, n_kin)    rad/s,
                 'ACC':    (3, n_kin)    rad/s^2,
                 'LinACC': (3, n_kin)    m/s^2,
                 'Theta':  (3, 3, n_kin) rad/s  (skew-symmetric)}}
        Required segments: RightRearfoot/RightFoot, RightShank, RightThigh,
                           RightPelvis (left side analogous).

    fp_data : dict
        Force-plate data (see markersub.kinetics):
          {'cop':    (3, n_fp) mm,
           'force':  (3, n_fp) N,
           'moment': (3, n_fp) N*mm, converted internally to N*m (/1000)
           'sampling_factor': int, ...}

    jc_dyn_kin : dict
        Dynamic joint centres at the kinematic rate (from compute_dynamic_jc()):
          {'AnkleJC': (3, n_kin) mm,
           'KneeJC':  (3, n_kin) mm,
           'HipJC':   (3, n_kin) mm}

    anthro : dict
        From compute_anthropometry():
          {'SegmentMass': {'Foot': kg, 'Shank': kg, 'Thigh': kg},
           'MoI': {'Foot': {'X','Y','Z'} kg*m^2, ...}}

    leg : str
        'right' or 'left'

    fp_moment_unit : str
        Unit of fp_data['moment']:
          - 'Nmm' (default): divided by 1000 internally
          - 'Nm' : used directly

    Returns
    -------
    KINETICS : dict
        Main results (N*m) and intermediate terms:
          'AnkleMomentInProximal' : (3, n_fp)  ankle moment in the shank frame
          'KneeMomentInProximal'  : (3, n_fp)  knee moment in the thigh frame
          'HipMomentInProximal'   : (3, n_fp)  hip moment in the pelvis frame
          'Ankle' / 'Knee' / 'Hip': dicts with Term1..Term4 and Moment (global)

    LL : dict
        Inertia tensors in the global frame:
          '{S}Foot'/'Shank'/'Thigh' : (3, 3, n_fp)

    Notes
    -----
    Sign convention for the left side:
      the X and Z components of the projected moments are inverted so that the
      axes have the same interpretation on both sides.

    The ankle step uses the Foot segment (RightFoot/LeftFoot), not the Rearfoot.
    """
    s = leg.lower()
    S = s.capitalize()

    n_fp = fp_data['cop'].shape[1]

    # Gravity vector (Z-up coordinate system)
    g_vec = np.array([0.0, 0.0, -9.81])   # m/s^2

    KINETICS = {}
    LL       = {}

    # Segment keys (the ankle step uses Foot, not Rearfoot)
    foot_seg  = f'{S}Foot'
    rear_seg  = f'{S}Rearfoot'
    shank_seg = f'{S}Shank'
    thigh_seg = f'{S}Thigh'
    pelv_seg  = f'{S}Pelvis'

    if foot_seg not in frames_anat_dyn:
        raise KeyError(f"Segment '{foot_seg}' missing in frames_anat_dyn")

    # --- Force-plate data (already at force-plate rate) ---
    COP = fp_data['cop']                  # (3, n_fp) mm
    GRF = fp_data['force']                # (3, n_fp) N
    if fp_moment_unit == 'Nmm':
        FM = fp_data['moment'] / 1000.0    # (3, n_fp) N*mm -> N*m
    elif fp_moment_unit == 'Nm':
        FM = fp_data['moment']             # already N*m
    else:
        raise ValueError("fp_moment_unit must be 'Nmm' or 'Nm'")

    # Pre-compute GRF magnitude for lever arm calculations (all joints share same GRF)
    _grf_mag = np.linalg.norm(GRF, axis=0)                                  # (n_fp,)
    _grf_mag_safe = np.where(_grf_mag > 50.0, _grf_mag, np.nan)             # mask low-force frames

    # --- Resampling helpers ---
    def rs_linear(x):
        return resample_signal(x, n_fp, method='polyphase', project_rotation=False)

    def rs_rotation(x):
        return resample_signal(x, n_fp, method='polyphase', project_rotation=True)

    # =====================================================================
    # MPJ (MTP) joint (rearfoot frame)
    # =====================================================================
    try:
        if 'MPJC' not in jc_dyn_kin:
            raise KeyError("'MPJC' missing in jc_dyn_kin")
        if rear_seg not in frames_anat_dyn:
            raise KeyError(f"Segment '{rear_seg}' missing in frames_anat_dyn")

        MPJC = rs_linear(jc_dyn_kin['MPJC'])

        Term1_mpj = _cross_cols(COP - MPJC, GRF) / 1000.0
        # MPJ simplified: only Term1 (no gravity or inertia term). The forefoot mass is
        # about 1.4% of body mass, so the error is below 2% of the ankle moment.
        # Sign convention: MPJMom[1] is inverted below (*= -1.0) to give the external
        # moment. The ankle, knee and hip moments are returned as internal moments; their
        # sign change to external moments is applied by the caller.
        Moment_mpj = Term1_mpj

        R_rear_fp = rs_rotation(frames_anat_dyn[rear_seg]['R'])
        MPJMom = _batch_project_moment(R_rear_fp, Moment_mpj)

        if s == 'left':
            MPJMom[0, :] *= -1.0
            MPJMom[2, :] *= -1.0

        # Sagittal-only output for MPJ: X/Z are set to zero (single-DOF joint).
        # Y is inverted for the external moment convention (plantarflexion positive).
        MPJMom[0, :] = 0.0
        MPJMom[1, :] *= -1.0  # invert Y for external moments
        MPJMom[2, :] = 0.0

        KINETICS['MPJ'] = {
            'Term1': Term1_mpj,
            'Moment': Moment_mpj,
        }
        KINETICS['MPJMomentInProximal'] = MPJMom

        print(f"  MPJ joint    [{S}]: OK")

    except (KeyError, ValueError) as e:
        print(f"  MPJ joint    [{S}]: error: {e}")

    # =====================================================================
    # Ankle joint (Foot segment)
    # =====================================================================
    try:
        seg_data = frames_anat_dyn[foot_seg]
        m_foot   = anthro['SegmentMass']['Foot']
        I_foot   = np.diag([anthro['MoI']['Foot']['X'],
                             anthro['MoI']['Foot']['Y'],
                             anthro['MoI']['Foot']['Z']])   # kg*m^2

        # Resample kinematics to the force-plate rate
        O_foot      = rs_linear(seg_data['O'])       # (3, n_fp) mm
        R_foot      = rs_rotation(seg_data['R'])     # (3, 3, n_fp)
        V_foot      = rs_linear(seg_data['V'])       # (3, n_fp) rad/s
        ACC_foot    = rs_linear(seg_data['ACC'])     # (3, n_fp) rad/s^2
        La_foot     = rs_linear(seg_data['LinACC'])  # (3, n_fp) m/s^2
        Theta_foot  = rs_linear(seg_data['Theta'])   # (3, 3, n_fp)

        AnkleJC    = rs_linear(jc_dyn_kin['AnkleJC'])  # (3, n_fp) mm

        # Linear inertial force and weight
        Weight_foot = np.tile((m_foot * g_vec).reshape(3, 1), (1, n_fp))   # (3, n_fp) N
        MA_foot     = m_foot * La_foot                                       # (3, n_fp) N

        # Terms (N*m after division by 1000, because positions are in mm)
        Term1_ank = _cross_cols(COP    - AnkleJC, GRF)         / 1000.0  # GRF lever arm
        Term2_ank = _cross_cols(O_foot - AnkleJC, Weight_foot) / 1000.0  # gravity
        Term3_ank = _cross_cols(O_foot - AnkleJC, MA_foot)     / 1000.0  # linear inertia

        # Inertia tensor in the global frame and Term4 (rotational inertia)
        LL_foot, Term4_ank = _batch_inertia_term(
            R_foot, np.diag(I_foot), ACC_foot, Theta_foot, V_foot
        )
        LL[f'{S}Foot'] = LL_foot

        # Moment balance (Hof 1992, eq. 6)
        Moment_ank = -FM - Term1_ank - Term2_ank + Term3_ank + Term4_ank

        # Projection into the shank frame (proximal)
        R_shank_fp = rs_rotation(frames_anat_dyn[shank_seg]['R'])
        AnkleMom   = _batch_project_moment(R_shank_fp, Moment_ank)

        # Lever arm: signed perpendicular distance COP-JC per axis [mm]
        LeverArm_ank_mm = Term1_ank * 1000.0 / _grf_mag_safe[np.newaxis, :]  # (3, n_fp)

        KINETICS['Ankle'] = {
            'Term1': Term1_ank, 'Term2': Term2_ank,
            'Term3': Term3_ank, 'Term4': Term4_ank,
            'Moment': Moment_ank,
            # segment primitives kept so that proximal joints can re-reference this segment's
            # gravity/linear-inertia moment arm to their joint centre (recursive Newton-Euler transfer).
            'O': O_foot, 'Weight': Weight_foot, 'MA': MA_foot,
        }
        KINETICS['AnkleMomentInProximal'] = AnkleMom.copy()
        KINETICS['AnkleLeverArm_mm'] = LeverArm_ank_mm.copy()

        # Left side: invert X/Z (biomechanical sign convention)
        if s == 'left':
            KINETICS['AnkleMomentInProximal'][0, :] *= -1.0
            KINETICS['AnkleMomentInProximal'][2, :] *= -1.0

        print(f"  Sprunggelenk [{S}]: OK  (n_fp={n_fp})")

    except (KeyError, ValueError) as e:
        print(f"  Ankle joint  [{S}]: error: {e}")

    # =====================================================================
    # Knee joint (Shank segment)
    # =====================================================================
    try:
        seg_data = frames_anat_dyn[shank_seg]
        m_shank  = anthro['SegmentMass']['Shank']
        I_shank  = np.diag([anthro['MoI']['Shank']['X'],
                             anthro['MoI']['Shank']['Y'],
                             anthro['MoI']['Shank']['Z']])

        O_shank     = rs_linear(seg_data['O'])
        R_shank     = rs_rotation(seg_data['R'])
        V_shank     = rs_linear(seg_data['V'])
        ACC_shank   = rs_linear(seg_data['ACC'])
        La_shank    = rs_linear(seg_data['LinACC'])
        Theta_shank = rs_linear(seg_data['Theta'])

        KneeJC = rs_linear(jc_dyn_kin['KneeJC'])

        Weight_shank = np.tile((m_shank * g_vec).reshape(3, 1), (1, n_fp))
        MA_shank     = m_shank * La_shank

        Term1_kne = _cross_cols(COP     - KneeJC, GRF)          / 1000.0
        Term2_kne = _cross_cols(O_shank - KneeJC, Weight_shank)  / 1000.0
        Term3_kne = _cross_cols(O_shank - KneeJC, MA_shank)      / 1000.0

        LL_shank, Term4_kne = _batch_inertia_term(
            R_shank, np.diag(I_shank), ACC_shank, Theta_shank, V_shank
        )
        LL[f'{S}Shank'] = LL_shank

        # Distal segments are accumulated with their gravity/linear-inertia moment arms
        # re-referenced to the knee joint centre (recursive Newton-Euler transfer term, which
        # includes (AnkleJC - KneeJC) x F_foot). Term4 (rotational-inertia couple) is independent
        # of the reference point and is summed directly; Term1 (GRF) is computed per joint.
        ank = KINETICS['Ankle']
        Term2_foot_kne = _cross_cols(ank['O'] - KneeJC, ank['Weight']) / 1000.0
        Term3_foot_kne = _cross_cols(ank['O'] - KneeJC, ank['MA'])     / 1000.0
        Moment_kne = (
            -FM - Term1_kne
            - (Term2_kne + Term2_foot_kne)
            + (Term3_kne + Term3_foot_kne)
            + (Term4_kne + ank['Term4'])
        )

        R_thigh_fp = rs_rotation(frames_anat_dyn[thigh_seg]['R'])
        KneeMom    = _batch_project_moment(R_thigh_fp, Moment_kne)

        # Lever arm: signed perpendicular distance COP-KJC per axis [mm]
        LeverArm_kne_mm = Term1_kne * 1000.0 / _grf_mag_safe[np.newaxis, :]  # (3, n_fp)

        KINETICS['Knee'] = {
            'Term1': Term1_kne, 'Term2': Term2_kne,
            'Term3': Term3_kne, 'Term4': Term4_kne,
            'Moment': Moment_kne,
            'O': O_shank, 'Weight': Weight_shank, 'MA': MA_shank,   # for hip re-referencing
        }
        KINETICS['KneeMomentInProximal'] = KneeMom.copy()
        KINETICS['KneeLeverArm_mm'] = LeverArm_kne_mm.copy()

        if s == 'left':
            KINETICS['KneeMomentInProximal'][0, :] *= -1.0
            KINETICS['KneeMomentInProximal'][2, :] *= -1.0

        print(f"  Kniegelenk   [{S}]: OK")

    except (KeyError, ValueError) as e:
        print(f"  Knee joint   [{S}]: error: {e}")

    # =====================================================================
    # Hip joint (Thigh segment)
    # =====================================================================
    try:
        seg_data = frames_anat_dyn[thigh_seg]
        m_thigh  = anthro['SegmentMass']['Thigh']
        I_thigh  = np.diag([anthro['MoI']['Thigh']['X'],
                             anthro['MoI']['Thigh']['Y'],
                             anthro['MoI']['Thigh']['Z']])

        O_thigh     = rs_linear(seg_data['O'])
        R_thigh     = rs_rotation(seg_data['R'])
        V_thigh     = rs_linear(seg_data['V'])
        ACC_thigh   = rs_linear(seg_data['ACC'])
        La_thigh    = rs_linear(seg_data['LinACC'])
        Theta_thigh = rs_linear(seg_data['Theta'])

        HipJC = rs_linear(jc_dyn_kin['HipJC'])

        Weight_thigh = np.tile((m_thigh * g_vec).reshape(3, 1), (1, n_fp))
        MA_thigh     = m_thigh * La_thigh

        Term1_hip = _cross_cols(COP     - HipJC, GRF)           / 1000.0
        Term2_hip = _cross_cols(O_thigh - HipJC, Weight_thigh)   / 1000.0
        Term3_hip = _cross_cols(O_thigh - HipJC, MA_thigh)       / 1000.0

        LL_thigh, Term4_hip = _batch_inertia_term(
            R_thigh, np.diag(I_thigh), ACC_thigh, Theta_thigh, V_thigh
        )
        LL[f'{S}Thigh'] = LL_thigh

        # Distal segments (shank + foot) re-referenced to the hip joint centre for their
        # gravity/linear-inertia moment arms (recursive Newton-Euler transfer). Term4 couples are summed directly.
        ank = KINETICS['Ankle']
        kne = KINETICS['Knee']
        Term2_foot_hip  = _cross_cols(ank['O'] - HipJC, ank['Weight']) / 1000.0
        Term3_foot_hip  = _cross_cols(ank['O'] - HipJC, ank['MA'])     / 1000.0
        Term2_shank_hip = _cross_cols(kne['O'] - HipJC, kne['Weight']) / 1000.0
        Term3_shank_hip = _cross_cols(kne['O'] - HipJC, kne['MA'])     / 1000.0
        Moment_hip = (
            -FM - Term1_hip
            - (Term2_hip + Term2_shank_hip + Term2_foot_hip)
            + (Term3_hip + Term3_shank_hip + Term3_foot_hip)
            + (Term4_hip + kne['Term4'] + ank['Term4'])
        )

        R_pelv_fp = rs_rotation(frames_anat_dyn[pelv_seg]['R'])
        HipMom    = _batch_project_moment(R_pelv_fp, Moment_hip)

        # Lever arm: signed perpendicular distance COP-HJC per axis [mm]
        LeverArm_hip_mm = Term1_hip * 1000.0 / _grf_mag_safe[np.newaxis, :]  # (3, n_fp)

        KINETICS['Hip'] = {
            'Term1': Term1_hip, 'Term2': Term2_hip,
            'Term3': Term3_hip, 'Term4': Term4_hip,
            'Moment': Moment_hip,
        }
        KINETICS['HipMomentInProximal'] = HipMom.copy()
        KINETICS['HipLeverArm_mm'] = LeverArm_hip_mm.copy()

        if s == 'left':
            KINETICS['HipMomentInProximal'][0, :] *= -1.0
            KINETICS['HipMomentInProximal'][2, :] *= -1.0

        print(f"  Hip joint    [{S}]: OK")

    except (KeyError, ValueError) as e:
        print(f"  Hip joint    [{S}]: error: {e}")

    return KINETICS, LL
