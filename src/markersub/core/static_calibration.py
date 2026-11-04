"""Static calibration of the segment model (technical to anatomical frames).

The static marker trajectories are low-pass filtered (zero-lag Butterworth), joint
centres are computed (see joint_centers), technical frames are built from marker
clusters and anatomical frames from anatomical landmarks. The constant
technical-to-anatomical rotation and offset of each segment is the mean over the
static frames (calibrated anatomical systems technique, Cappozzo et al. 1995,
Clin Biomech). Positions are in mm.
"""

import numpy as np
from scipy.signal import butter
from .segment_frames import (
    markersfilt, get_technical_frame, getanaframe,
    Rmean, Vmean,
)
from .joint_centers import compute_all_joint_centers


def calibrate_static_lower_limb(markers_raw, labels, freq_kinematics, cut_off_markers,
                                 leg_length_right=None, leg_length_left=None,
                                 anthro=None, static_joint_center_source='filtered',
                                 static_hip_center_source='pelvis_origin'):
    """Perform the static calibration of the lower-limb (and upper-body) segments.

    Parameters
    ----------
    markers_raw : ndarray
        Raw marker trajectories, shape (n_markers, 3, n_frames)
    labels : list
        Marker labels
    freq_kinematics : float
        Sampling frequency (Hz)
    cut_off_markers : float
        Cutoff frequency for the Butterworth filter (Hz)
    leg_length_right, leg_length_left : float, optional
        Manual leg lengths for Harrington (if None, computed from markers)
    anthro : dict, optional
        Anthropometric data
    static_joint_center_source : str, optional
        Marker input for the static joint centres: 'filtered' (default, uses the
        filtered static marker trajectories) or 'raw'.
    static_hip_center_source : str, optional
        Source of the static hip joint centres used by the thigh anatomical frame:
        - 'pelvis_origin' (default): origin of the side-specific pelvis anatomical frame
        - 'harrington': Harrington regression from compute_all_joint_centers

    Returns
    -------
    REFFRAME : dict
        Reference frames with Technical_to_Anatomical calibrations
    MARKERS : dict
        Filtered marker trajectories and derived joint centers
    technical_frames : dict
        Technical frames for all segments
    anatomical_frames : dict
        Anatomical frames for all segments
    References: Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Clin Biomech 10:171-178.; Harrington ME et al. (2007) J Biomech 40:595-602.; Wu G et al. (2002) J Biomech 35:543-548.
    """
    
    if anthro is None:
        anthro = {'FootLength': 250}
    
    # ========== FILTER MARKERS ==========
    # Design Butterworth filter
    nyquist = freq_kinematics / 2.0
    normalized_cutoff = cut_off_markers / nyquist
    b, a = butter(4, normalized_cutoff, btype='low', analog=False)
    
    # Build marker dictionary
    markers_raw_dict = {}
    for i, label in enumerate(labels):
        # Take only first 3 dimensions (x,y,z), skip confidence if present
        markers_raw_dict[label] = markers_raw[i, :3, :]
    
    # Filter each marker
    markers_filt_dict = {}
    for label, raw_data in markers_raw_dict.items():
        markers_filt_dict[label] = markersfilt(raw_data, b, a)
    
    # ========== COMPUTE HIP/KNEE/ANKLE CENTERS ==========
    # By default the joint centres use the filtered static marker trajectories.
    labels = list(labels)
    markers_filt_arr = np.zeros((len(labels), 3, markers_raw.shape[2]), dtype=float)
    for i, label in enumerate(labels):
        markers_filt_arr[i, :, :] = markers_filt_dict[label]

    jc_source = str(static_joint_center_source).strip().lower()
    if jc_source not in {'filtered', 'raw'}:
        raise ValueError("static_joint_center_source must be 'filtered' or 'raw'")

    hip_source = str(static_hip_center_source).strip().lower()
    if hip_source not in {'pelvis_origin', 'harrington'}:
        raise ValueError("static_hip_center_source must be 'pelvis_origin' or 'harrington'")

    jc_input = markers_filt_arr if jc_source == 'filtered' else markers_raw[:, :3, :]
    hip_centers = compute_all_joint_centers(
        jc_input,
        labels,
        frame=None,
    )
    
    # Add hip centers to filtered markers (Harrington mode)
    n_frames = markers_raw.shape[2]
    if hip_source == 'harrington':
        for side in ['left', 'right']:
            hip_key = f'hip_joint_center_{side}'
            hjc_name = f'HJC_{side}'
            if hjc_name in hip_centers:
                markers_filt_dict[hip_key] = np.tile(hip_centers[hjc_name].reshape(3, 1), (1, n_frames))
    
    # ========== BUILD TECHNICAL FRAMES ==========
    technical_frames = {}

    def _try_add_technical_frame(seg_name, marker_names):
        available = [markers_filt_dict[mn] for mn in marker_names if mn in markers_filt_dict]
        if len(available) < 3:
            return
        if len(available) < len(marker_names):
            missing = [mn for mn in marker_names if mn not in markers_filt_dict]
            print(f"  [WARN] {seg_name}: marker(s) missing in static ({', '.join(missing)}), using {len(available)}/{len(marker_names)}")
        # get_technical_frame takes 3 or 4 markers; the first <=4 available markers (in list
        # order) are used. A marker list may therefore carry more than 4 candidates, and each
        # marker set uses the first 4 it provides.
        tech_o, tech_r = get_technical_frame(*available[:4])
        technical_frames[seg_name] = {'O': tech_o, 'R': tech_r}
    
    # Right leg
    _try_add_technical_frame('RightThigh', [
        'cluster_femur_right_1',
        'cluster_femur_right_2',
        'cluster_femur_right_3',
        'cluster_femur_right_4',
    ])

    _try_add_technical_frame('RightShank', [
        'cluster_tibia_right_1',
        'cluster_tibia_right_2',
        'cluster_tibia_right_3',
        'cluster_tibia_right_4',
    ])

    # Pelvis technical marker candidates (the first <=4 available are used, in this order).
    # With only SIAS/SIPS present: [SIAS_R, SIAS_L, SIPS_R, SIPS_L].
    # With sacrum and pelvis cluster markers: [SIAS_R, SIAS_L, sacrum, cluster_pelvis_right],
    # matching the dynamic segment configuration so that Technical->Anatomical stays consistent.
    _try_add_technical_frame('RightPelvis', [
        'SIAS_right',
        'SIAS_left',
        'sacrum',
        'cluster_pelvis_right',
        'cluster_pelvis_left',
        'SIPS_right',
        'SIPS_left',
    ])
    
    # Left leg
    _try_add_technical_frame('LeftThigh', [
        'cluster_femur_left_1',
        'cluster_femur_left_2',
        'cluster_femur_left_3',
        'cluster_femur_left_4',
    ])

    _try_add_technical_frame('LeftShank', [
        'cluster_tibia_left_1',
        'cluster_tibia_left_2',
        'cluster_tibia_left_3',
        'cluster_tibia_left_4',
    ])

    _try_add_technical_frame('LeftPelvis', [   # same candidate order as RightPelvis
        'SIAS_right',
        'SIAS_left',
        'sacrum',
        'cluster_pelvis_right',
        'cluster_pelvis_left',
        'SIPS_right',
        'SIPS_left',
    ])

    # Torso (4-marker cluster: 2 spine + 2 acromia)
    _try_add_technical_frame('Torso', [
        'C_7',
        'B_10',
        'acrom_right',
        'acrom_left',
    ])

    # Head technical candidates (first <=4 available, in order). A 4-marker head set uses the
    # front/back markers; a 3-marker head cluster (front + left/right) gives
    # [head_front, head_left, head_right]. Candidates that exist only in markerless marker
    # sets are appended last, so marker-based sets always use their head markers and
    # markerless sets fall through to their own 3 markers.
    _try_add_technical_frame('Head', [
        'head_front_right',
        'head_front_left',
        'head_back_right',
        'head_back_left',
        'head_front',
        'head_left',
        'head_right',
        'head_center', 'nose', 'neck',          # markerless keypoint set
    ])

    _try_add_technical_frame('RightUpperArm', [
        'cluster_upperarm_right_1',
        'cluster_upperarm_right_2',
        'cluster_upperarm_right_3',
        'shoulder_ml_right', 'elbow_lat_right', 'elbow_med_right',   # markerless
    ])

    _try_add_technical_frame('LeftUpperArm', [
        'cluster_upperarm_left_1',
        'cluster_upperarm_left_2',
        'cluster_upperarm_left_3',
        'shoulder_ml_left', 'elbow_lat_left', 'elbow_med_left',      # markerless
    ])

    _try_add_technical_frame('RightLowerArm', [
        'cluster_lowerarm_right_1',
        'cluster_lowerarm_right_2',
        'cluster_lowerarm_right_3',
        'elbow_lat_right', 'wrist_lat_right', 'wrist_med_right',     # markerless
    ])

    _try_add_technical_frame('LeftLowerArm', [
        'cluster_lowerarm_left_1',
        'cluster_lowerarm_left_2',
        'cluster_lowerarm_left_3',
        'elbow_lat_left', 'wrist_lat_left', 'wrist_med_left',        # markerless
    ])

    # Rearfoot (tracked by anatomical calc markers directly)
    _try_add_technical_frame('RightRearfoot', [
        'calc_back_right',
        'calc_med_right',
        'calc_lat_right',
    ])

    _try_add_technical_frame('RightForefoot', [
        'toe_right',
        'forfoot_lat_right',
        'forfoot_med_right',
    ])
    _try_add_technical_frame('RightForefoot', [
        'toe_right',
        'forefoot_lat_right',
        'forefoot_med_right',
    ])

    _try_add_technical_frame('RightFoot', [
        'calc_back_right',
        'forfoot_lat_right',
        'forfoot_med_right',
    ])

    _try_add_technical_frame('LeftRearfoot', [
        'calc_back_left',
        'calc_med_left',
        'calc_lat_left',
    ])

    _try_add_technical_frame('LeftForefoot', [
        'toe_left',
        'forefoot_lat_left',
        'forefoot_med_left',
    ])
    _try_add_technical_frame('LeftForefoot', [
        'toe_left',
        'forfoot_lat_left',
        'forfoot_med_left',
    ])

    _try_add_technical_frame('LeftFoot', [
        'calc_back_left',
        'forefoot_lat_left',
        'forefoot_med_left',
    ])
    
    # ========== BUILD ANATOMICAL FRAMES ==========
    anatomical_frames = {}
    
    segment_definitions = [
        ('RightPelvis_regular', 'RightPelvis'),
        ('RightThigh_regular', 'RightThigh'),
        ('RightShank_regular', 'RightShank'),
        ('RightRearfoot_regular', 'RightRearfoot'),
        ('RightForefoot_regular', 'RightForefoot'),
        ('RightFoot_regular', 'RightFoot'),
        ('Torso_regular', 'Torso'),
        ('Head_regular', 'Head'),
        ('RightUpperArm_regular', 'RightUpperArm'),
        ('RightLowerArm_regular', 'RightLowerArm'),
        ('LeftUpperArm_regular', 'LeftUpperArm'),
        ('LeftLowerArm_regular', 'LeftLowerArm'),
        ('LeftPelvis_regular', 'LeftPelvis'),
        ('LeftThigh_regular', 'LeftThigh'),
        ('LeftShank_regular', 'LeftShank'),
        ('LeftRearfoot_regular', 'LeftRearfoot'),
        ('LeftForefoot_regular', 'LeftForefoot'),
        ('LeftFoot_regular', 'LeftFoot'),
    ]
    
    for seg_name, seg_key in segment_definitions:
        try:
            O, R, SLength = getanaframe(markers_filt_dict, seg_name, anthro)
            anatomical_frames[seg_key] = {'O': O, 'R': R, 'SLength': SLength}

            # 'pelvis_origin' mode: the side-specific pelvis anatomical origin is used as hip
            # joint centre for the thigh frame.
            if hip_source == 'pelvis_origin' and seg_key in {'RightPelvis', 'LeftPelvis'}:
                side = 'right' if seg_key.startswith('Right') else 'left'
                markers_filt_dict[f'hip_joint_center_{side}'] = np.tile(
                    np.asarray(O, dtype=float).reshape(3, 1),
                    (1, n_frames),
                )
                hip_centers[f'HJC_{side}'] = np.asarray(O, dtype=float).reshape(3)
        except Exception as e:
            print(f"Warning: Skipping {seg_key} due to missing/invalid markers: {e}")
    
    # ========== COMPUTE TECHNICAL-TO-ANATOMICAL CALIBRATION ==========
    REFFRAME = {
        'Technical_to_Anatomical': {},
        'Anatomical': anatomical_frames,
    }
    
    # Upper body segments
    for tech_key, seg_key in [
        ('Torso', 'Torso'),
        ('Head', 'Head'),
        ('RightUpperArm', 'RightUpperArm'),
        ('RightLowerArm', 'RightLowerArm'),
        ('LeftUpperArm', 'LeftUpperArm'),
        ('LeftLowerArm', 'LeftLowerArm'),
    ]:
        if tech_key in technical_frames and seg_key in anatomical_frames:
            R_tech = technical_frames[tech_key]['R']
            O_tech = technical_frames[tech_key]['O']
            R_anat = anatomical_frames[seg_key]['R']
            O_anat = anatomical_frames[seg_key]['O']

            if R_anat.ndim == 3:
                R_anat = Rmean(R_anat)
            if O_anat.ndim == 2:
                O_anat = Vmean(O_anat)

            R_rel_frames = np.zeros((3, 3, n_frames))
            for i in range(n_frames):
                R_rel_frames[:, :, i] = R_tech[:, :, i].T @ R_anat
            R_rel = Rmean(R_rel_frames)

            O_rel = np.zeros((3, n_frames))
            for i in range(n_frames):
                O_rel[:, i] = R_tech[:, :, i].T @ (O_anat - O_tech[:, i])
            O_rel_mean = Vmean(O_rel)

            REFFRAME['Technical_to_Anatomical'][tech_key] = {
                'R': R_rel,
                'O': O_rel_mean,
            }

    # Right leg
    for tech_key, seg_key in [
        ('RightThigh', 'RightThigh'),
        ('RightShank', 'RightShank'),
        ('RightPelvis', 'RightPelvis'),
        ('RightRearfoot', 'RightRearfoot'),
        ('RightForefoot', 'RightForefoot'),
        ('RightFoot', 'RightFoot'),
    ]:
        if tech_key in technical_frames and seg_key in anatomical_frames:
            R_tech = technical_frames[tech_key]['R']  # shape (3, 3, n_frames)
            O_tech = technical_frames[tech_key]['O']  # shape (3, n_frames)
            R_anat = anatomical_frames[seg_key]['R']  # shape (3, 3)
            O_anat = anatomical_frames[seg_key]['O']  # shape (3,) or (3, n_frames)
            
            # Ensure R_anat is 2D (3, 3)
            if R_anat.ndim == 3:
                R_anat = Rmean(R_anat)
            if O_anat.ndim == 2:
                O_anat = Vmean(O_anat)
            
            # Compute relative rotation per frame, then mean
            R_rel_frames = np.zeros((3, 3, n_frames))
            for i in range(n_frames):
                R_rel_frames[:, :, i] = R_tech[:, :, i].T @ R_anat
            R_rel = Rmean(R_rel_frames)
            
            # Compute relative offset per frame, then mean
            O_rel = np.zeros((3, n_frames))
            for i in range(n_frames):
                O_rel[:, i] = R_tech[:, :, i].T @ (O_anat - O_tech[:, i])
            O_rel_mean = Vmean(O_rel)
            
            REFFRAME['Technical_to_Anatomical'][tech_key] = {
                'R': R_rel,
                'O': O_rel_mean,
            }
    
    # Left leg (analogous, but using Left... keys)
    for tech_key, seg_key in [
        ('LeftThigh', 'LeftThigh'),
        ('LeftShank', 'LeftShank'),
        ('LeftPelvis', 'LeftPelvis'),
        ('LeftRearfoot', 'LeftRearfoot'),
        ('LeftForefoot', 'LeftForefoot'),
        ('LeftFoot', 'LeftFoot'),
    ]:
        if tech_key in technical_frames and seg_key in anatomical_frames:
            R_tech = technical_frames[tech_key]['R']  # shape (3, 3, n_frames)
            O_tech = technical_frames[tech_key]['O']  # shape (3, n_frames)
            R_anat = anatomical_frames[seg_key]['R']  # shape (3, 3)
            O_anat = anatomical_frames[seg_key]['O']  # shape (3,) or (3, n_frames)
            
            # Ensure R_anat is 2D (3, 3)
            if R_anat.ndim == 3:
                R_anat = Rmean(R_anat)
            if O_anat.ndim == 2:
                O_anat = Vmean(O_anat)
            
            # Compute relative rotation per frame, then mean
            R_rel_frames = np.zeros((3, 3, n_frames))
            for i in range(n_frames):
                R_rel_frames[:, :, i] = R_tech[:, :, i].T @ R_anat
            R_rel = Rmean(R_rel_frames)
            
            # Compute relative offset per frame, then mean
            O_rel = np.zeros((3, n_frames))
            for i in range(n_frames):
                O_rel[:, i] = R_tech[:, :, i].T @ (O_anat - O_tech[:, i])
            O_rel_mean = Vmean(O_rel)
            
            REFFRAME['Technical_to_Anatomical'][tech_key] = {
                'R': R_rel,
                'O': O_rel_mean,
            }
    
    # ========== COMPUTE JOINT ROTATIONS ==========
    REFFRAME['Joint'] = {}
    
    # Hip joint: Pelvis to Thigh
    if 'RightPelvis' in anatomical_frames and 'RightThigh' in anatomical_frames:
        R_joint = anatomical_frames['RightPelvis']['R'].T @ anatomical_frames['RightThigh']['R']
        REFFRAME['Joint']['Right_Pelvis_to_Thigh'] = R_joint
    
    if 'LeftPelvis' in anatomical_frames and 'LeftThigh' in anatomical_frames:
        R_joint = anatomical_frames['LeftPelvis']['R'].T @ anatomical_frames['LeftThigh']['R']
        REFFRAME['Joint']['Left_Pelvis_to_Thigh'] = R_joint
    
    # Knee joint: Thigh to Shank
    if 'RightThigh' in anatomical_frames and 'RightShank' in anatomical_frames:
        R_joint = anatomical_frames['RightThigh']['R'].T @ anatomical_frames['RightShank']['R']
        REFFRAME['Joint']['Right_Thigh_to_Shank'] = R_joint
    
    if 'LeftThigh' in anatomical_frames and 'LeftShank' in anatomical_frames:
        R_joint = anatomical_frames['LeftThigh']['R'].T @ anatomical_frames['LeftShank']['R']
        REFFRAME['Joint']['Left_Thigh_to_Shank'] = R_joint

    # Ankle joint: Shank to Rearfoot
    if 'RightShank' in anatomical_frames and 'RightRearfoot' in anatomical_frames:
        R_joint = anatomical_frames['RightShank']['R'].T @ anatomical_frames['RightRearfoot']['R']
        REFFRAME['Joint']['Right_Shank_to_Rearfoot'] = R_joint

    if 'LeftShank' in anatomical_frames and 'LeftRearfoot' in anatomical_frames:
        R_joint = anatomical_frames['LeftShank']['R'].T @ anatomical_frames['LeftRearfoot']['R']
        REFFRAME['Joint']['Left_Shank_to_Rearfoot'] = R_joint

    # MPJ (MTP) joint: Rearfoot to Forefoot
    if 'RightRearfoot' in anatomical_frames and 'RightForefoot' in anatomical_frames:
        R_joint = anatomical_frames['RightRearfoot']['R'].T @ anatomical_frames['RightForefoot']['R']
        REFFRAME['Joint']['Right_Rearfoot_to_Forefoot'] = R_joint

    if 'LeftRearfoot' in anatomical_frames and 'LeftForefoot' in anatomical_frames:
        R_joint = anatomical_frames['LeftRearfoot']['R'].T @ anatomical_frames['LeftForefoot']['R']
        REFFRAME['Joint']['Left_Rearfoot_to_Forefoot'] = R_joint

    # Trunk-Pelvis joint: Pelvis to Torso (lumbar/trunk flexion, lateral lean, rotation)
    # Uses RightPelvis as the shared pelvic reference frame (identical to LeftPelvis).
    if 'RightPelvis' in anatomical_frames and 'Torso' in anatomical_frames:
        R_pelvis = anatomical_frames['RightPelvis']['R']
        R_torso  = anatomical_frames['Torso']['R']
        if R_pelvis.ndim == 3:
            R_pelvis = Rmean(R_pelvis)
        if R_torso.ndim == 3:
            R_torso = Rmean(R_torso)
        REFFRAME['Joint']['Pelvis_to_Torso'] = R_pelvis.T @ R_torso
    
    # ========== PACKAGE MARKERS ==========
    MARKERS = {
        'Filt': markers_filt_dict,
        'Raw': markers_raw_dict,
        'Derived': {
            'HJC_right': hip_centers.get('HJC_right'),
            'HJC_left': hip_centers.get('HJC_left'),
            'KJC_right': hip_centers.get('KJC_right'),
            'KJC_left': hip_centers.get('KJC_left'),
            'AJC_right': hip_centers.get('AJC_right'),
            'AJC_left': hip_centers.get('AJC_left'),
        }
    }
    
    return REFFRAME, MARKERS, technical_frames, anatomical_frames
