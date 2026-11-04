"""Default lower-limb segment model: segments, their marker sets and joint pairs.

Marker names are the package's internal segment-model names. Each segment lists the
markers that define its technical frame and the key of its technical-to-anatomical
calibration entry.
"""

def get_default_segments_config():
    """
    Return the standard lower-limb segment configuration dict.

    Each entry maps a segment name to:
        'markers'   - ordered list of marker names used for the technical frame
        'gap_fill_markers' - optional superset used only for marker-gap filling
        'n_markers' - 3 or 4
        'ref_key'   - key into REFFRAME['Technical_to_Anatomical']
    """
    return {
        'RightForefoot': {
            'markers': ['toe_right', 'forfoot_lat_right', 'forfoot_med_right'],
            'n_markers': 3, 'ref_key': 'RightForefoot',
        },
        'RightRearfoot': {
            'markers': ['calc_back_right', 'calc_med_right', 'calc_lat_right'],
            'n_markers': 3, 'ref_key': 'RightRearfoot',
        },
        'RightFoot': {
            'markers': ['calc_back_right', 'forfoot_lat_right', 'forfoot_med_right'],
            'n_markers': 3, 'ref_key': 'RightFoot',
        },
        'RightShank': {
            'markers': ['cluster_tibia_right_1', 'cluster_tibia_right_2',
                        'cluster_tibia_right_3', 'cluster_tibia_right_4'],
            'n_markers': 4, 'ref_key': 'RightShank',
        },
        'RightThigh': {
            'markers': ['cluster_femur_right_1', 'cluster_femur_right_2',
                        'cluster_femur_right_3', 'cluster_femur_right_4'],
            'n_markers': 4, 'ref_key': 'RightThigh',
        },
        'RightPelvis': {
            'markers': ['SIAS_right', 'SIAS_left', 'SIPS_right', 'SIPS_left'],
            'n_markers': 4, 'ref_key': 'RightPelvis',
        },
        'LeftForefoot': {
            'markers': ['toe_left', 'forefoot_lat_left', 'forefoot_med_left'],
            'n_markers': 3, 'ref_key': 'LeftForefoot',
        },
        'LeftRearfoot': {
            'markers': ['calc_back_left', 'calc_med_left', 'calc_lat_left'],
            'n_markers': 3, 'ref_key': 'LeftRearfoot',
        },
        'LeftFoot': {
            'markers': ['calc_back_left', 'forefoot_lat_left', 'forefoot_med_left'],
            'n_markers': 3, 'ref_key': 'LeftFoot',
        },
        'LeftShank': {
            'markers': ['cluster_tibia_left_1', 'cluster_tibia_left_2',
                        'cluster_tibia_left_3', 'cluster_tibia_left_4'],
            'n_markers': 4, 'ref_key': 'LeftShank',
        },
        'LeftThigh': {
            'markers': ['cluster_femur_left_1', 'cluster_femur_left_2',
                        'cluster_femur_left_3', 'cluster_femur_left_4'],
            'n_markers': 4, 'ref_key': 'LeftThigh',
        },
        'LeftPelvis': {
            'markers': ['SIAS_right', 'SIAS_left', 'SIPS_right', 'SIPS_left'],
            'n_markers': 4, 'ref_key': 'LeftPelvis',
        },
        'Torso': {
            'markers': ['C_7', 'B_10', 'acrom_right', 'acrom_left'],
            'gap_fill_markers': ['clav', 'C_7', 'sternum', 'B_10', 'acrom_right', 'acrom_left'],
            'n_markers': 4, 'ref_key': 'Torso',
        },
        'Head': {
            'markers': ['head_front_right', 'head_front_left', 'head_back_right', 'head_back_left'],
            'n_markers': 4, 'ref_key': 'Head',
        },
        'RightUpperArm': {
            'markers': ['cluster_upperarm_right_1', 'cluster_upperarm_right_2', 'cluster_upperarm_right_3'],
            'n_markers': 3, 'ref_key': 'RightUpperArm',
        },
        'LeftUpperArm': {
            'markers': ['cluster_upperarm_left_1', 'cluster_upperarm_left_2', 'cluster_upperarm_left_3'],
            'n_markers': 3, 'ref_key': 'LeftUpperArm',
        },
        'RightLowerArm': {
            'markers': ['cluster_lowerarm_right_1', 'cluster_lowerarm_right_2', 'cluster_lowerarm_right_3'],
            'n_markers': 3, 'ref_key': 'RightLowerArm',
        },
        'LeftLowerArm': {
            'markers': ['cluster_lowerarm_left_1', 'cluster_lowerarm_left_2', 'cluster_lowerarm_left_3'],
            'n_markers': 3, 'ref_key': 'LeftLowerArm',
        },
    }


def get_default_joint_pairs():
    """
    Return the standard lower-limb joint-pair dict.

    Maps joint_name -> (proximal_segment, distal_segment).
    """
    return {
        'Right_Hip':     ('RightPelvis',  'RightThigh'),
        'Right_Knee':    ('RightThigh',   'RightShank'),
        'Right_Ankle':   ('RightShank',   'RightRearfoot'),
        'Right_MidFoot': ('RightRearfoot','RightForefoot'),
        'Left_Hip':      ('LeftPelvis',   'LeftThigh'),
        'Left_Knee':     ('LeftThigh',    'LeftShank'),
        'Left_Ankle':    ('LeftShank',    'LeftRearfoot'),
        'Left_MidFoot':  ('LeftRearfoot', 'LeftForefoot'),
        'Trunk_Pelvis':  ('RightPelvis',  'Torso'),
    }
