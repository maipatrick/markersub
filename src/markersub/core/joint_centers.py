"""Lower-limb joint centres from marker positions (lab frame, mm).

Ankle and knee joint centres are the midpoints of the malleolus and epicondyle
markers (Wu et al. 2002, J Biomech). The hip joint centre uses the Harrington
et al. (2007, J Biomech 40:595-602) regression with leg length when left and
right PSIS markers exist, and the Bell et al. (1990, J Biomech) regression on
the inter-ASIS distance for marker sets with a single sacrum marker.
"""

import numpy as np


def _get_marker(marker_data, labels, name, frame=None):
    """Return a marker position; frame=None returns the mean over all frames."""
    labels = list(labels)
    idx = labels.index(name)
    if frame is not None:
        return marker_data[idx, :3, frame].astype(float)
    else:
        return marker_data[idx, :3, :].mean(axis=1).astype(float)


def get_marker_pos(marker_data, labels, name, frame=None):
    """Public wrapper that returns a (3,) array."""
    return _get_marker(marker_data, labels, name, frame)


# --- Ankle joint centres ---
def ankle_joint_center(marker_data, labels, side='left', frame=None):
    """Ankle joint centre: midpoint of mal_med and mal_lat (Wu et al. 2002).

    Parameters
    ----------
    side : 'left' or 'right'
    frame : int or None (None = static mean)

    Returns
    -------
    AJC : (3,) array, mm
    """
    s = side.lower()
    mal_med = get_marker_pos(marker_data, labels, f'mal_med_{s}', frame)
    mal_lat = get_marker_pos(marker_data, labels, f'mal_lat_{s}', frame)
    AJC = (mal_med + mal_lat) / 2.0
    return AJC


# --- Knee joint centres ---
def knee_joint_center(marker_data, labels, side='left', frame=None):
    """Knee joint centre: midpoint of epi_med and epi_lat (Wu et al. 2002).

    Parameters
    ----------
    side : 'left' or 'right'
    frame : int or None

    Returns
    -------
    KJC : (3,) array, mm
    """
    s = side.lower()
    epi_med = get_marker_pos(marker_data, labels, f'epi_med_{s}', frame)
    epi_lat = get_marker_pos(marker_data, labels, f'epi_lat_{s}', frame)
    KJC = (epi_med + epi_lat) / 2.0
    return KJC


# --- Pelvic coordinate system (ISB) ---
def pelvic_coordinate_system(marker_data, labels, frame=None):
    """Pelvic coordinate system after ISB (Wu et al. 2002).

      Origin : midpoint of ASIS_left and ASIS_right
      Z axis : origin -> ASIS_right (mediolateral, + right)
      X axis : anterior, orthogonal to Z, in the ASIS/PSIS plane
      Y axis : superior, Y = Z x X

    Returns
    -------
    origin : (3,) array
    R      : (3,3) rotation matrix with columns [X|Y|Z]
    """
    ASIS_L = get_marker_pos(marker_data, labels, 'SIAS_left',  frame)
    ASIS_R = get_marker_pos(marker_data, labels, 'SIAS_right', frame)
    PSIS_L = get_marker_pos(marker_data, labels, 'SIPS_left',  frame)
    PSIS_R = get_marker_pos(marker_data, labels, 'SIPS_right', frame)

    origin   = (ASIS_L + ASIS_R) / 2.0
    mid_PSIS = (PSIS_L + PSIS_R) / 2.0

    Z = ASIS_R - origin
    Z = Z / np.linalg.norm(Z)

    temp = origin - mid_PSIS                # posterior -> anterior
    X    = temp - np.dot(temp, Z) * Z       # Gram-Schmidt: orthogonal to Z
    X    = X / np.linalg.norm(X)

    Y = np.cross(Z, X)
    Y = Y / np.linalg.norm(Y)

    R = np.column_stack([X, Y, Z])          # 3x3, columns are the axes
    return origin, R


# --- Hip joint centres (Harrington 2007) ---
def hip_joint_center(marker_data, labels, side='left', frame=None):
    """Hip joint centre after Harrington et al. (2007, J Biomech 40:595-602).

    The Harrington regression with leg length is used.

    Inputs (from marker data):
      SIAS_left, SIAS_right  (= ASIS)
      SIPS_left, SIPS_right  (= PSIS)
      Leg length: marker-based approximation along the ipsilateral chain
      ASIS -> knee (epi_med) -> ankle (mal_med)

    Pelvic CS (ISB):
      Origin = mid-ASIS
      X = anterior, Y = superior, Z = right

    Harrington equations in the pelvic CS (mm):
      x_HJC = -0.24*PD - 9.9                  (AP, + anterior)
      y_HJC = -0.16*PW - 0.04*LL - 7.1        (SI, + superior)
      z_HJC = +-(0.28*PD + 0.16*PW + 7.9)     (ML, + right)

    If a single 'sacrum' marker is present (no left/right PSIS), the Bell et al.
    (1990) regression on the inter-ASIS distance is used instead.

    Parameters
    ----------
    side : 'left' or 'right'
    frame : int or None

    Returns
    -------
    HJC : (3,) array in the global lab frame (mm)
    """
    s = side.lower()

    ASIS_L = get_marker_pos(marker_data, labels, 'SIAS_left',  frame)
    ASIS_R = get_marker_pos(marker_data, labels, 'SIAS_right', frame)

    # Single sacrum marker (no left/right PSIS): Bell et al. (1990). Only the inter-ASIS
    # distance T is used as predictor; the sacrum only orients the pelvis frame and is not
    # used to estimate pelvis depth. Marker sets with left and right PSIS use the
    # Harrington form below.
    if 'sacrum' in labels:
        sacrum = get_marker_pos(marker_data, labels, 'sacrum', frame)
        op = (ASIS_L + ASIS_R) / 2.0
        e_y = ASIS_L - ASIS_R;  e_y = e_y / np.linalg.norm(e_y)     # mediolateral (left +)
        ante = op - sacrum;     ante_n = ante / np.linalg.norm(ante)
        e_z = np.cross(e_y, ante_n); e_z = e_z / np.linalg.norm(e_z)
        if e_z[2] < 0:
            e_z = -e_z                                              # superior (lab Z up)
        e_x = np.cross(e_y, e_z); e_x = e_x / np.linalg.norm(e_x)
        if np.dot(e_x, ante) < 0:
            e_x = -e_x                                              # anterior
        T = np.linalg.norm(ASIS_R - ASIS_L)                        # inter-ASIS (the only Bell predictor)
        off_x = -0.19 * T                                          # posterior
        off_z = -0.30 * T                                          # inferior
        off_y = -0.36 * T if s == 'right' else 0.36 * T           # toward the hip side
        return op + e_x * off_x + e_y * off_y + e_z * off_z

    PSIS_L = get_marker_pos(marker_data, labels, 'SIPS_left',  frame)
    PSIS_R = get_marker_pos(marker_data, labels, 'SIPS_right', frame)

    # Anthropometric measures (mm)
    PW = np.linalg.norm(ASIS_R - ASIS_L)                           # inter-ASIS width
    PD = np.linalg.norm((ASIS_L + ASIS_R) / 2 - (PSIS_L + PSIS_R) / 2)  # pelvis depth

    # Leg length: ASIS -> epi_med (knee) -> mal_med (ankle)
    ASIS_ipsi = get_marker_pos(marker_data, labels, f'SIAS_{s}',    frame)
    knee_med  = get_marker_pos(marker_data, labels, f'epi_med_{s}', frame)
    ankle_med = get_marker_pos(marker_data, labels, f'mal_med_{s}', frame)
    LL = (np.linalg.norm(knee_med - ASIS_ipsi) +
          np.linalg.norm(ankle_med - knee_med))

    # Harrington regression equations in the pelvic CS
    x_HJC = -0.24 * PD - 9.9
    y_HJC = -0.16 * PW - 0.04 * LL - 7.1
    z_HJC =  0.28 * PD + 0.16 * PW + 7.9

    if s == 'left':
        z_HJC = -z_HJC           # left = negative Z axis of the ISB pelvic CS

    # Back to global coordinates
    origin, R = pelvic_coordinate_system(marker_data, labels, frame)
    HJC_local = np.array([x_HJC, y_HJC, z_HJC])
    HJC_global = origin + R @ HJC_local

    return HJC_global


# --- All joint centres at once ---
def compute_all_joint_centers(marker_data, labels, frame=None, verbose=False):
    """Compute all lower-limb joint centres for the left and right side.

    Returns
    -------
    dict with keys HJC_left, HJC_right, KJC_left, KJC_right, AJC_left, AJC_right.
    All values are (3,) arrays in global lab coordinates (mm). A side whose
    markers are missing is skipped.
    References: Harrington ME et al. (2007) J Biomech 40:595-602.; Wu G et al. (2002) J Biomech 35:543-548.
    """
    jc = {}
    for side in ('left', 'right'):
        try:
            jc[f'HJC_{side}'] = hip_joint_center(marker_data, labels, side, frame)
            jc[f'KJC_{side}'] = knee_joint_center(marker_data, labels, side, frame)
            jc[f'AJC_{side}'] = ankle_joint_center(marker_data, labels, side, frame)
        except ValueError as exc:
            if verbose:
                print(f"  compute_all_joint_centers: side '{side}' skipped ({exc})")

    # Optional summary
    if verbose:
        print("\n=== Joint centres (mm, global lab frame) ===")
        for name, pos in jc.items():
            print(f"  {name:12s}: [{pos[0]:8.1f}, {pos[1]:8.1f}, {pos[2]:8.1f}]")

    return jc
