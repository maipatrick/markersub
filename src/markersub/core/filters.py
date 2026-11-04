"""Zero-lag Butterworth low-pass filtering of signals that may contain NaN gaps.

Gaps are bridged by linear interpolation before filtfilt and restored as NaN
afterwards. Series with fewer than 3 finite samples are returned unfiltered,
series not longer than the filtfilt pad length are returned gap-filled only.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def _filter_series_nan_safe(values, b, a):
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 3:
        return values.copy()

    x = np.arange(values.size, dtype=float)
    filled = values.copy()
    if not finite_mask.all():
        filled[~finite_mask] = np.interp(x[~finite_mask], x[finite_mask], values[finite_mask])

    padlen = 3 * (max(len(a), len(b)) - 1)
    if values.size <= padlen:
        return filled

    filtered = filtfilt(b, a, filled)
    filtered[~finite_mask] = np.nan
    return filtered


def butterworth_lowpass(data, cutoff_hz, sample_rate_hz, order=4, axis=-1):
    """References: Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley."""
    data = np.asarray(data, dtype=float)
    if cutoff_hz is None:
        return data.copy()

    nyquist = 0.5 * float(sample_rate_hz)
    normalized_cutoff = float(cutoff_hz) / nyquist
    if not 0 < normalized_cutoff < 1:
        raise ValueError(
            f'cutoff_hz must lie between 0 and Nyquist ({nyquist:.3f} Hz), got {cutoff_hz}'
        )

    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    moved = np.moveaxis(data, axis, -1)
    filtered = np.empty_like(moved, dtype=float)

    leading_shape = moved.shape[:-1]
    for index in np.ndindex(leading_shape):
        filtered[index] = _filter_series_nan_safe(moved[index], b, a)

    return np.moveaxis(filtered, -1, axis)
