"""Signal processing helpers: zero-lag Butterworth filters, gap filling, crop alignment."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, filtfilt


def lowpass_markers(xyz: np.ndarray, cutoff_hz: float, rate: float) -> np.ndarray:
    """4th-order zero-lag Butterworth low-pass of (3, n_markers, n_frames).

    Markers containing any non-finite sample are left unfiltered (they are reported as missing
    downstream). Windows shorter than 16 samples are returned unchanged.
    References: Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    norm = cutoff_hz / (rate / 2.0)
    if norm >= 1.0 or xyz.shape[2] < 16:
        return xyz
    b, a = butter(4, norm, btype="low", analog=False)
    out = xyz.copy()
    for m in range(out.shape[1]):
        if not np.all(np.isfinite(out[:, m, :])):
            continue
        for d in range(3):
            out[d, m, :] = filtfilt(b, a, out[d, m, :])
    return out


def lowpass_3n(arr: np.ndarray, cutoff_hz: float, rate: float) -> np.ndarray:
    """4th-order zero-lag Butterworth low-pass of a (3, N) signal (e.g. centre of pressure).
    References: Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    norm = cutoff_hz / (rate / 2.0)
    if norm >= 1.0 or arr.shape[-1] < 16:
        return arr
    b, a = butter(4, norm, btype="low", analog=False)
    out = arr.copy()
    for d in range(3):
        if np.all(np.isfinite(out[d])):
            out[d] = filtfilt(b, a, out[d])
    return out


def lowpass_force(arr: np.ndarray, cutoff_hz: float, rate: float) -> np.ndarray:
    """4th-order zero-lag Butterworth low-pass of a (3, N) force or moment signal with
    half-cosine padding to zero at both ends (max(30, 3 * rate / cutoff) samples), which
    suppresses filter ringing at contact onset and offset.
    References: Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    norm = cutoff_hz / (rate / 2.0)
    if norm >= 1.0:
        return arr.copy()
    b, a = butter(4, norm, btype="low", analog=False)
    pad = max(30, int(3.0 * rate / cutoff_hz))
    t = np.arange(pad, dtype=float)
    out = arr.copy()
    for d in range(3):
        x = arr[d]
        if not np.all(np.isfinite(x)):
            continue
        start = x[0] * 0.5 * (1.0 - np.cos(np.pi * t / pad))
        end = x[-1] * 0.5 * (1.0 + np.cos(np.pi * t / pad))
        out[d] = filtfilt(b, a, np.concatenate([start, x, end]))[pad: pad + len(x)]
    return out


def fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    """Replace non-finite samples of a (3, N) array (force plate signals): each gap takes the
    next valid sample, trailing gaps take the last valid sample."""
    out = arr.copy()
    for d in range(3):
        row = out[d]
        bad = ~np.isfinite(row)
        if not bad.any():
            continue
        good = np.where(~bad)[0]
        if good.size == 0:
            out[d] = 0.0
            continue
        idx = np.arange(row.size)
        prev = np.maximum.accumulate(np.where(~bad, idx, -1))       # last valid at or before i
        nxt = np.minimum.accumulate(np.where(~bad, idx, row.size)[::-1])[::-1]
        src = np.where(nxt < row.size, nxt, prev)                    # back fill, trailing gap: forward fill
        out[d] = row[src]
    return out


def missing_mask(xyz: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """(n_markers, n_frames) True where a sample is missing: residual < 0, non-finite, or exactly
    (0, 0, 0) (some exports store occluded samples as zeros)."""
    return (residual < 0) | ~np.all(np.isfinite(xyz), axis=0) | np.all(xyz == 0.0, axis=0)


def fill_gaps_linear(xyz: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Linear interpolation of missing samples per marker; leading and trailing gaps take the
    first or last valid sample.
    Markers without any valid sample are left untouched."""
    out = xyz.copy()
    miss = missing_mask(xyz, residual)
    frames = np.arange(out.shape[2], dtype=float)
    for m in range(out.shape[1]):
        valid = ~miss[m]
        if valid.all() or not valid.any():
            continue
        for d in range(3):
            out[d, m, :] = np.interp(frames, frames[valid], out[d, m, valid])
    return out


def aligned_crop(t_start: float, t_end: float, rate: float, analog_rate: float,
                 n_frames: int, n_force: int) -> tuple[int, int, int, int]:
    """Crop indices (f0, f1, s0, s1) of markers and force on one common time grid.

    With force/marker ratio p/q (e.g. 5/1 for 1000/200 Hz) the marker start is moved to a frame
    that coincides with a force sample and the window holds a whole number of force samples,
    so marker frame k and force sample k * p / q are simultaneous."""
    f0 = max(0, int(round(t_start * rate)))
    f1 = min(n_frames, int(round(t_end * rate)) + 1)
    if rate <= 0 or analog_rate <= 0:
        s0 = max(0, int(round(t_start * analog_rate)))
        return f0, f1, s0, min(n_force, int(round(t_end * analog_rate)) + 1)
    r = float(analog_rate) / float(rate)
    ratio = Fraction(r).limit_denominator(1000)
    p, q = ratio.numerator, ratio.denominator
    if q <= 10 and abs(float(ratio) - r) <= 1e-9 * r:
        f0 -= f0 % q
        f1 = min(f1, f0 + ((n_force - (f0 * p) // q) * q) // p)
        nk = f1 - f0
        nk -= nk % q
        f1 = f0 + nk
        s0 = (f0 * p) // q
        return f0, f1, s0, s0 + (nk * p) // q
    s0 = int(round(f0 * r))
    f1 = min(f1, f0 + int((n_force - s0) / r))
    return f0, f1, s0, min(n_force, s0 + int(round((f1 - f0) * r)))
