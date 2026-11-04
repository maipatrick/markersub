"""Analysis settings.

All user-adjustable parameters of the chain live in one dataclass, so a complete analysis is
described by a single ``Settings`` object (see ``examples/run_example.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Settings:
    """Parameters of the processing chain. The defaults reproduce the companion article."""

    # --- file discovery -------------------------------------------------------------------
    static_token: str = "static"
    """Case-insensitive token that marks the static (standing) trial in the c3d file name."""

    mb_prefix: str = "MB_"
    """Prefix of marker-based marker labels. ``MB_r_knee_study`` is the marker-based
    counterpart of the markerless marker ``r_knee_study``."""

    # --- markers ----------------------------------------------------------------------------
    marker_cutoff_hz: float = 20.0
    """Low-pass cut-off of the marker-based trajectories [Hz] (4th-order Butterworth applied forward
    and backward). Also used for the static trial and for the segment velocities before
    differentiation. Must be below 50 [Hz] (the static posture is evaluated at a nominal 100 [Hz])."""

    ml_marker_cutoff_hz: float | None = None
    """Low-pass cut-off of the markerless trajectories [Hz]. ``None`` keeps them as delivered
    by the markerless software (they are usually filtered there already)."""

    gap_fill: bool = True
    """Linear interpolation of missing marker-based samples (static trial and analysed window).
    Markerless markers are never gap-filled; a markerless marker with missing samples in the
    analysed window is dropped."""

    crop_buffer_frames: int = 20
    """Marker frames kept before initial contact and after toe-off when cropping a trial."""

    # --- force plates and contact detection ------------------------------------------------------
    force_cutoff_hz: float = 20.0
    """Low-pass cut-off of force, centre of pressure and plate moment [Hz]."""

    force_threshold_n: float = 20.0
    """Contact threshold on the raw (unfiltered) vertical force [N]."""

    min_contact_s: float = 0.100
    """Minimum duration of a sustained contact [s]. Shorter supra-threshold runs are ignored."""

    cop_valid_threshold_n: float = 50.0
    """The centre of pressure is only trusted where the raw vertical force exceeds this value [N]."""

    # --- outcomes ---------------------------------------------------------------------------
    analysis_window_s: float = 0.100
    """Length of the early-stance window after initial contact used for the outcomes [s]."""

    # --- model ------------------------------------------------------------------------------
    massless_foot: bool = True
    """Set the foot segment mass and inertia to zero in the inverse dynamics (as in the article,
    so that the knee moment does not depend on the foot markers). False = full inverse dynamics."""

    # --- substitution -----------------------------------------------------------------------
    substitution: bool = True
    """Run the marker substitution variants when a trial carries both marker sources."""

    def validate(self) -> None:
        """Raise ValueError for settings outside the supported range."""
        if not self.marker_cutoff_hz or not 0 < float(self.marker_cutoff_hz) < 50:
            raise ValueError("marker_cutoff_hz must be a number between 0 and 50 [Hz]")
        if self.ml_marker_cutoff_hz is not None and float(self.ml_marker_cutoff_hz) <= 0:
            raise ValueError("ml_marker_cutoff_hz must be None or a positive number [Hz]")
        if self.force_threshold_n <= 0 or self.min_contact_s <= 0 or self.analysis_window_s <= 0:
            raise ValueError("force_threshold_n, min_contact_s and analysis_window_s must be positive")
        if not self.static_token or not self.mb_prefix:
            raise ValueError("static_token and mb_prefix must not be empty")

    def to_dict(self) -> dict:
        return asdict(self)
