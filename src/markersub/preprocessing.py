"""Static and dynamic trial preparation.

Dynamic trials: contact detection per analysed leg, crop to the union of the contact windows plus
a marker buffer (markers and force on one time grid), linear gap filling and low-pass filtering of
the markers, and filtering of force, centre of pressure (COP) and plate moment. The free moment
is computed as the exact residual of the filtered plate wrench at the filtered COP.

Static trial: gap filling and low-pass filtering, then the time-averaged posture per marker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import signals as sg
from .c3d_io import C3DTrial
from .contacts import Contact, point_on_plate, select_contact
from .markers import split_sources
from .settings import Settings


@dataclass
class MarkerSet:
    """Marker trajectories of one source in internal model names."""
    data: np.ndarray            # (n_frames, n_markers, 3) [mm] (static pose: (n_markers, 3))
    labels: list[str]


@dataclass
class LegEvents:
    leg: str
    plate: int                  # 1-based plate number
    td_frame: int               # initial contact, marker frame in the cropped trial
    to_frame: int               # toe-off, marker frame in the cropped trial
    td_sample: int              # initial contact, force sample in the cropped trial
    to_sample: int
    td_s: float                 # initial contact [s] from the start of the file
    to_s: float
    rule: str                   # contact selection rule (see contacts.py)
    n_contacts: int


@dataclass
class PreparedTrial:
    name: str
    rate: float                 # marker rate [Hz]
    analog_rate: float          # force rate [Hz]
    sources: dict[str, MarkerSet]
    plates: list[dict]          # per plate: force (3,N) [N], cop (3,N) [mm], moment (3,N) [N mm]
    events: dict[str, LegEvents]
    window: tuple[int, int, int, int]   # marker f0:f1 and force s0:s1 of the crop
    notes: list[str] = field(default_factory=list)


def _foot_on_plate_test(trial: C3DTrial, mb_cols: dict[str, int], plate, leg: str):
    """Return test(peak_force_sample) -> True | False | None for the foot of *leg* (columns of
    the c3d by internal name), or None if the foot markers are absent."""
    """Rule 3 of the contact selection: heel and forefoot midpoint over the plate at peak force."""
    ff = "forfoot" if leg == "right" else "forefoot"
    names = (f"calc_back_{leg}", f"{ff}_lat_{leg}", f"{ff}_med_{leg}")
    if any(n not in mb_cols for n in names) or trial.analog_rate <= 0:
        return None
    heel_i, lat_i, med_i = (mb_cols[n] for n in names)
    corners_xy = plate.corners[:2].T

    def test(peak_sample: int):
        m = int(round(peak_sample / trial.analog_rate * trial.rate))
        m = min(max(m, 0), trial.n_frames - 1)
        heel, a, b = (trial.xyz[:2, i, m] for i in (heel_i, lat_i, med_i))
        for p in (heel, a, b):
            if not np.all(np.isfinite(p)) or (abs(p[0]) < 1e-9 and abs(p[1]) < 1e-9):
                return None
        return point_on_plate(heel, corners_xy) and point_on_plate((a + b) / 2.0, corners_xy)
    return test


def prepare_static(trial: C3DTrial, settings: Settings) -> dict[str, MarkerSet]:
    """Time-averaged static posture per marker source, ``{"MB": MarkerSet, "ML": MarkerSet}``.
    References: Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    out = {}
    for src, cols in split_sources(trial.labels, settings.mb_prefix).items():
        if not cols:
            continue
        idx = [c for c, _ in cols]
        xyz = trial.xyz[:, idx, :]
        miss = sg.missing_mask(xyz, trial.residual[idx])
        fill = settings.gap_fill and src == "MB"                       # ML markers are never filled
        if fill:
            xyz = sg.fill_gaps_linear(xyz, trial.residual[idx])
        cutoff = settings.marker_cutoff_hz if src == "MB" else settings.ml_marker_cutoff_hz
        if cutoff:
            xyz = sg.lowpass_markers(xyz, float(cutoff), trial.rate)
        if not fill:
            xyz = xyz.copy()
            xyz[:, miss] = np.nan                                      # exclude missing samples
        with np.errstate(invalid="ignore"):
            pose = np.nanmean(np.transpose(xyz, (2, 1, 0)), axis=0)     # (n_markers, 3)
        ok = np.all(np.isfinite(pose), axis=1)
        out[src] = MarkerSet(data=pose[ok], labels=[n for (_, n), k in zip(cols, ok) if k])
    return out


def prepare_dynamic(trial: C3DTrial, leg_plates: dict[str, int], settings: Settings) -> PreparedTrial:
    """Detect contacts, crop and filter one dynamic trial. ``leg_plates`` e.g. ``{"left": 3}``.
    References: Tirosh O, Sparrow WA (2003) J Appl Biomech 19:178-184.; Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley.
    """
    if not trial.plates:
        raise ValueError(f"{trial.name}: no force platform data in the c3d")
    sources = split_sources(trial.labels, settings.mb_prefix)
    mb_cols = {n: c for c, n in sources["MB"]}

    contacts: dict[str, tuple[int, Contact]] = {}
    pre_notes: list[str] = []
    for leg, plate in leg_plates.items():
        if not 1 <= plate <= len(trial.plates):
            raise ValueError(f"{trial.name}: plate {plate} requested, file has {len(trial.plates)}")
        p = trial.plates[plate - 1]
        con = select_contact(p.force[2], trial.analog_rate, settings.force_threshold_n,
                             settings.min_contact_s, _foot_on_plate_test(trial, mb_cols, p, leg))
        if con is None:
            raise ValueError(f"{trial.name}: no sustained contact on plate {plate} ({leg} leg); "
                             f"the vertical force must be positive upward and exceed "
                             f"{settings.force_threshold_n:g} [N] for {settings.min_contact_s:g} [s]")
        contacts[leg] = (plate, con)
        # plausibility of the leg token: the named foot should be over the plate at peak force
        for src in ("MB", "ML"):
            cols = {n: c for c, n in sources[src]}
            test = _foot_on_plate_test(trial, cols, p, leg)
            if test is None:
                continue
            i0, i1 = int(round(con.td_s * trial.analog_rate)), int(round(con.to_s * trial.analog_rate))
            ok_foot = test(i0 + int(np.argmax(p.force[2][i0:i1 + 1])))
            if ok_foot is False:
                pre_notes.append(f"{src} foot markers of the {leg} leg are not over plate {plate} at peak "
                                 f"force; check the leg token in the file name")
            if ok_foot is not None:
                break

    # crop to the union of all analysed contacts plus the marker buffer
    buf = settings.crop_buffer_frames / trial.rate
    t0 = max(0.0, min(c.td_s for _, c in contacts.values()) - buf)
    t1 = max(c.to_s for _, c in contacts.values()) + buf
    n_force = trial.plates[0].force.shape[1]
    f0, f1, s0, s1 = sg.aligned_crop(t0, t1, trial.rate, trial.analog_rate, trial.n_frames, n_force)
    if f1 - f0 < 16:
        raise ValueError(f"{trial.name}: analysed window too short ({f1 - f0} frames)")

    notes: list[str] = list(pre_notes)
    prepared: dict[str, MarkerSet] = {}
    for src, cols in sources.items():
        if not cols:
            continue
        idx = [c for c, _ in cols]
        xyz = trial.xyz[:, idx, f0:f1]
        res = trial.residual[idx, f0:f1]
        miss = sg.missing_mask(xyz, res)
        if src == "MB" and settings.gap_fill and miss.any():
            xyz = sg.fill_gaps_linear(xyz, res)
            notes.append(f"{src}: {int(miss.sum())} missing marker samples filled linearly")
        else:
            xyz = np.where(miss[None], np.nan, xyz)                     # missing samples stay missing
        cutoff = settings.marker_cutoff_hz if src == "MB" else settings.ml_marker_cutoff_hz
        if cutoff:
            xyz = sg.lowpass_markers(xyz, float(cutoff), trial.rate)
        ok = np.all(np.isfinite(xyz), axis=(0, 2))
        if not ok.all():
            notes.append(f"{src}: markers with missing samples in the analysed window dropped: "
                         + ", ".join(n for (_, n), k in zip(cols, ok) if not k))
        prepared[src] = MarkerSet(data=np.transpose(xyz[:, ok, :], (2, 1, 0)),
                                  labels=[n for (_, n), k in zip(cols, ok) if k])

    plates_out = []
    for p in trial.plates:
        f = sg.fill_nan_nearest(p.force[:, s0:s1])
        cop = sg.fill_nan_nearest(p.cop[:, s0:s1])
        mc = sg.fill_nan_nearest(p.moment_ctr[:, s0:s1])
        if settings.force_cutoff_hz:
            fz_raw = np.abs(f[2]).copy()
            f = sg.lowpass_force(f, float(settings.force_cutoff_hz), trial.analog_rate)
            mc = sg.lowpass_force(mc, float(settings.force_cutoff_hz), trial.analog_rate)
            valid = fz_raw >= settings.cop_valid_threshold_n
            if int(np.count_nonzero(valid)) >= 2:          # COP is unreliable at low load
                cop = cop.copy()
                cop[:, ~valid] = np.nan
                cop = sg.fill_nan_nearest(cop)
            cop = sg.lowpass_3n(cop, float(settings.force_cutoff_hz), trial.analog_rate)
        free = mc - np.cross((cop - p.centre[:, None]).T, f.T).T   # free moment about the COP
        plates_out.append({"force": f, "cop": cop, "moment": free})

    events = {}
    for leg, (plate, c) in contacts.items():
        events[leg] = LegEvents(
            leg=leg, plate=plate,
            td_frame=int(round(c.td_s * trial.rate)) - f0, to_frame=int(round(c.to_s * trial.rate)) - f0,
            td_sample=int(round(c.td_s * trial.analog_rate)) - s0,
            to_sample=int(round(c.to_s * trial.analog_rate)) - s0,
            td_s=c.td_s, to_s=c.to_s, rule=c.rule, n_contacts=c.n_contacts)
    return PreparedTrial(name=trial.name, rate=trial.rate, analog_rate=trial.analog_rate,
                         sources=prepared, plates=plates_out, events=events,
                         window=(f0, f1, s0, s1), notes=notes)
