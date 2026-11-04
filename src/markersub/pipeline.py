"""Subject- and dataset-level analysis.

Folder layout (one folder per subject)::

    data/
      S01/
        subject.csv            mass_kg,height_m,sex   (one data row; sex = female | male)
        S01_static.c3d         static trial (file name contains Settings.static_token)
        S01_cut01_L3.c3d       dynamic trials, analysed leg and plate in the name (see naming.py)
        ...
      S02/ ...

Each c3d may contain markerless markers (OpenCap augmenter names), marker-based markers
(the same names with the MB prefix) or both, plus the force platforms. With both sources and
``Settings.substitution`` the substitution variants are analysed as well.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import kinematics, kinetics, outcomes
from .c3d_io import read_c3d
from .naming import is_static, parse_leg_plates
from .preprocessing import prepare_dynamic, prepare_static
from .settings import Settings
from .substitution import Variant, build, standard_variants


@dataclass
class Subject:
    id: str
    mass_kg: float
    height_m: float
    sex: str                    # "female" | "male"


@dataclass
class SubjectResult:
    subject: Subject
    outcomes: pd.DataFrame
    curves: pd.DataFrame
    curves_normalised: pd.DataFrame = field(default_factory=pd.DataFrame)
    log: list[str] = field(default_factory=list)


def read_subject(folder: str | Path) -> Subject:
    """Read ``subject.csv`` (columns mass_kg, height_m, sex)."""
    folder = Path(folder)
    path = folder / "subject.csv"
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    need = {"mass_kg", "height_m", "sex"}
    if not rows or not need <= set(rows[0]):
        raise ValueError(f"{path}: expected a comma-separated header mass_kg,height_m,sex and one data row")
    row = {k: str(v).strip() for k, v in rows[0].items()}
    sex = row["sex"].lower()
    if sex not in ("female", "male"):
        raise ValueError(f"{path}: sex must be 'female' or 'male', got {row['sex']!r}")
    mass, height = float(row["mass_kg"]), float(row["height_m"])
    if not 20.0 <= mass <= 250.0:
        raise ValueError(f"{path}: mass_kg = {mass} is implausible (expected kilograms)")
    if not 0.8 <= height <= 2.5:
        raise ValueError(f"{path}: height_m = {height} is implausible (expected metres, e.g. 1.74)")
    return Subject(id=folder.name, mass_kg=mass, height_m=height, sex=sex)


def discover(folder: str | Path, settings: Settings) -> tuple[Path, list[tuple[Path, dict]], list[str]]:
    """Static trial, dynamic trials with their leg/plate assignment, and skipped files."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() == ".c3d")
    statics = [f for f in files if is_static(f, settings.static_token)]
    if len(statics) != 1:
        raise ValueError(f"{Path(folder).name}: expected one static c3d containing "
                         f"'{settings.static_token}', found {len(statics)}")
    dynamic, skipped = [], []
    for f in files:
        if f in statics:
            continue
        try:
            lp = parse_leg_plates(f)
        except ValueError as exc:
            skipped.append(f"{exc}, skipped")
            continue
        if lp:
            dynamic.append((f, lp))
        else:
            skipped.append(f"{f.name}: name does not end with a leg/plate token (e.g. _L3), skipped")
    return statics[0], dynamic, skipped


def analyse_subject(folder: str | Path, settings: Settings | None = None,
                    variants: list[Variant] | None = None,
                    curve_variants: tuple[str, ...] = ("ML", "MB")) -> SubjectResult:
    """Analyse all trials of one subject.

    variants        None: ML and/or MB (whatever the files contain) plus, if both are present and
                    ``settings.substitution`` is set, all substitution variants.
    curve_variants  variants whose curves (per frame and time-normalised) are exported; all
                    variants: ``tuple(v.id for v in standard_variants())``."""
    settings = settings or Settings()
    settings.validate()
    folder = Path(folder)
    subject = read_subject(folder)
    static_path, dynamic, log = discover(folder, settings)

    static_sets = prepare_static(read_c3d(static_path), settings)
    if not static_sets:
        raise ValueError(f"{static_path.name}: no markers of the segment model found")
    if variants is None:
        variants = [v for v in standard_variants() if v.id in static_sets]
        if {"ML", "MB"} <= set(static_sets) and settings.substitution:
            variants = standard_variants()
    rows, curve_rows, norm_rows = [], [], []
    calibs = {}
    for v in list(variants):
        try:
            pose = build(v, static_sets, static=True)
            calibs[v.id] = kinematics.calibrate(pose.data, pose.labels, subject.sex, settings.marker_cutoff_hz)
        except Exception as exc:
            log.append(f"{static_path.name}: static calibration of variant {v.id} failed: {type(exc).__name__}: {exc}")
            rows.append({"subject": subject.id, "trial": static_path.stem, "variant": v.id, "stage": v.stage,
                         "error": f"static calibration failed: {type(exc).__name__}: {exc}"})
    variants = [v for v in variants if v.id in calibs]

    for path, leg_plates in dynamic:
        try:
            trial = prepare_dynamic(read_c3d(path), leg_plates, settings)
        except Exception as exc:                      # a broken trial must not stop the subject
            log.append(f"{path.name}: {exc}")
            rows.append({"subject": subject.id, "trial": path.stem, "error": str(exc)})
            continue
        log += [f"{path.name}: {n}" for n in trial.notes]
        for v in variants:
            base = {"subject": subject.id, "trial": trial.name, "variant": v.id, "stage": v.stage}
            needed = {"ML", "MB"} if v.stage > 0 else {v.id}
            if v.stage == 1:
                needed = {"ML"}
            lacking = needed - set(trial.sources)
            if lacking:
                rows.append({**base, "error": f"trial has no {' or '.join(sorted(lacking))} markers"})
                continue
            try:
                dyn = build(v, trial.sources, static=False)
                frames, angles = kinematics.reconstruct(dyn.data, dyn.labels, trial.rate, calibs[v.id],
                                                        settings.marker_cutoff_hz)
            except Exception as exc:
                rows.append({**base, "error": f"{type(exc).__name__}: {exc}"})
                continue
            log += [f"{path.name} [{v.id}]: {w}" for w in frames.pop("warnings", [])]
            for leg, ev in trial.events.items():
                r = {**base, "leg": leg, "plate": ev.plate, "contact_rule": ev.rule,
                     "ic_time_s": ev.td_s, "stance_time_s": ev.to_s - ev.td_s}
                try:
                    mom = kinetics.external_jcs_moments(frames, calibs[v.id], trial, leg, subject.mass_kg,
                                                        subject.height_m, subject.sex, settings.massless_foot)
                    r.update(outcomes.knee_outcomes(angles, mom, ev, trial.rate, settings.analysis_window_s))
                    if v.id in curve_variants:
                        curve_rows += [{**base, "leg": leg, **c}
                                       for c in outcomes.stance_curves(angles, mom, ev, trial.rate)]
                        norm_rows += [{**base, "leg": leg, **c}
                                      for c in outcomes.normalised_curves(angles, mom, ev)]
                except Exception as exc:
                    r["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(r)
    return SubjectResult(subject=subject, outcomes=pd.DataFrame(rows), curves=pd.DataFrame(curve_rows),
                         curves_normalised=pd.DataFrame(norm_rows), log=log)


def analyse_dataset(root: str | Path, out_dir: str | Path, settings: Settings | None = None,
                    **kwargs) -> pd.DataFrame:
    """Analyse every subject folder below *root* that contains a ``subject.csv``.

    Writes ``outcomes.csv``, ``curves.csv``, ``curves_normalised.csv``, ``settings.json`` and
    ``log.txt`` to *out_dir* and
    returns the outcome table."""
    settings = settings or Settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res, cur, cnorm, log = [], [], [], []
    folders = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "subject.csv").exists())
    if not folders:
        raise FileNotFoundError(f"no subject folders with a subject.csv found in {root} "
                                "(pass the folder that CONTAINS the subject folders)")
    for p in sorted(p for p in Path(root).iterdir() if p.is_dir() and p not in folders):
        log.append(f"{p.name}: no subject.csv, folder skipped")
    for folder in folders:
        try:
            r = analyse_subject(folder, settings, **kwargs)
        except Exception as exc:
            log.append(f"{folder.name}: {type(exc).__name__}: {exc}")
            continue
        res.append(r.outcomes)
        cur.append(r.curves)
        cnorm.append(r.curves_normalised)
        log += [f"{folder.name}: {line}" for line in r.log]
    table = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    table.to_csv(out_dir / "outcomes.csv", index=False)
    (pd.concat(cur, ignore_index=True) if cur else pd.DataFrame()).to_csv(out_dir / "curves.csv", index=False)
    (pd.concat(cnorm, ignore_index=True) if cnorm else pd.DataFrame()).to_csv(
        out_dir / "curves_normalised.csv", index=False)
    (out_dir / "settings.json").write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    return table
