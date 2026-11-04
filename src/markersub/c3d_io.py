"""Reading and relabelling c3d files (ezc3d).

Coordinates are returned in the laboratory frame of the c3d (Z up) in millimetres; files
exported in metres are scaled to millimetres from the declared POINT:UNITS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Plate:
    """One force platform (laboratory frame)."""
    force: np.ndarray        # (3, N) [N]
    cop: np.ndarray          # (3, N) [mm]
    moment_ctr: np.ndarray   # (3, N) [N mm], plate moment about the plate centre
    centre: np.ndarray       # (3,)  [mm], mean of the four corners
    corners: np.ndarray      # (3, 4) [mm]


@dataclass
class C3DTrial:
    """Raw content of one c3d file."""
    path: Path
    labels: list[str]
    xyz: np.ndarray          # (3, n_markers, n_frames) [mm]
    residual: np.ndarray     # (n_markers, n_frames), < 0 = missing sample
    rate: float              # marker rate [Hz]
    first_frame: int
    plates: list[Plate] = field(default_factory=list)
    analog_rate: float = 0.0  # force rate [Hz]

    @property
    def n_frames(self) -> int:
        return int(self.xyz.shape[2])

    @property
    def name(self) -> str:
        return self.path.stem


def _mm_scale(c) -> float:
    try:
        unit = str(c["parameters"]["POINT"]["UNITS"]["value"][0]).strip().lower()
    except Exception:
        return 1.0
    return 1000.0 if unit in ("m", "meter", "metre") else 1.0


def read_c3d(path: str | Path) -> C3DTrial:
    """Read markers and force platforms. Unlabelled markers (``*12``) are skipped."""
    import ezc3d

    path = Path(path)
    c = ezc3d.c3d(str(path), extract_forceplat_data=True)
    mm = _mm_scale(c)
    pts = c["data"]["points"]
    labels = list(c["parameters"]["POINT"]["LABELS"]["value"])
    keep = [i for i, lbl in enumerate(labels) if not lbl.strip().startswith("*")]
    labels = [labels[i].strip() for i in keep]
    xyz = pts[:3, keep, :].astype(float) * mm
    residual = pts[3, keep, :].astype(float)
    plates = []
    analog_rate = 0.0
    if c["data"]["platform"]:
        analog_rate = float(c["parameters"]["ANALOG"]["RATE"]["value"][0])
        for p in c["data"]["platform"]:
            corners = np.asarray(p["corners"], dtype=float) * mm
            plates.append(Plate(force=np.asarray(p["force"], dtype=float),
                                cop=np.asarray(p["center_of_pressure"], dtype=float) * mm,
                                moment_ctr=np.asarray(p["moment"], dtype=float) * mm,
                                centre=corners.mean(axis=1), corners=corners))
    return C3DTrial(path=path, labels=labels, xyz=xyz, residual=residual,
                    rate=float(c["parameters"]["POINT"]["RATE"]["value"][0]),
                    first_frame=int(c["header"]["points"]["first_frame"]),
                    plates=plates, analog_rate=analog_rate)


def relabel_c3d(src: str | Path, dst: str | Path, mapping: dict[str, str],
                drop_unmapped: bool = False) -> list[str]:
    """Write a copy of *src* with marker labels renamed by *mapping* (old -> new).

    Typical use: rename marker-based labels to the ``MB_<markerless name>`` convention, e.g.
    ``{"LKNE": "MB_L_knee_study", "LKNM": "MB_L_mknee_study", ...}``. Analog and force-plate
    data are copied unchanged. With ``drop_unmapped`` the unmapped markers are removed.
    Returns the new label list."""
    import ezc3d

    c = ezc3d.c3d(str(src))
    labels = [str(lbl).strip() for lbl in c["parameters"]["POINT"]["LABELS"]["value"]]
    if drop_unmapped:
        keep = [i for i, lbl in enumerate(labels) if lbl in mapping]
        c["data"]["points"] = c["data"]["points"][:, keep, :]
        labels = [labels[i] for i in keep]
        if "meta_points" in c["data"]:
            del c["data"]["meta_points"]
    new = [mapping.get(lbl, lbl) for lbl in labels]
    if len(set(new)) != len(new):
        raise ValueError("relabelling produces duplicate marker labels")
    c["parameters"]["POINT"]["LABELS"]["value"] = new
    if "DESCRIPTIONS" in c["parameters"]["POINT"]:
        c["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * len(new)
    c.write(str(dst))
    return new
