"""Visual check of the c3d files of one subject before the analysis.

For every file it plots (1) the vertical force of all plates with the detected initial contact
(IC) and toe-off (TO), (2) a frontal and (3) a sagittal stick figure of the marker-based and the
markerless markers at IC, and (4) which markers are present over time. Problems with marker
names, leg or plate tokens and synchronisation are visible at a glance.

Usage:  python examples/check_data.py <subject_folder> [output.png]
Needs matplotlib (pip install matplotlib).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from markersub import Settings
from markersub.c3d_io import read_c3d
from markersub.markers import split_sources
from markersub.naming import is_static, parse_leg_plates
from markersub.preprocessing import prepare_dynamic

COLOURS = {"MB": "tab:blue", "ML": "tab:orange"}


def chain(pos: dict, side: str) -> np.ndarray:
    """Pelvis (ASIS), thigh and shank cluster centres, heel and big toe of one side."""
    ff = "forefoot" if side == "left" else "forfoot"
    def centre(names):
        pts = [pos[n] for n in names if n in pos]
        return np.mean(pts, axis=0) if pts else np.full(3, np.nan)
    pts = [pos.get(f"SIAS_{side}", np.full(3, np.nan)),
           centre([f"cluster_femur_{side}_{i}" for i in (1, 2, 3)]),
           centre([f"cluster_tibia_{side}_{i}" for i in (1, 2, 3)]),
           pos.get(f"calc_back_{side}", np.full(3, np.nan)),
           pos.get(f"{ff}_med_{side}", np.full(3, np.nan))]
    return np.array(pts)


def main(folder: Path, out: Path, settings: Settings) -> None:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".c3d")
    fig, axs = plt.subplots(len(files), 4, figsize=(12, 2.8 * len(files)), squeeze=False)
    for r, path in enumerate(files):
        tr = read_c3d(path)
        src = split_sources(tr.labels, settings.mb_prefix)
        frame, ev, title = tr.n_frames // 2, None, path.name
        if not is_static(path, settings.static_token):
            lp = parse_leg_plates(path)
            if lp:
                try:
                    p = prepare_dynamic(tr, lp, settings)
                    leg = sorted(p.events)[0]
                    ev = p.events[leg]
                    frame = int(round(ev.td_s * tr.rate))
                    title += f"  ({leg} leg, plate {ev.plate}, rule {ev.rule})"
                    for note in p.notes:
                        print(f"{path.name}: {note}")
                except Exception as exc:
                    title += f"  ERROR: {exc}"
            else:
                title += "  (no leg token, not analysed)"
        ax = axs[r, 0]
        if tr.plates:
            t = np.arange(tr.plates[0].force.shape[1]) / tr.analog_rate
            for k, plate in enumerate(tr.plates):
                ax.plot(t, plate.force[2], lw=0.8, label=f"Plate {k + 1}")
            ax.legend(frameon=False, fontsize=7)
        if ev is not None:
            for x in (ev.td_s, ev.to_s):
                ax.axvline(x, color="k", ls=":", lw=0.8)
        ax.set_ylabel("Vertical force [N]")
        ax.set_xlabel("Time [s]")
        ax.set_title(title, fontsize=8, loc="left")
        for c, (i0, i1, xl) in enumerate(((1, 2, "Y [mm]"), (0, 2, "X [mm]"))):
            ax = axs[r, 1 + c]
            for s, col in COLOURS.items():
                pos = {n: tr.xyz[:, i, frame] for i, n in src[s] if np.all(np.isfinite(tr.xyz[:, i, frame]))}
                for side, ls in (("left", "-"), ("right", "--")):
                    ch = chain(pos, side)
                    ax.plot(ch[:, i0], ch[:, i1], ls, color=col, marker="o", ms=2, lw=1)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(xl)
            ax.set_ylabel("Z [mm]")
        ax = axs[r, 3]
        rows = [c for c, _ in src["MB"]] + [c for c, _ in src["ML"]]
        if rows:
            present = np.all(np.isfinite(tr.xyz[:, rows, :]), axis=0).astype(float)
            ax.imshow(present, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                      extent=[0, tr.n_frames / tr.rate, len(rows), 0])
            ax.axhline(len(src["MB"]), color="r", lw=0.8)
        ax.set_yticks([len(src["MB"]) / 2, len(src["MB"]) + len(src["ML"]) / 2])
        ax.set_yticklabels([f"MB ({len(src['MB'])})", f"ML ({len(src['ML'])})"])
        ax.set_xlabel("Time [s]")
    axs[0, 1].plot([], [], color=COLOURS["MB"], label="marker-based")
    axs[0, 1].plot([], [], color=COLOURS["ML"], label="markerless")
    axs[0, 1].plot([], [], "k-", label="left")
    axs[0, 1].plot([], [], "k--", label="right")
    axs[0, 1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"written {out}")


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "data_check.png"
    main(folder, out, Settings())
