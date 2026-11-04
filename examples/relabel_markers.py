"""Rename marker-based labels to the ``MB_<markerless name>`` convention.

Fill MAPPING with your own marker names (left column) once; the script then writes relabelled
copies of all c3d files of a folder. Markers that are not listed are kept unchanged (or removed
with DROP_UNMAPPED = True). Analog and force-plate data are copied unchanged.

Usage:  python examples/relabel_markers.py <input_folder> <output_folder>
"""
import sys
from pathlib import Path

from markersub.c3d_io import relabel_c3d
from markersub.markers import REQUIRED_MARKERS

MAPPING = {
    # "your label": "MB_<markerless name>",
    # "LASI": "MB_L.ASIS_study",
    # "RASI": "MB_r.ASIS_study",
    # "LPSI": "MB_L.PSIS_study",
    # "RPSI": "MB_r.PSIS_study",
    # "LKNE": "MB_L_knee_study",
    # "LKNM": "MB_L_mknee_study",
    # "LANK": "MB_L_ankle_study",
    # "LMED": "MB_L_mankle_study",
    # "LHEE": "MB_L_calc_study",
    # "LTOE": "MB_LBigToe",          # marker on the big toe
    # "LMT5": "MB_L_5meta_study",
    # "LTH1": "MB_L_thigh1_study", "LTH2": "MB_L_thigh2_study", "LTH3": "MB_L_thigh3_study",
    # "LSH1": "MB_L_sh1_study", "LSH2": "MB_L_sh2_study", "LSH3": "MB_L_sh3_study",
    # ... and the same for the right side (r_ / r.)
}
DROP_UNMAPPED = False

if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    missing = sorted(set("MB_" + m for m in REQUIRED_MARKERS) - set(MAPPING.values()))
    if missing:
        print("Not mapped (needed for the full model):", ", ".join(missing))
    for f in sorted(src.glob("*.c3d")):
        relabel_c3d(f, dst / f.name, MAPPING, drop_unmapped=DROP_UNMAPPED)
        print("written", dst / f.name)
