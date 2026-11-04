# markersub

Segment-wise six-degree-of-freedom lower-limb kinematics and kinetics for marker-based (MB) and
markerless (ML) motion capture, with a marker substitution analysis that locates which landmarks
limit markerless knee frontal-plane biomechanics.

The package reads c3d files, detects initial contact on the force plate, reconstructs pelvis,
thigh, shank and foot with a classic calibrated six-degree-of-freedom segment model, computes
Cardan joint angles and external joint moments by Newton-Euler inverse dynamics, and reports knee
outcomes in early stance. MB and ML data run through exactly the same chain. Because both use the
same marker names, any group of ML markers can be replaced by its MB counterpart.

Companion article: Mai P, Bill K, Mausehund L, Rice H, Werkhausen A, Willwacher S, Pasanen K,
Krosshaug T. *Which landmarks limit markerless knee frontal-plane
biomechanics? A marker substitution analysis in cutting.* Gait & Posture (submitted).

## Installation

Python 3.10 or newer. c3d files are read and written with ezc3d (Michaud and Begon 2021,
https://github.com/pyomeca/ezc3d); files written by ezc3d carry an `EZC3D` parameter group
with its version and contact address, as in the example data.

```bash
git clone https://github.com/maipatrick/markersub.git
cd markersub
pip install -e .[test,plot]  # installs numpy, scipy, pandas, ezc3d, pytest and matplotlib
python examples/run_example.py
pytest
```

With conda: `conda install -c conda-forge ezc3d numpy scipy pandas`, then `pip install -e . --no-deps`.

## Data conventions

### Folder layout

```
data/
  S01/
    subject.csv              mass_kg,height_m,sex   (one row; sex = female or male)
    S01_static.c3d           static standing trial
    S01_cut01_L3.c3d         dynamic trials
    S01_cut02_L3.c3d
  S02/
    ...
```

### File names

* **Static trial:** the file name contains the static token (default `static`, case-insensitive,
  `Settings.static_token`). Exactly one static trial per subject folder.
* **Dynamic trials:** the file name ENDS with one token per analysed leg, `L<n>` or `R<n>`
  (upper case), where `n` is the 1-based number of the force platform in the c3d. Tokens are
  separated by `_`, `-` or a space. Only trailing tokens count, so a subject code such as `R01` at
  the start of the name is not read as a leg. Naming the same leg twice or both legs on one plate
  is an error; such files are skipped and reported in `log.txt`.

| File name | Analysed |
|---|---|
| `S01_cut01_L3.c3d` | left leg on plate 3 |
| `drop_jump_L1_R2.c3d` | left leg on plate 1 and right leg on plate 2 |
| `R01_cut01_L3.c3d` | left leg on plate 3 (`R01` is not a trailing token) |
| `S01_walk.c3d` | skipped (no leg token) |

### Marker names

Markerless markers keep the names of the OpenCap marker augmenter (Uhlrich et al. 2023).
Marker-based markers carry **the same names with the prefix `MB_`**. To use your own
marker-based data, rename your markers once (see `examples/relabel_markers.py`). A c3d may
contain ML markers, MB markers or both. Other markers in the file are ignored.

| Landmark / role | ML name (left) | MB name (left) |
|---|---|---|
| Anterior superior iliac spine | `L.ASIS_study` | `MB_L.ASIS_study` |
| Posterior superior iliac spine | `L.PSIS_study` | `MB_L.PSIS_study` |
| Thigh tracking markers 1 to 3 | `L_thigh1_study` ... `L_thigh3_study` | `MB_L_thigh1_study` ... |
| Lateral / medial femoral epicondyle | `L_knee_study` / `L_mknee_study` | `MB_L_knee_study` / `MB_L_mknee_study` |
| Shank tracking markers 1 to 3 | `L_sh1_study` ... `L_sh3_study` | `MB_L_sh1_study` ... |
| Lateral / medial malleolus | `L_ankle_study` / `L_mankle_study` | `MB_L_ankle_study` / `MB_L_mankle_study` |
| Heel | `L_calc_study` | `MB_L_calc_study` |
| 5th metatarsal head | `L_5meta_study` | `MB_L_5meta_study` |
| Big toe | `LBigToe` (raw pose-estimation keypoint) | `MB_LBigToe` |

Right side: replace `L` by `r` (`r.ASIS_study`, `r_knee_study`, ...) and `LBigToe` by `RBigToe`.
The full list is `markersub.markers.LSTM_TO_MODEL`.

Notes on the pairing:
* Anatomical landmarks (ASIS, PSIS, epicondyles, malleoli, heel, 5th metatarsal head, big toe)
  are paired by location.
* Thigh and shank tracking markers (`thigh1` to `thigh3`, `sh1` to `sh3`) are paired by their
  role in the cluster only. Marker-based tracking markers sit wherever they were attached, so
  their positions differ from the markerless ones; this does not matter for the analysis because
  each segment is tracked by its own cluster.
* Foot: the marker augmenter has no marker on the big toe or the 1st metatarsal head. The front
  point of the foot is therefore the raw big-toe keypoint of the pose estimator (e.g. RTMPose,
  HALPE-26 keypoints), paired with a marker-based marker on the big toe. The foot frame is built
  from heel, 5th metatarsal head and big toe for both sources.

### Requirements on the data

* Laboratory frame with Z vertical (up); coordinates in mm or m (declared in `POINT:UNITS`).
* Force platforms stored in the c3d (corners, calibration) so that force, centre of pressure and
  plate moment can be computed (ezc3d `extract_forceplat_data`). The vertical force must be
  positive upward.
* **Static trial:** all 30 markers of a source (pelvis, three tracking markers per thigh and
  shank, epicondyles, malleoli, heel, 5th metatarsal head, big toe). It is a quiet standing
  posture; the time average of every marker over all frames is used.
* **Dynamic trials:** pelvis, thigh and shank tracking markers and the foot markers. Epicondyle
  and malleolus markers are optional (they are only needed in the static trial).
* ML trajectories synchronised with the force platforms and stored at the c3d point rate. They
  must cover the analysed window, i.e. from initial contact minus `crop_buffer_frames` to toe-off
  plus `crop_buffer_frames`; outside this window they may be missing.
* ML markers are used as delivered (never gap-filled). An ML marker with missing samples in the
  analysed window is dropped. Missing MB samples are filled linearly.

## Analysing your own data, step by step

1. **Install** the package (see Installation) and run `pytest`. All tests must pass; this
   confirms that your installation reproduces the reference results of the example subject.
2. **Rename the marker-based labels once.** Copy `examples/relabel_markers.py`, fill the
   `MAPPING` table with your own labels (e.g. `"LKNE": "MB_L_knee_study"`) and run
   `python relabel_markers.py <raw_folder> <renamed_folder>`. The script lists every model marker
   you have not mapped yet.
3. **Prepare the markerless data.** The c3d needs the augmenter markers (`L_knee_study`, ...)
   and the big-toe keypoints (`LBigToe`, `RBigToe`), synchronised with the force platforms and
   at the c3d point rate. MB and ML markers may be in the same c3d or in separate files (one
   folder per subject and source).
4. **Arrange the folders.** One folder per subject with `subject.csv`
   (`mass_kg,height_m,sex`, e.g. `65.8,1.74,female`), one static trial (name contains
   `static`) and the dynamic trials ending with the leg and plate token (e.g. `_L3`).
5. **Check the data visually** before the analysis:

   ```bash
   pip install matplotlib
   python examples/check_data.py data/S01
   ```

   This writes `data/S01/data_check.png`. Check for every file:
   * the force curve of the named plate rises at IC and falls at TO (dotted lines), and the
     vertical force is positive;
   * the stick figures of MB (blue) and ML (orange) at IC lie on top of each other and the left
     (solid) and right (dashed) legs are not swapped. Large offsets point to wrong marker names,
     wrong synchronisation or a different coordinate system;
   * the marker availability shows all markers of both sources across the analysed window;
   * the console prints no warning such as "foot markers of the left leg are not over plate 3".
6. **Run the analysis on one subject first**, then on all:

   ```python
   from markersub import Settings, analyse_subject, analyse_dataset
   res = analyse_subject("data/S01", Settings())
   print(res.log)                                   # every warning of this subject
   print(res.outcomes[["trial", "variant", "leg", "peak_kam_nmkg", "contact_rule"]])
   table = analyse_dataset("data", "results", Settings())
   ```
7. **Check the results.**
   * `log.txt` lists skipped files, filled gaps, reconstructed markers and leg warnings.
   * Rows with a value in `error` were not analysed; the text says why.
   * `contact_rule` should be `single` for ordinary trials; `loaded_at_start` means that the
     recording started during stance (IC not observed).
   * Plausible ranges for cutting (as orientation): stance time 0.15 to 0.40 [s], knee flexion
     at IC 10 to 40 [deg], peak external knee abduction moment roughly -0.5 to 3 [Nm/kg].
8. **Use the curves.** `curves_normalised.csv` holds hip, knee and ankle angles and moments in
   all three axes from IC (0 %) to toe-off (100 %) on 101 points. One row per curve:

   ```python
   import pandas as pd
   c = pd.read_csv("results/curves_normalised.csv")
   wide = c.pivot_table(index=["subject", "trial", "variant", "leg", "joint", "quantity", "component"],
                        columns="percent_stance", values="value")
   knee_kam = wide.xs(("knee", "external_moment_nmkg", "abduction"),
                      level=["joint", "quantity", "component"])
   ```

## Quick start

```python
from markersub import Settings, analyse_dataset

settings = Settings(static_token="static", marker_cutoff_hz=20.0, force_threshold_n=20.0)
table = analyse_dataset("example_data", "results", settings)
```

`examples/run_example.py` runs the example subject and prints a summary.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `static_token` | `"static"` | token of the static trial in the file name |
| `mb_prefix` | `"MB_"` | prefix of marker-based labels |
| `marker_cutoff_hz` | 20.0 | low-pass cut-off of MB markers, the static trial and segment velocities [Hz], below 50 |
| `ml_marker_cutoff_hz` | None | low-pass cut-off of ML markers [Hz]; None = use as delivered |
| `gap_fill` | True | linear interpolation of missing MB samples (ML samples are never filled) |
| `crop_buffer_frames` | 20 | frames kept before initial contact and after toe-off [frames] |
| `force_cutoff_hz` | 20.0 | low-pass cut-off of force, centre of pressure and plate moment [Hz] |
| `force_threshold_n` | 20.0 | contact threshold on the raw vertical force [N] |
| `min_contact_s` | 0.100 | minimum duration of a contact [s] |
| `cop_valid_threshold_n` | 50.0 | centre of pressure used only above this raw vertical force [N] |
| `analysis_window_s` | 0.100 | early-stance window after initial contact [s] |
| `massless_foot` | True | foot segment without mass and inertia (as in the article); False = full inverse dynamics |
| `substitution` | True | run the substitution variants when both sources are present |

## Outputs

`analyse_dataset` writes to the output folder:

* `outcomes.csv`: one row per subject, trial, variant and analysed leg (columns `subject`,
  `trial`, `variant`, `stage`, `leg`, `plate` and the outcomes below). A trial or variant that
  could not be processed gets a row with the reason in `error`.

  | Column | Unit | Meaning |
  |---|---|---|
  | `peak_kam_nmkg` | [Nm/kg] | peak external knee abduction moment, 0 to 100 [ms] after IC |
  | `peak_kfm_nmkg` | [Nm/kg] | peak external knee flexion moment, same window |
  | `knee_frontal_ic_deg` | [°] | knee frontal-plane angle at IC (adduction positive, abduction or valgus negative) |
  | `knee_frontal_min_deg` | [°] | minimum frontal-plane angle in the window |
  | `knee_frontal_mean_deg` | [°] | mean frontal-plane angle in the window |
  | `knee_flexion_ic_deg` | [°] | knee flexion at IC |
  | `ic_time_s`, `stance_time_s` | [s] | initial contact time in the file, contact duration |
  | `contact_rule` | | how the analysed contact was chosen (see `contacts.py`) |

* `curves_normalised.csv`: the same signals time-normalised from IC (0 %) to toe-off (100 %) on
  101 points (`percent_stance`), all three axes.
* `curves.csv`: hip, knee and ankle angles [°] and external moments [Nm/kg] from IC to toe-off
  for the ML and MB reference sets, long format (`joint`, `quantity`, `component`,
  `frame_from_ic`, `time_from_ic_s`, `value`). Angle components: flexion, adduction, internal
  rotation (Cardan, distal relative to proximal segment). Moment components: external flexion,
  abduction and internal rotation moments, orthogonal projections onto the joint coordinate
  system axes.
* `settings.json`, `log.txt`.

## Marker substitution

With both sources present, the analysis starts from the ML marker set and replaces groups of ML
markers by their MB counterparts (both legs):

| Stage | Variant | Markers taken from MB |
|---|---|---|
| 0 | `ML`, `MB` | none, all |
| 1, static landmarks | `S1_M`, `S1_E`, `S1_P`, `S1_ME`, `S1_MP`, `S1_EP`, `S1_MEP` | malleoli (M), epicondyles (E), pelvis (P) in the static trial only |
| 2, tracking markers | `S2_T`, `S2_F`, `S2_FT`, `S2_PT`, `S2_PF`, `S2_PFT` | shank cluster (T), thigh cluster (F), pelvis (P), static and dynamic |
| 3, segment-wise | `S3_prox`, `S3_dist`, `S3_both`, `S3_both_P` | epicondyles and thigh cluster, malleoli and shank cluster, both, both and pelvis |
| 4, foot | `S4_foot`, `S4_dist_foot` | foot markers (heel, 5th metatarsal head, big toe); malleoli, shank cluster and foot |

Stages 1 to 3 keep the ML foot (as in the article), so `S3_both_P` differs from `MB` only in the
foot. With the massless foot (default) the knee outcomes do not depend on the foot markers and
`S3_both_P` equals `MB` for the knee; stage 4 substitutes the foot, which matters for the ankle
and, with full inverse dynamics (`massless_foot=False`), also for the knee.

The share of the ML-MB difference that a substitution removes indicates how much the substituted
landmarks limit the ML result.

Own variants take internal model names (`markersub.markers.GROUPS`, `LSTM_TO_MODEL`); the ids
`ML` and `MB` are reserved:

```python
from markersub import Variant, analyse_subject
from markersub.markers import GROUPS
v = Variant("knee_landmarks", 1, frozenset(GROUPS["epicondyles"]), frozenset(), "epicondyles in the static")
res = analyse_subject("example_data/S01", variants=[v])
```

## Methods and sources

The chain implements established, published methods; it introduces no new method for
calibration, joint angles or inverse dynamics. Details: [docs/method.md](docs/method.md).

| Step | Method | Source |
|---|---|---|
| Contact detection | threshold on the raw vertical ground reaction force | Tirosh and Sparrow 2003 |
| Filtering | 4th-order Butterworth low-pass, applied forward and backward | Winter 2009 |
| Static calibration | technical (tracking) and anatomical frames, calibrated anatomical systems | Cappozzo et al. 1995 |
| Hip joint centre | regression on pelvis width, depth and leg length | Harrington et al. 2007 |
| Segment frames | ISB anatomical landmarks (axes relabelled: X anterior, Y left, Z up) | Wu et al. 2002 |
| Segment pose | least-squares rigid-body fit by singular value decomposition | Soederkvist and Wedin 1993; Arun et al. 1987 |
| Joint angles | Cardan angles, distal relative to proximal segment | Grood and Suntay 1983 |
| Segment parameters | sex-specific mass, centre of mass and radii of gyration | de Leva 1996 |
| Inverse dynamics | Newton-Euler, foot to thigh | Hof 1992 |
| External joint moments | orthogonal projection onto the joint coordinate system axes | Grood and Suntay 1983; Desroches et al. 2010 |
| Mean static rotation | mean rotation projected onto the rotation group | Moakher 2002 |
| Markerless marker set | OpenCap marker augmenter names; big-toe keypoint of the pose estimator | Uhlrich et al. 2023; Jiang et al. 2023 |

The functions that implement each step cite their source in the docstring.

## Example data and tests

`example_data/S01` holds one anonymised participant (body mass, height and sex only) (static trial and three cutting trials, left
leg on plate 3) with both marker sources. `tests/` contains unit tests and a regression test that
reproduces `tests/reference/S01_outcomes.csv`.

```bash
pip install pytest
pytest
```

## Limitations

* Lower extremity and pelvis only (no trunk or arms, no whole-body centre of mass).
* One contact per leg and trial is analysed.
* Missing MB samples are filled linearly. Fill longer gaps in the capture software before export.
* A tracking marker that is missing for a whole trial is reconstructed from the static geometry
  of its segment (rigid-body assumption); such cases are reported in `log.txt`.
* Initial contact is taken from the force plate; a recording that already starts in contact is
  flagged as `contact_rule = loaded_at_start`.
* Plausibility check of the file name: if the foot markers of the named leg are not over the
  named plate at peak force, `log.txt` reports it (wrong leg or plate token).

## Citation

Please cite the companion article (see `CITATION.cff`).

## Licence

MIT, see `LICENSE`.
