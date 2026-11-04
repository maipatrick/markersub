# Processing chain

All steps are identical for marker-based (MB) and markerless (ML) markers. Only the input
treatment differs: MB markers are gap-filled and filtered with `marker_cutoff_hz`; ML markers are
used as delivered (no gap filling) or filtered with `ml_marker_cutoff_hz`. The chain implements
published methods only; the references are listed at the end.

## 1. Static calibration (`kinematics.calibrate`)

1. MB markers of the static trial: linear gap filling and 4th-order Butterworth low-pass applied
   forward and backward (`marker_cutoff_hz`, Winter 2009). ML markers: used as delivered (or filtered with `ml_marker_cutoff_hz`).
2. The time average of every marker over all frames defines the standing posture.
3. The posture is rotated about the vertical axis so that the pelvis faces +X (mid-ASIS minus
   mid-PSIS, horizontal component).
4. Joint centres: hip after Harrington et al. (2007) from ASIS, PSIS and leg length (ASIS to
   medial epicondyle to medial malleolus); knee at the midpoint of
   the femoral epicondyles; ankle at the midpoint of the malleoli.
5. The technical-to-anatomical rotation of each segment is the mean over the static frames,
   projected onto the rotation group (Moakher 2002).
6. Anatomical frames of pelvis, thigh, shank and foot (foot: heel, 5th metatarsal head, big toe) from the ISB landmarks (Wu et al. 2002; axes
   relabelled X anterior, Y left, Z up) and the constant transformation from each technical
   (tracking marker) frame to its anatomical frame (Cappozzo et al. 1995).

## 2. Dynamic trials (`preprocessing.prepare_dynamic`)

1. **Contact detection** on the raw vertical force of the plate named in the file name (force
   threshold method, Tirosh and Sparrow 2003): a contact is a run above `force_threshold_n`
   lasting at least `min_contact_s`; initial contact (IC) and
   toe-off are its first and last sample. Rules for several contacts: see `contacts.py`.
2. **Crop** to the analysed contact(s) plus `crop_buffer_frames` on both sides, markers and force
   on one common time grid.
3. **Markers:** MB linear gap filling and low-pass (`marker_cutoff_hz`); ML as delivered or
   low-pass (`ml_marker_cutoff_hz`), never gap-filled.
4. **Force plate:** missing samples filled with the next valid sample (trailing gaps with the last); force and plate moment
   low-pass filtered (`force_cutoff_hz`, half-cosine padding against edge ringing); centre of
   pressure discarded below `cop_valid_threshold_n` raw vertical force, filled and low-pass
   filtered; free moment computed as the residual of the filtered plate wrench at the filtered
   centre of pressure.

## 3. Kinematics (`kinematics.reconstruct`)

Technical frames are fitted to the tracking markers of each segment in every frame by singular
value decomposition (Soederkvist and Wedin 1993; Arun et al. 1987) relative to the static
positions. A tracking marker missing for the whole trial is reconstructed from the static
geometry of its segment (rigid-body assumption, Cappozzo et al. 1995); this is reported in the
log. The anatomical
frames follow from the static calibration. Joint angles are Cardan angles of the distal relative
to the proximal anatomical frame (flexion, adduction, internal rotation). Hip flexion is reported
positive.

## 4. Kinetics (`kinetics.external_jcs_moments`)

Segment masses, centres of mass and moments of inertia from sex-specific regressions (de Leva
1996), scaled with body mass, height and the measured segment lengths. Newton-Euler inverse
dynamics from the foot upwards (Hof 1992); the ground reaction force is applied only between IC
and toe-off of the analysed leg. Segment linear velocities are low-pass filtered
(`marker_cutoff_hz`, 2nd-order Butterworth applied forward and backward) before differentiation. The net joint moment is expressed as the external
moment and projected orthogonally onto the axes of the joint coordinate system (Grood and Suntay
1983; projection as discussed by Desroches et al. 2010): flexion axis of the proximal segment,
floating axis, long axis of the distal segment. It is
normalised to body mass [Nm/kg]. Moments are computed at the force rate and interpolated to the
marker frames; peaks are taken at marker rate. With `massless_foot` the foot has no mass or inertia.

## 5. Outcomes (`outcomes.py`)

Peak external knee abduction and flexion moments and knee frontal-plane and flexion angles in
the window from IC to IC + `analysis_window_s`.

## 6. Marker substitution (`substitution.py`)

A variant takes a defined group of markers from MB and all others from ML, separately for the
static and the dynamic trials. Stages 1 to 3 keep the markerless foot; stage 4 substitutes the
foot (alone, and together with the distal chain). The static calibration and every dynamic trial are then processed
exactly as above.

## References

* Arun KS, Huang TS, Blostein SD (1987) Least-squares fitting of two 3-D point sets. IEEE Trans Pattern Anal Mach Intell 9:698-700.
* Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Position and orientation in space of bones during movement: anatomical frame definition and determination. Clin Biomech 10:171-178.
* Desroches G, Cheze L, Dumas R (2010) Expression of joint moment in the joint coordinate system. J Biomech Eng 132:114503. https://doi.org/10.1115/1.4002537
* de Leva P (1996) Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. J Biomech 29:1223-1230.
* Grood ES, Suntay WJ (1983) A joint coordinate system for the clinical description of three-dimensional motions: application to the knee. J Biomech Eng 105:136-144.
* Harrington ME, Zavatsky AB, Lawson SEM, Yuan Z, Theologis TN (2007) Prediction of the hip joint centre in adults, children, and patients with cerebral palsy based on magnetic resonance imaging. J Biomech 40:595-602.
* Hof AL (1992) An explicit expression for the moment in multibody systems. J Biomech 25:1209-1211.
* Jiang T, Lu P, Zhang L, et al. (2023) RTMPose: real-time multi-person pose estimation based on MMPose. arXiv:2303.07399.
* Michaud B, Begon M (2021) ezc3d: an easy C3D file I/O cross-platform solution for C++, Python and MATLAB. J Open Source Softw 6:2911.
* Moakher M (2002) Means and averaging in the group of rotations. SIAM J Matrix Anal Appl 24:1-16. https://doi.org/10.1137/S0895479801383877
* Soederkvist I, Wedin PA (1993) Determining the movements of the skeleton using well-configured markers. J Biomech 26:1473-1477.
* Tirosh O, Sparrow WA (2003) Identifying heel contact and toe-off using forceplate thresholds with a range of digital-filter cutoff frequencies. J Appl Biomech 19:178-184. https://doi.org/10.1123/jab.19.2.178
* Uhlrich SD, Falisse A, Kidzinski L, et al. (2023) OpenCap: human movement dynamics from smartphone videos. PLoS Comput Biol 19:e1011462.
* Winter DA (2009) Biomechanics and motor control of human movement, 4th ed. Wiley, Hoboken.
* Wu G, Siegler S, Allard P, et al. (2002) ISB recommendation on definitions of joint coordinate system of various joints for the reporting of human joint motion, part I: ankle, hip, and spine. J Biomech 35:543-548.
