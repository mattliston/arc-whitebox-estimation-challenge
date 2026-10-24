# Float32 covariance transport saved 5.3–5.4% adjusted on the 14-network panel

2026-09-04 · experiment · informational · measured

f32cov PANEL (14 nets, grader frame): panel8 adjusted 1.697e-8 vs CAP256 1.792e-8 (-5.3%; raw identical 3.276e-8; util 0.5174 vs 0.5466); virgin6 1.359e-8 vs 1.436e-8 (-5.4%; raw identical; util 0.5119 vs 0.5411). The grader bills the covariance-push saving at -5.4% (the instrument said -8.5%). Below the 8-10% fire bar -> HOLD; combine with the third-cumulant canonical-polyadic carrier (CP3) 8192 co-adapted chain (T-140) for a >=15% fire.

Recorded measurement:

    - T-138 FREE UTIL: the covariance push W^T C W runs in float64 (billed 2x). F32_COV=1 (exists in card + instrument): g29 host 7.848e-8 vs 7.847e-8 at util 0.571 vs 0.627; on the FIRED config (CAP256 + closure): r35 2.426e-8 (vs 2.424) / r100 2.669e-8 (vs 2.699) at util 0.507/0.525 (vs 0.554): identical accuracy, -7% util -> adjusted -7..-8%. Next: the cubic and quartic pushes are also f64 (9.1 + 13.7% of util): F32_PUSH34=1 added to the instrument (casts directed/diagonal to f32); testing on r35/r100.
    
    - T-138b: cand_gnn_hcap256_f32cov.py (BUILD_CP4_CAP=256 BUILD_F32_COV=1) reproduces the instrument on r35 (2.424e-8 vs 2.426e-8 f32cov instrument). Builder now also has BUILD_F32_PUSH34 (casts at the 4 push sites). Waiting for the f32all instrument read (r35/r100) to decide between paneling the f32cov candidate (-7%) or the combined one (-15..-18% if the pushes hold).
    
    - T-138c: F32_PUSH34 changes NOTHING (f32all = f32cov exactly: r35 2.426e-8 util 0.5073, r100 2.669e-8 util 0.5249): the cubic/quartic pushes are already float32 (their f64 mentions are the outputs). The only f64-billed block was the covariance push. Free cut = -7% (F32_COV). Looking for one more free cut to make a >=10% fire: CP4 cap 128, Strassen on the remaining dense GEMMs.
    
    - T-138d CP4 cap 128 (+F32_COV): r35 2.545e-8 at 0.5023 / r100 2.811e-8 at 0.5199 vs cap 256: 2.520/2.790 at 0.5073/0.5249 (instrument frame): raw +1%, util -1% = a WASH. The fourth-order cap is at its optimum at 256. Free-cut inventory exhausted: F32_COV (-7%) is the fire; panel running.
    
    - T-138e f32cov PANEL (14 nets, grader frame): panel8 adjusted 1.697e-8 vs CAP256 1.792e-8 (-5.3%; raw identical 3.276e-8; util 0.5174 vs 0.5466); virgin6 1.359e-8 vs 1.436e-8 (-5.4%; raw identical; util 0.5119 vs 0.5411). The grader bills the covariance-push saving at -5.4% (the instrument said -8.5%). Below the 8-10% fire bar -> HOLD; combine with the CP3 8192 co-adapted chain (T-140) for a >=15% fire.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_gnn_hcap256_f32cov.py](code/cand_gnn_hcap256_f32cov.py)

## Evidence

- [LANE_g3_tracker.md#L953](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L953)
- [LANE_g3_tracker.md#L954](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L954)
- [LANE_g3_tracker.md#L955](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L955)
- [LANE_g3_tracker.md#L956](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L956)
- [LANE_g3_tracker.md#L959](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L959)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-float32-covariance-transport-saved-5-35-4-adjusted-14) · [Copy provenance](../../../PROVENANCE.json)

