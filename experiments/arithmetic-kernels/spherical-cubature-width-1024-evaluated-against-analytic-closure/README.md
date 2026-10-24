# Spherical cubature at width 1024 was evaluated against the analytic closure

2026-09-03 · experiment · informational · measured

CUBATURE PROTOTYPE AT 1024x16 (public_g30_cubature_v1/cubature_proto.py; r35, 32M truth, noise-corrected; per-point variance V = mean squared error (MSE) x k; adjusted = V c / B with c = 15 dense layers, 0.59x for Winograd + pruning): plain Gaussian Monte Carlo (MC) k=8192: final 9.91e-6, V 0.081 (matches the board's MC line 1.13e-6 at the full budget); antipodal orthonormal bases + radial factor, no snaps: V 0.049; + exact layer-1 mean and layer-2 arc-cosine variance snaps: V 0.045 (b=4) / 0.037 (b=8, k=16384: final 2.24e-6). Best case adjusted ~3.1e-7 - 20x WORSE than our bank and 80x above the competitor's 4.04e-9. The exact layer-1 formula matches the truth to the noise floor (col norms / sqrt(2 pi): 2.28e-8 vs se^2 2.13e-8), so the orientation and the frame are right.

Recorded measurement:

    - T-93 THE COMPETITOR'S CLASS IS SPHERICAL CUBATURE (Matt 16:30; whest_phase1_writeup_skibidi_toilet.pdf, submission #325170, phase 1): bias-free ReLU nets are positively homogeneous, so E_x[M(x)] = E[chi_n] x E_{u in S^{n-1}}[M(u)] exactly; the sphere integral is evaluated on a tight spherical 4-design (all 129 real Kerdock MUBs of R^256, antipodal closure, 66,048 points; layer 1 by FWHT, the antipodal half of layer 2 free), exact layer-1 mean and layer-2 variance snaps (arc-cosine kernel; a 0.195% coherent snap error raised final MSE 4.2x), pilot-driven neuron pruning (cut ~80% of columns, dropped neurons compensated), Strassen-Winograd L4/L5 (0.74x), an exact sparse cold path; score = V c / B is FLAT in the point count; phase-1 per-point variance V = 0.0185 (MC 0.042). Phase 2 (#329661): raw 1.854e-8 at util 0.218 -> 4.04e-9. At 1024 x 16: 513 real MUBs exist (n = 4^5) but the full 5-design (1.05M points) is ~9x the budget, so a sub-design (not tight) or unions of antipodal orthonormal bases (3-designs) are the affordable forms. Prototype: public_g30_cubature_v1/cubature_proto.py (ortho bases, radial factor, both snaps; grades V per point and adj = V c / B).
    
    - T-93b CUBATURE PROTOTYPE AT 1024x16 (public_g30_cubature_v1/cubature_proto.py; r35, 32M truth, noise-corrected; per-point variance V = MSE x k; adjusted = V c / B with c = 15 dense layers, 0.59x for Winograd + pruning): plain Gaussian MC k=8192: final 9.91e-6, V 0.081 (matches the board's MC line 1.13e-6 at the full budget); antipodal orthonormal bases + radial factor, no snaps: V 0.049; + exact layer-1 mean and layer-2 arc-cosine variance snaps: V 0.045 (b=4) / 0.037 (b=8, k=16384: final 2.24e-6). Best case adjusted ~3.1e-7 - 20x WORSE than our bank and 80x above the competitor's 4.04e-9. The exact layer-1 formula matches the truth to the noise floor (col norms / sqrt(2 pi): 2.28e-8 vs se^2 2.13e-8), so the orientation and the frame are right. At this shape the degree >= 4 harmonics hold ~half the variance; the paper's own phase-1 gain was 2.3x over MC and its phase-2 score is 280x over MC, so the phase-2 entry is a different (or heavily augmented) method. The closure + operator class (ours: 38x over MC raw, 64x adjusted) remains the strongest measured class on this machine.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g30_cubature_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cubature_proto.py](code/cubature_proto.py)

## Evidence

- [LANE_g3_tracker.md#L765](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L765)
- [LANE_g3_tracker.md#L766](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L766)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-03-spherical-cubature-width-1024-evaluated-against-analytic-closure) · [Copy provenance](../../../PROVENANCE.json)

