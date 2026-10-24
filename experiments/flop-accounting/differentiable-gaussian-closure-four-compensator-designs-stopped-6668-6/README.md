# Differentiable Gaussian closure: four compensator designs stopped at 66–68.6% held-out removal

2026-09-04 · experiment · rejected · measured

M2 CLOSED: four compensator designs trained THROUGH the differentiable Gaussian card (per-unit small/large, W-power messages, recurrent 64-ch state, rank-32 covariance) all plateau at 66-68.6% held-out removal (1.19-1.28e-6 vs plain 3.78e-6). Per-unit corrections of a moment card cap at ~2/3 regardless of training regime (teacher-forced or through the loop). Stopped.

Recorded measurement:

    - T-124 DESIGN (the one route named and never built: the DIFFERENTIABLE CARD, Matt's preauth memo + the Keenan mechanism): a torch port of the GAUSSIAN card (mean + full covariance recursion, the card's own 37-term Hermite/Mehler pair map, W C W^T pushes) on MPS, and a rank-k covariance-plus-mean compensator per layer TRAINED THROUGH THE LOOP against the 1e9 truth post-means (150 train / 50 valid nets). Why it is not the closed family: the closed closure is per-unit and teacher-forced (62-65% ceiling; state mismatch kills co-adaptation in the loop); the loop-trained compensator optimises the final error directly and corrects the COVARIANCE (structured supplies are the only ones the laws allow). Milestones: M1 the torch card reproduces the fnp Gaussian card's open-loop prediction on r35 (2.8e-6 class) to ~1e-3 relative; M2 the compensator's held-out final error vs the bar 4.5e-8 (1e9 frame) at util ~0.17-0.19; the teacher-forced closure's loop is 1.2e-6. GPU only; no card workers.
    
    - T-124 M1 PASSED: diff_card.py (torch, float64 CPU, 0.9 s per net) reproduces the fnp Gaussian card's open-loop per-layer post means to 4e-7 relative and variances to 3e-6 (r35; final prediction rel 2.3e-4; final mse vs truth 2.897e-6 vs the instrument's 2.880e-6). Conventions: weights (in,out) row-vector (pre = post @ W), x ~ N(0,I), Mehler 37 terms off-diagonal + exact raw2 diagonal. The card is now differentiable end to end.
    
    - T-124 M2 first reads (through-the-loop, held-out 50 nets, plain Gaussian card 3.78e-6): DC1 (per-unit heads, 64 hidden, one W hop) 67.4% after ONE epoch, 68.5% at epoch 5 (1.19e-6) then flat; DC2 (256 hidden, 16 msg x W/W^2/W^3, lr 2e-3) 67.9% at epoch 1. The same ~65-68% as every teacher-forced closure: the ceiling is a property of PER-UNIT corrections, not of the training regime. Reading: the Gaussian card's mean error is the missing third-order content; the diagonal part (T_aaa, via the W^3 message) is per-unit-expressible, the OFF-DIAGONAL part (the K21 plane / the CP factor bank, 45% of the full card's cost) is not - a per-unit head with linear messages cannot form sum_ab W_ai^2 W_bi T_aab. DC3: a RECURRENT per-unit state (64 channels) transported through W, W^2, W^3 with energies = a LEARNED factor bank (a CP factor p transports as W^T p), heads from the state, trained through the loop. Deploy cost ~0.3% util.
    
    - T-124 M2 CLOSED: four compensator designs trained THROUGH the differentiable Gaussian card (per-unit small/large, W-power messages, recurrent 64-ch state, rank-32 covariance) all plateau at 66-68.6% held-out removal (1.19-1.28e-6 vs plain 3.78e-6). Per-unit corrections of a moment card cap at ~2/3 regardless of training regime (teacher-forced or through the loop). Stopped.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/diff_card.py](code/diff_card.py)

## Evidence

- [LANE_g3_tracker.md#L919](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L919)
- [LANE_g3_tracker.md#L920](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L920)
- [LANE_g3_tracker.md#L921](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L921)
- [LANE_g3_tracker.md#L922](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L922)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-differentiable-gaussian-closure-four-compensator-designs-stopped-6668-6) · [Copy provenance](../../../PROVENANCE.json)

