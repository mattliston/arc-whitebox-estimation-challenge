# Rank-256 fourth-cumulant closure passed the 14-network independent-truth submission gate

2026-09-04 · experiment · informational · measured

GRADED #329754: adjusted 1.6116e-8 (-9.6% vs the bank 1.7836e-8), raw 2.965e-8, util 0.5435. Prediction 1.59e-8 was within noise. New board best; promoted.

Recorded measurement:

    - T-135 CAP256_v2 GRADER PANEL (14 nets): panel8 raw 3.276e-8 util 0.547 adjusted 1.792e-8; virgin6 raw 2.661e-8 util 0.541 adjusted 1.436e-8 vs the deployed card's virgin6 1.645e-8 (raw 2.712e-8): -12.7% adjusted on the virgin nets (raw -2%, util -11%). Running the deployed D11 card's panel8 for the same-frame reference before deciding to fire.
    
    - T-135b SAME-FRAME REFERENCE + FIRE: deployed D11 panel8 raw 3.272e-8 util 0.609 adjusted 1.992e-8; virgin6 raw 2.712e-8 util 0.608 adjusted 1.645e-8. CAP256_v2: panel8 -10.0%, virgin6 -12.7%, raw equal, util -10.5% (the closure adds ~nothing over the deployed operator; the gain is the CP4 cap). FIRED as #329754 (ledger). Expected board ~1.59e-8.
    
    - T-135c GRADED #329754: adjusted 1.6116e-8 (-9.6% vs the bank 1.7836e-8), raw 2.965e-8, util 0.5435. Prediction 1.59e-8 was within noise. New board best; promoted. Rank ~22 of the phase-2 board (rank 10 = 9.4e-9). LEAN256 co-adapted: closure 86.6% teacher-forced (healthy rows), loop r35 F2J-off 3.31e-8 at util 0.463 (instrument frame; 1e9 ranking pending).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_d11_util_halving_v1/estimator_cap.py](code/public_d11_util_halving_v1/estimator_cap.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/build_gnn_candidate.py](code/public_g22_operator_features_v1/build_gnn_candidate.py)
- [code/public_g22_operator_features_v1/cand_gnn_hcap256_v2.py](code/public_g22_operator_features_v1/cand_gnn_hcap256_v2.py)
- [code/public_g22_operator_features_v1/score_candidate.py](code/public_g22_operator_features_v1/score_candidate.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L945](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L945)
- [LANE_g3_tracker.md#L946](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L946)
- [LANE_g3_tracker.md#L950](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L950)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-rank-256-fourth-cumulant-closure-passed-14-network-independent) · [Copy provenance](../../../PROVENANCE.json)

