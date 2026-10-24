# Removing the dense quartic transport after co-adaptation traded compute for comparable adjusted error

2026-09-04 · experiment · informational · measured

LEAN256 RESULT (fourth-cumulant canonical-polyadic carrier (CP4) 256 + no dense quartic push, closure co-adapted on healthy rows, 86.6% teacher-forced): loop 1e9 frame r35 3.127e-8 / r100 3.174e-8 at util 0.463 -> adjusted 1.45 / 1.47e-8 vs CAP256 1.34 / 1.50e-8: a WASH (the quartic push's -16% util costs +29%/+18% raw). Confirms the T-118 law (the fourth-order machinery is replaceable by the closure only at a wash). Not a bank.

Code availability: The lean-256 candidate, row generator, trainer and builder were recovered. A dedicated launch script tying every training and evaluation setting to this tracker result was not identified.

Recorded measurement:

    - T-136 LEAN256 RESULT (CP4 256 + no dense quartic push, closure co-adapted on healthy rows, 86.6% teacher-forced): loop 1e9 frame r35 3.127e-8 / r100 3.174e-8 at util 0.463 -> adjusted 1.45 / 1.47e-8 vs CAP256 1.34 / 1.50e-8: a WASH (the quartic push's -16% util costs +29%/+18% raw). Confirms the T-118 law (the fourth-order machinery is replaceable by the closure only at a wash). Not a bank. The cheap-card family's best is CAP256 (fired).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_d11_util_halving_v1/estimator_cap.py](code/public_d11_util_halving_v1/estimator_cap.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/build_gnn_candidate.py](code/public_g22_operator_features_v1/build_gnn_candidate.py)
- [code/public_g22_operator_features_v1/cand_gnn_hlean256.py](code/public_g22_operator_features_v1/cand_gnn_hlean256.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L951](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L951)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-removing-dense-quartic-transport-after-co-adaptation-traded-compute) · [Copy provenance](../../../PROVENANCE.json)

