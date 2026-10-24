# Pure Gaussian closure remained inaccurate after co-adapted learning

2026-09-04 · experiment · rejected · measured

PURE GAUSSIAN CARD, co-adapted: oracle one-step host 3.11e-4 (the plain host was 2.8e-6: the oracle-recursion one-step error is far larger because the Gaussian recursion's covariance drifts without the third-order terms), closure 90.7% teacher-forced, loop r35 1.234e-6 @ 0.163 / r100 1.431e-6 @ 0.181 (1e9): stable but 50x the full card -> raw x util ~2.2e-7, 15x worse than the bank. DEAD. The cheap-card family is closed at every member.

Recorded measurement:

    - T-121 PURE GAUSSIAN CARD (no third- or fourth-order machinery at all: RETRO_K21_ZERO + no cubic + CP4 off + no quartic; CP3 cap 1024 because cap 1 breaks the arena kernel): util 0.178, plain host 2.83e-6 (33x the full card; the diagonal card was 2.54e-6). Bar at 0.178 needs raw < ~5.6e-8 grader = ~98% removal in the loop. The last member of the family; its recursion is self-consistent (unlike the diagonal card) so the loop may at least be stable. Co-adaptation round launched (chain_gausscard.sh); prior ~5%.
    
    - T-121 PURE GAUSSIAN CARD, co-adapted: oracle one-step host 3.11e-4 (the plain host was 2.8e-6: the oracle-recursion one-step error is far larger because the Gaussian recursion's covariance drifts without the third-order terms), closure 90.7% teacher-forced, loop r35 1.234e-6 @ 0.163 / r100 1.431e-6 @ 0.181 (1e9): stable but 50x the full card -> raw x util ~2.2e-7, 15x worse than the bank. DEAD. The cheap-card family is closed at every member. Machine idle; nothing running.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/chain_gausscard.sh](code/public_g22_operator_features_v1/chain_gausscard.sh)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py)
- [code/public_g22_operator_features_v1/score_vs_truthpost.py](code/public_g22_operator_features_v1/score_vs_truthpost.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L908](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L908)
- [LANE_g3_tracker.md#L909](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L909)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-pure-gaussian-closure-remained-inaccurate-after-co-adapted-learning) · [Copy provenance](../../../PROVENANCE.json)

