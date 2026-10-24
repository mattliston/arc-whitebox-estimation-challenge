# Third-cumulant birth energy was concentrated, but unadapted rank cuts worsened the learned closure

2026-09-04 · finding · informational · measured

third-cumulant canonical-polyadic carrier (CP3) cap 12288 UN-co-adapted (cap-256 closure): r35 3.38e-8 at 0.497 vs 2.52e-8 at 0.507; r100 3.23e-8 at 0.514 vs 2.79e-8: raw +34%/+16% for util -2%. Even a mild third-order cap breaks the co-adapted closure; only the co-adapted chain can judge the cap.

Recorded measurement:

    - T-140: third-order BIRTH energy is concentrated: the top 512 of 1024 units carry 87% (L2), 96% (L5), 97.5% (L8), 98.6% (L14) of sum T_aaa^2 (|T| top/median 2 -> 8 with depth). The CP3 cap 8192 (g29: host +2% for util -13.5%) was never judged with an honest co-adapted closure (its rows exploded, T-123 audit). Launched chain_cp3_8k.sh: CP4 256 + CP3 8192 + F32_COV, PSD rows, closure retrain, loop, build (BUILD_CP3_CAP=8192). Expected board ~1.35e-8 if raw holds within +3%. Also testing CP3 12288 un-co-adapted on r35/r100.
    
    - T-140a CP3 cap 12288 UN-co-adapted (cap-256 closure): r35 3.38e-8 at 0.497 vs 2.52e-8 at 0.507; r100 3.23e-8 at 0.514 vs 2.79e-8: raw +34%/+16% for util -2%. Even a mild third-order cap breaks the co-adapted closure; only the co-adapted chain can judge the cap.

## Programs

- [chain_cp3_8k.sh](chain_cp3_8k.sh)
- [../../../experiments/arithmetic-kernels/third-cumulant-rank-cuts-sharply-worsened-raw-error-while/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py](../../../experiments/arithmetic-kernels/third-cumulant-rank-cuts-sharply-worsened-raw-error-while/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py)
- [../2026-09-04-candidate-parity-bug-exported-variance-head-active-although-instrument/build_gnn_candidate.py](../2026-09-04-candidate-parity-bug-exported-variance-head-active-although-instrument/build_gnn_candidate.py)
- [cand_vs_instrument.py](cand_vs_instrument.py)
- [../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/export_gnn_payload.py](../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/export_gnn_payload.py)
- [../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py](../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py)
- [../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/score_vs_truthpost.py](../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/score_vs_truthpost.py)
- [../../../instruments/oracle-recursion-row-generator-online-payload-exporter-were-mechanically/code/public_g22_operator_features_v1/train_gnn_oracle.py](../../../instruments/oracle-recursion-row-generator-online-payload-exporter-were-mechanically/code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L958](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L958)
- [LANE_g3_tracker.md#L960](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L960)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-third-cumulant-birth-energy-concentrated-but-unadapted-rank-cuts) · [Copy provenance](../../../PROVENANCE.json)

