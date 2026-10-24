# Current-state covariance-supervised pair model: both covariance and next-response gates fail

2026-09-27 · experiment · rejected · measured

A fresh6177-parameter nonlinear pair model was supervised on current-estimator covariance error with an exact zero-diagonal, mean-orthogonal projection. The fixed six-network training and two-network check study failed both requirements.

Recorded measurement:

    60 epochs/360 updates;CHECK covariance ratios0.982267/0.961188;following-response ratios1.048420/0.950966.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_covariance_pair_supervision_v1/collect.py](code/current_covariance_pair_supervision_v1/collect.py)
- [code/current_covariance_pair_supervision_v1/common.py](code/current_covariance_pair_supervision_v1/common.py)
- [code/current_covariance_pair_supervision_v1/data.py](code/current_covariance_pair_supervision_v1/data.py)
- [code/current_covariance_pair_supervision_v1/fit.py](code/current_covariance_pair_supervision_v1/fit.py)
- [code/current_covariance_pair_supervision_v1/pair_base.py](code/current_covariance_pair_supervision_v1/pair_base.py)
- [code/current_covariance_pair_supervision_v1/projection.py](code/current_covariance_pair_supervision_v1/projection.py)
- [code/current_covariance_pair_supervision_v1/unit.py](code/current_covariance_pair_supervision_v1/unit.py)
- [code/current_covariance_pair_supervision_v1/verify_fit.py](code/current_covariance_pair_supervision_v1/verify_fit.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/terminal_pair_objective_v1/consumer_v4.py](code/terminal_pair_objective_v1/consumer_v4.py)

## Evidence

- [RESULTS.md#L169](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L169)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-current-state-covariance-supervised-pair-model-both-covariance-next) · [Copy provenance](../../../PROVENANCE.json)

