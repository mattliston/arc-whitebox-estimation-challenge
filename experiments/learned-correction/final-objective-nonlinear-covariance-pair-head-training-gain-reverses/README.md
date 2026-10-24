# Final-objective nonlinear covariance pair head: training gain reverses on eight check networks

2026-09-27 · experiment · rejected · measured

A6177-parameter late covariance action was trained directly on current-native final means. Corrected native derivative qualification passed, but the fixed fitted head failed development generalization.

Recorded measurement:

    40epochs/480updates;TRAIN12 raw-0.757253%;CHECK8 raw+0.275953%,2/8 wins;corrected derivative worst discrepancy2.2270%;22 passive captures retain all16 bytes.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_pair_objective_v1/assess_native.py](code/terminal_pair_objective_v1/assess_native.py)
- [code/terminal_pair_objective_v1/build.py](code/terminal_pair_objective_v1/build.py)
- [code/terminal_pair_objective_v1/capture.py](code/terminal_pair_objective_v1/capture.py)
- [code/terminal_pair_objective_v1/capture_exact.py](code/terminal_pair_objective_v1/capture_exact.py)
- [code/terminal_pair_objective_v1/capture_exact_all.py](code/terminal_pair_objective_v1/capture_exact_all.py)
- [code/terminal_pair_objective_v1/common.py](code/terminal_pair_objective_v1/common.py)
- [code/terminal_pair_objective_v1/consumer.py](code/terminal_pair_objective_v1/consumer.py)
- [code/terminal_pair_objective_v1/consumer_v2.py](code/terminal_pair_objective_v1/consumer_v2.py)
- [code/terminal_pair_objective_v1/consumer_v3.py](code/terminal_pair_objective_v1/consumer_v3.py)
- [code/terminal_pair_objective_v1/consumer_v4.py](code/terminal_pair_objective_v1/consumer_v4.py)
- [code/terminal_pair_objective_v1/fit.py](code/terminal_pair_objective_v1/fit.py)
- [code/terminal_pair_objective_v1/fragment.py](code/terminal_pair_objective_v1/fragment.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/terminal_pair_objective_v1/model.py](code/terminal_pair_objective_v1/model.py)
- [code/terminal_pair_objective_v1/qualify_native.py](code/terminal_pair_objective_v1/qualify_native.py)
- [code/terminal_pair_objective_v1/qualify_native_v4.py](code/terminal_pair_objective_v1/qualify_native_v4.py)
- [code/terminal_pair_objective_v1/qualify_torch.py](code/terminal_pair_objective_v1/qualify_torch.py)
- [code/terminal_pair_objective_v1/qualify_torch_v2.py](code/terminal_pair_objective_v1/qualify_torch_v2.py)
- [code/terminal_pair_objective_v1/qualify_torch_v3.py](code/terminal_pair_objective_v1/qualify_torch_v3.py)
- [code/terminal_pair_objective_v1/qualify_torch_v4.py](code/terminal_pair_objective_v1/qualify_torch_v4.py)
- [code/terminal_pair_objective_v1/unit_native.py](code/terminal_pair_objective_v1/unit_native.py)
- [code/terminal_pair_objective_v1/verify_fit.py](code/terminal_pair_objective_v1/verify_fit.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)

</details>

## Evidence

- [RESULTS.md#L217](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L217)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-final-objective-nonlinear-covariance-pair-head-training-gain-reverses) · [Copy provenance](../../../PROVENANCE.json)

