# Frozen covariance observer with physical feedback: native adjusted error rises 13.6731%

2026-09-27 · experiment · rejected · measured

The frozen reset observer was inserted into the actual physical covariance recurrence after helper and disabled-control checks. Its enabled endpoint failed on both native training networks.

Recorded measurement:

    TRAIN35/100 raw+11.1105%,bill+2.30579%,adjusted+13.6731%,0 wins;first9 means exact.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/covariance_supplier_feedback_v1/assess.py](code/covariance_supplier_feedback_v1/assess.py)
- [code/covariance_supplier_feedback_v1/build.py](code/covariance_supplier_feedback_v1/build.py)
- [code/covariance_supplier_feedback_v1/candidate.py](code/covariance_supplier_feedback_v1/candidate.py)
- [code/covariance_supplier_feedback_v1/disabled.py](code/covariance_supplier_feedback_v1/disabled.py)
- [code/covariance_supplier_feedback_v1/evaluate.py](code/covariance_supplier_feedback_v1/evaluate.py)
- [code/covariance_supplier_feedback_v1/fragment.py](code/covariance_supplier_feedback_v1/fragment.py)
- [code/covariance_supplier_feedback_v1/unit.py](code/covariance_supplier_feedback_v1/unit.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [RESULTS.md#L137](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L137)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-frozen-covariance-observer-physical-feedback-native-adjusted-error-rises) · [Copy provenance](../../../PROVENANCE.json)

