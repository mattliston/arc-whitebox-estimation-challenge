# Recurrent covariance-error observer: history worsens check-network raw error relative to reset

2026-09-27 · experiment · rejected · measured

Matched6209-parameter observers were fitted with and without covariance memory over seven layers. Transporting the predicted covariance error through actual weights and Gaussian gates did not improve the fixed check set.

Recorded measurement:

    TRAIN6/CHECK2;30epochs/180 whole-network updates per arm;HISTORY raw ratio1.015261425333;HISTORY/RESET1.007003992870.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/recurrent_covariance_observer_v1/collect.py](code/recurrent_covariance_observer_v1/collect.py)
- [code/recurrent_covariance_observer_v1/common.py](code/recurrent_covariance_observer_v1/common.py)
- [code/recurrent_covariance_observer_v1/data.py](code/recurrent_covariance_observer_v1/data.py)
- [code/recurrent_covariance_observer_v1/fit.py](code/recurrent_covariance_observer_v1/fit.py)
- [code/recurrent_covariance_observer_v1/model.py](code/recurrent_covariance_observer_v1/model.py)
- [code/recurrent_covariance_observer_v1/numpy_replay.py](code/recurrent_covariance_observer_v1/numpy_replay.py)
- [code/recurrent_covariance_observer_v1/projection.py](code/recurrent_covariance_observer_v1/projection.py)
- [code/recurrent_covariance_observer_v1/unit.py](code/recurrent_covariance_observer_v1/unit.py)
- [code/recurrent_covariance_observer_v1/verify_fit.py](code/recurrent_covariance_observer_v1/verify_fit.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)

## Evidence

- [RESULTS.md#L1618](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1618)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-recurrent-covariance-error-observer-history-worsens-check-network-raw) · [Copy provenance](../../../PROVENANCE.json)

