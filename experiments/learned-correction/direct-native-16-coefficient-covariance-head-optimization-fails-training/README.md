# Direct native 16-coefficient covariance-head optimization fails training: adjusted ratio 1.00078780578

2026-09-20 · experiment · rejected · measured

A fixed 36-update simultaneous-perturbation stochastic approximation run used full-estimator paired losses and averaged its final 18 centers.

Recorded measurement:

    Parent331684 is the official3.5844144212297072e-9 bank,100/100 completed.
    One36-update projected SPSA run changed only16 FP32 output coefficients.
    the fixed average of the final18 centers; no checkpoint or gain selection.
    Independent scalar replay verifies all72 losses, every update and the
    terminal FP32 head. The earlier linear-response failure remains closed.
    - train: adjusted ratio1.00078780578,wins4/6,worst1.03793699,passed=False.
    Native calls82; client calls0; all85 groups retired,zero added swap.
    Peak sampled group RSS6341033984B. No new samples,datasets or GPU.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/covariance_head_spsa_native_v1/algorithm.py](code/covariance_head_spsa_native_v1/algorithm.py)
- [code/covariance_head_spsa_native_v1/audit_fit.py](code/covariance_head_spsa_native_v1/audit_fit.py)
- [code/covariance_head_spsa_native_v1/candidate.py](code/covariance_head_spsa_native_v1/candidate.py)
- [code/covariance_head_spsa_native_v1/common.py](code/covariance_head_spsa_native_v1/common.py)
- [code/covariance_head_spsa_native_v1/finalize.py](code/covariance_head_spsa_native_v1/finalize.py)
- [code/covariance_head_spsa_native_v1/package.py](code/covariance_head_spsa_native_v1/package.py)
- [code/covariance_head_spsa_native_v1/prepare.py](code/covariance_head_spsa_native_v1/prepare.py)
- [code/covariance_head_spsa_native_v1/prepare_release.py](code/covariance_head_spsa_native_v1/prepare_release.py)
- [code/covariance_head_spsa_native_v1/qualify_release.py](code/covariance_head_spsa_native_v1/qualify_release.py)
- [code/covariance_head_spsa_native_v1/run_release.py](code/covariance_head_spsa_native_v1/run_release.py)
- [code/covariance_head_spsa_native_v1/run_train.py](code/covariance_head_spsa_native_v1/run_train.py)
- [code/covariance_head_spsa_native_v1/unit.py](code/covariance_head_spsa_native_v1/unit.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/covariance_head_spsa_native_v1/variants/unit_parent.py](code/covariance_head_spsa_native_v1/variants/unit_parent.py)
- [code/covariance_head_spsa_native_v1/variants/unit_probe.py](code/covariance_head_spsa_native_v1/variants/unit_probe.py)
- [code/full_lower_covariance_edge_native_v1/candidate.py](code/full_lower_covariance_edge_native_v1/candidate.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/readout_residual_runtime_v1/plain_client.py](code/readout_residual_runtime_v1/plain_client.py)
- [code/readout_residual_runtime_v1/plain_driver.py](code/readout_residual_runtime_v1/plain_driver.py)
- [code/readout_residual_runtime_v1/predict_only_probe_v1/client.py](code/readout_residual_runtime_v1/predict_only_probe_v1/client.py)
- [code/readout_residual_runtime_v1/predict_only_probe_v1/driver.py](code/readout_residual_runtime_v1/predict_only_probe_v1/driver.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L7](RESULT.md#L7)
- [RESULT.md#L8](RESULT.md#L8)
- [REEXAMINATION_INDEX_20260919.md#L844](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L844)
- [REEXAMINATION_INDEX_20260919.md#L848](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L848)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-direct-native-16-coefficient-covariance-head-optimization-fails-training) · [Copy provenance](../../../PROVENANCE.json)

