# Exact Gaussian-gate reuse removes 60 requests but misses residual gate: ratio 0.993664585825508

2026-09-20 · experiment · rejected · measured

Sharing clipped variance, standard deviation, standardized mean and Gaussian gate preserved all output bytes and complete accounting.

Recorded measurement:

    Parent #331684 is adjusted 3.5844144212297072e-9 with 100/100 completed.
    Each complete prediction removes 1566720 counted operations.
    - Five-pair prediction-only residual ratio: 0.993664585825508; required <=0.98.
    - Residual wins: 3/5; required >=4.
    - Maximum candidate residual: 0.175165819s; required <=0.25s.
    - Worst residual ratio: 1.01201407112272; required <=1.10.
    - Summed total-wall ratio: 0.998118999182032; required <=1.05.
    - Worst total-wall ratio: 1.00156647806272; required <=1.10.
    All 18 guarded groups are retired; zero added swap; peak sampled RSS 6605897728 bytes.
    The approximately 1e-9 goal remains unachieved; official grading governs any new bank.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/full_lower_gaussian_gate_reuse_v1/build.py](code/full_lower_gaussian_gate_reuse_v1/build.py)
- [code/full_lower_gaussian_gate_reuse_v1/candidate.py](code/full_lower_gaussian_gate_reuse_v1/candidate.py)
- [code/full_lower_gaussian_gate_reuse_v1/common.py](code/full_lower_gaussian_gate_reuse_v1/common.py)
- [code/full_lower_gaussian_gate_reuse_v1/disabled.py](code/full_lower_gaussian_gate_reuse_v1/disabled.py)
- [code/full_lower_gaussian_gate_reuse_v1/finalize.py](code/full_lower_gaussian_gate_reuse_v1/finalize.py)
- [code/full_lower_gaussian_gate_reuse_v1/native_qualification.py](code/full_lower_gaussian_gate_reuse_v1/native_qualification.py)
- [code/full_lower_gaussian_gate_reuse_v1/poll_submission.py](code/full_lower_gaussian_gate_reuse_v1/poll_submission.py)
- [code/full_lower_gaussian_gate_reuse_v1/prepare_package.py](code/full_lower_gaussian_gate_reuse_v1/prepare_package.py)
- [code/full_lower_gaussian_gate_reuse_v1/qualify_boundary.py](code/full_lower_gaussian_gate_reuse_v1/qualify_boundary.py)
- [code/full_lower_gaussian_gate_reuse_v1/release.py](code/full_lower_gaussian_gate_reuse_v1/release.py)
- [code/full_lower_gaussian_gate_reuse_v1/run_boundary.py](code/full_lower_gaussian_gate_reuse_v1/run_boundary.py)
- [code/full_lower_gaussian_gate_reuse_v1/run_native.py](code/full_lower_gaussian_gate_reuse_v1/run_native.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/full_lower_gaussian_gate_reuse_v1/submit.py](code/full_lower_gaussian_gate_reuse_v1/submit.py)
- [code/full_lower_gaussian_gate_reuse_v1/unit.py](code/full_lower_gaussian_gate_reuse_v1/unit.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/readout_residual_runtime_v1/predict_only_probe_v1/client.py](code/readout_residual_runtime_v1/predict_only_probe_v1/client.py)
- [code/readout_residual_runtime_v1/predict_only_probe_v1/driver.py](code/readout_residual_runtime_v1/predict_only_probe_v1/driver.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [REEXAMINATION_INDEX_20260919.md#L807](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L807)
- [REEXAMINATION_INDEX_20260919.md#L810](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L810)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-exact-gaussian-gate-reuse-removes-60-requests-but-misses) · [Copy provenance](../../../PROVENANCE.json)

