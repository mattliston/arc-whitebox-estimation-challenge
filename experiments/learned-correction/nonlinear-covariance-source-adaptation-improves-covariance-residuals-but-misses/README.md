# Nonlinear covariance-source adaptation improves covariance residuals but misses next-mean gates

2026-09-21 · experiment · rejected · measured

Matched fits trained either 16 output coefficients or all 528 existing parameters on frozen current trajectories with coherent covariance-innovation labels.

Recorded measurement:

    Both fits used the exact current official331684 frozen states, the same
    coherent8M Gaussian-gain covariance-innovation labels, six TRAIN networks
    and seven layers per network. VALID400/500 were excluded from both fits;
    epochs and1680 updates completed for each arm, using identical cell orders,
    The matched control trained16 output weights. The nonlinear arm trained
    all528 parameters in the existing16-channel edge source. Input features,
    |400|output_only|0.995642554374|0.986418747507|
    |400|nonlinear|0.979719722266|0.981543255247|
    |500|output_only|0.984627969438|0.999227123010|
    |500|nonlinear|0.970613741842|0.991767418498|
    Each nonlinear held network requires covariance<=.98, response<=.95, and
    response<=.98 of the matched output-only control.
    - Rank400: response/control=0.995057380780; passed=False.
    - Rank500: response/control=0.992534525594; passed=False.
    or derivatives, and retain coherent8M sampling/numerical error. The reverse
    native gradient or an autonomous improvement. No1B final-mean reference,
    All528 derivative coordinates pass independent scalar complex-step checks
    permutation, null, directional-descent and FP32 checks. All56 wide bank
    source reconstructions pass3e-6 before each network's targets are decoded.
    reproduce within2e-10. FP32 target/response roundoff passes3e-6.
    Both final models were sealed before VALID prediction. All28 held layer
    FP64 reference. Direct and expanded quadratic losses agree within2e-10.
    The first closeout verifier stopped on exact pooled-ratio equality: Python3.13
    uses a different floating-point sum from the Python3.11 scoring runtime.
    The preserved diagnostic shows differences of0--4 rounding units. Additive

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/common.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/common.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/data_prep.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/data_prep.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/finalize.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/finalize.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/finalize_v2.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/finalize_v2.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/fit.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/fit.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/model.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/model.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/predict.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/predict.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/preflight.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/preflight.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/prepare.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/prepare.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/run.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/run.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/score.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/score.py)
- [code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/unit.py](code/codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/unit.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/astra_phase2_reexamination_v1/train_joint_edge_covariance_v1.py](code/astra_phase2_reexamination_v1/train_joint_edge_covariance_v1.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/fit.py](code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/fit.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L7](RESULT.md#L7)
- [REEXAMINATION_INDEX_20260919.md#L635](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L635)
- [REEXAMINATION_INDEX_20260919.md#L638](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L638)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-nonlinear-covariance-source-adaptation-improves-covariance-residuals-but-misses) · [Copy provenance](../../../PROVENANCE.json)

