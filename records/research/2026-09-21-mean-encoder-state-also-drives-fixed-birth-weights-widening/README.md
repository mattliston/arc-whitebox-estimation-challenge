# Mean-encoder state also drives fixed birth weights; widening state requires changing the 64-channel interface

2026-09-21 · finding · informational · measured

The native birth function consumes the current 64-channel encoder state through a fixed 64-by-4 matrix, while message width is a separate payload-derived dimension.

Recorded measurement:

    birth-node function consumes the returned GNN state, standardizes its64
    channels, multiplies by A[64,4], and applies tanh plus the fixed per-layer
    cstar coefficients. The current training translation updates only the11 P
    The64-channel birth-node shape is explicitly checked in native code. Simply
    A[64,4] interface. Message width is a different dimension: the previous
    while the state stays64. Neither observation demonstrates compatibility,
    Joint P/A learning was actually tested on the older COMP2e13d40a host. Two
    matched arms each used20 TRAIN networks,three epochs,60 backwards and15
    the11 P leaves and256 A entries; the control froze A while retaining its H
    dependence. Native CHECK37/39/42/48 compared each role's own complete bills.
    Joint/COMP adjusted ratio was1.0009260627183014 with2/4wins; joint/P-only
    was1.0044040046045324 with1/4win. The original joint-learning gates failed.
    The matched P-only arm's0.9965373078260472 ratio and3/4wins qualified only a
    small reused-development component, not native40, independent14, production
    Do not conflate this completed joint P/A test with current_bd_p_coadaptation_v1:

## Programs

- [audit.py](audit.py)
- [current_birth_node_function.py](current_birth_node_function.py)
- [../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [../../../instruments/existing-528-covariance-edge-parameters-pass-forward-pullback-implementation/code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py](../../../instruments/existing-528-covariance-edge-parameters-pass-forward-pullback-implementation/code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py)
- [../../../instruments/existing-528-covariance-edge-parameters-pass-forward-pullback-implementation/code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py](../../../instruments/existing-528-covariance-edge-parameters-pass-forward-pullback-implementation/code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L8](RESULT.md#L8)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L15](RESULT.md#L15)

<details>
<summary>All result files and records (1)</summary>

- [RESULT.md](RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-mean-encoder-state-also-drives-fixed-birth-weights-widening) · [Copy provenance](../../../PROVENANCE.json)

