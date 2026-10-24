# Existing 528 covariance-edge parameters pass forward and pullback implementation checks

2026-09-21 · tool · informational · measured

Only the five existing covariance-edge parameter leaves became trainable; optimizer state and repeated gradient calls passed exact transport checks.

Recorded measurement:

    The exact official331760 source initializes all parameters. Only the existing
    five covariance-edge leaves (528 FP32 values) are eligible for learning.
    inverse recovers the prior model_v2 exactly; only trainability and checks change.
    The7-node independent forward check has relative error
    1.15220872002e-07. The largest independent analytic pullback
    error is1.85174227315e-07, below the
    fixed2e-5 gate. Four full calls (frozen35, gradient35/100/35) preserve frozen /
    checkpointed outputs and repeated35 prediction, gradient, variance, selector
    The real528-leaf Adam boundary, settings and RNG round trip exactly. A second
    Both guards retired, peak5.371462GB, zero added swap,
    and post-graph retirement below2GB. No reference mean array was opened, no
    This is related to the older failed16-output native SPSA and joint covariance
    native validation are still required. Official331760 remains3.507738856975084e-9.

## Programs

- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/allocator.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/closeout.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/closeout.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/common.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/full.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/full.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/model_edge.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/model_edge.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/prepare.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/prepare.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/run.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/state_io.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/unit.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/unit.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)

<details>
<summary>Additional programs and dependencies (14)</summary>

- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/iso_common.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/iso_common.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/state_io.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/guard.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/allocator.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/allocator.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/common.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/common.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/guard.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/guard.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_branch.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_branch.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L8](RESULT.md#L8)
- [REEXAMINATION_INDEX_20260919.md#L355](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L355)
- [REEXAMINATION_INDEX_20260919.md#L358](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L358)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-21-existing-528-covariance-edge-parameters-pass-forward-pullback-implementation) · [Copy provenance](../../PROVENANCE.json)

