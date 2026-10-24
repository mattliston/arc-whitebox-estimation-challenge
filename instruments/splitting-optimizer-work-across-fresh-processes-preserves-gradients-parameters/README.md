# Splitting optimizer work across fresh processes preserves gradients, parameters and random-state bytes exactly

2026-09-21 · tool · informational · measured

A reference run and two resumed workers matched every optimizer boundary, enabling memory-bounded training without changing the optimization trajectory.

Recorded measurement:

    The study executed 16 backward graphs and four optimizer steps in total:
    Maximum observed group RSS was 6,009,126,912 bytes. Maximum post-graph
    retirement RSS was 1,406,697,472 bytes. Added system swap was zero. The
    unchanged 7.5 GB peak and 2 GB retirement limits passed. Total controller
    wall time was 507.031 seconds; session 24419 exited with status zero.
    start from the original zero branch, preserve all 40 fixed updates and
    verify the entire 134-graph/33-update completed old prefix.

## Programs

- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/iso_common.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/iso_common.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/prepare.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/prepare.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/qualify.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/qualify.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/run.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/state_io.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/worker.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/worker.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/train.py](code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/train.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/allocator.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/allocator.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/common.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/common.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/guard.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/guard.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_branch.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_branch.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L20](RESULT.md#L20)
- [RESULT.md#L21](RESULT.md#L21)
- [RESULT.md#L22](RESULT.md#L22)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [result.json](result.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-09-21-splitting-optimizer-work-across-fresh-processes-preserves-gradients-parameters) · [Copy provenance](../../PROVENANCE.json)

