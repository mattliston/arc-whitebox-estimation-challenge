# Process-split signed-message fit completes 160 graphs and exactly reproduces the interrupted 134-graph prefix

2026-09-21 · tool · informational · measured

Lossless optimizer-state transport across batch processes allowed the unchanged signed-message fit to complete. All previous completed graph and update receipts matched exactly, while native scoring remained a separate required stage.

Recorded measurement:

    160graphs,40updates;134oldgraphreceipts and33updates exact;maxretiredgraphRSS1838792704B,maxbatchRSS6339067904B;zero added swap.

## Programs

- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/finalize.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/finalize.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/iso_common.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/iso_common.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/prepare.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/prepare.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/run_fit.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/run_fit.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/state_io.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/worker.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/worker.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/iso_common.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/iso_common.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/state_io.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/worker.py](code/codex_resume_20260919_v1/current_p_batch_process_qualification_v1/worker.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/train.py](code/codex_resume_20260919_v1/current_p_signed_branch_fit_v1/train.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/allocator.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/allocator.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/common.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/common.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/guard.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/guard.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_branch.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_branch.py)

</details>

## Evidence

- [fit_result.json#L1](fit_result.json#L1)

<details>
<summary>All result files and records (1)</summary>

- [fit_result.json](fit_result.json) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-21-process-split-signed-message-fit-completes-160-graphs-exactly) · [Copy provenance](../../PROVENANCE.json)

