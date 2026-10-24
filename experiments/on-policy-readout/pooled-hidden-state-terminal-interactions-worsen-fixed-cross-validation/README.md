# Pooled-hidden-state terminal interactions worsen fixed cross-validation relative to the additive head

2026-09-30 · experiment · rejected · measured

A129-feature terminal head adding hidden-state times mean-hidden-state interactions was compared with the65-feature ridge control on old native trajectory training packets. The first runtime failed before scoring; the identical fitter completed under the qualified runtime and failed the scientific diagnostic.

Recorded measurement:

    Augmented raw+.0281301%,adjusted+.0280238%;additive adjusted-.0540019%;augmented versus additive adjusted+.0820700%,2/5fold wins;no full fit/export/native port.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_rrpi_pooled_head_v2/fit.py](code/codex_resume_20260926_v2/hill_climb_rrpi_pooled_head_v2/fit.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/disabled_control.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/disabled_control.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/fit.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/fit.py)
- [code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/source_edit.py](code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/source_edit.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L220](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L220)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [protocol.json](protocol.json) · result
- [cv_result.json](cv_result.json) · result
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-pooled-hidden-state-terminal-interactions-worsen-fixed-cross-validation) · [Copy provenance](../../../PROVENANCE.json)

