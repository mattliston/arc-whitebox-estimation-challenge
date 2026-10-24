# Terminal 65-parameter ridge fit on 160 cached training examples fails eight-network confirmation

2026-09-30 · experiment · rejected · measured

A fixed ridge1e-3 fit changed64 terminal head coefficients plus bias using old native graph packets with matching learned parameters. The old packet trajectory and missing unclipped host made this an approximate readout transfer, and its two-network gain did not persist on eight networks.

Recorded measurement:

    Prefix2 raw-.203988994%,adjusted-.303532439%,2wins;pooled8 raw-.0134073745%,adjusted-.1218879943%,5adjusted/4raw wins;raw CI[-.286810943,+.219919971]%,adjusted CI[-.395731709,+.112717054]%;bill-.105173086% inherited from the exact component.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/fit.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/fit.py)
- [code/codex_resume_20260919_v1/full_lower_moment_panel_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_moment_panel_v1/candidate.py)
- [code/codex_resume_20260919_v1/full_lower_readout_fit_v1/common.py](code/codex_resume_20260919_v1/full_lower_readout_fit_v1/common.py)
- [code/codex_resume_20260919_v1/full_lower_readout_fit_v1/fit.py](code/codex_resume_20260919_v1/full_lower_readout_fit_v1/fit.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)
- [code/codex_resume_20260926_v2/candidate_thin_star_route_v1.py](code/codex_resume_20260926_v2/candidate_thin_star_route_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/assess_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/assess_validation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/prepare_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/prepare_validation.py)

<details>
<summary>Additional programs and dependencies (13)</summary>

- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/validate_endpoint.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/validate_endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/validation_common.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/validation_common.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/assess_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/assess_validation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/disabled_control.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/disabled_control.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/prepare_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/prepare_validation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/validate_endpoint.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/validate_endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/validation_common.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/validation_common.py)
- [code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/candidate.py](code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/candidate.py)
- [code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/common.py](code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/common.py)
- [code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/disabled_control.py](code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/disabled_control.py)
- [code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/fit.py](code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/fit.py)
- [code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/readout.py](code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/readout.py)
- [code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/source_edit.py](code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/source_edit.py)

</details>

## Evidence

- [HILL_CLIMB_CURRENT.md#L359](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L359)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [fit_result.json](fit_result.json) · result
- [native_assessment.json](../../../records/research/2026-09-30-report-new-accuracy-fit-separately-compute-savings-already-present/native_assessment.json) · result
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-terminal-65-parameter-ridge-fit-160-cached-training-examples) · [Copy provenance](../../../PROVENANCE.json)

