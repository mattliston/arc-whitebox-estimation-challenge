# Rank-32 sketched norm-coupling carrier: two passes lower MSE 0.41% on 50 ranks; one pass raises it 4.50%

2026-10-02T12:21Z · experiment · rejected · measured

A carrier with state a C + Z M Z' (32 columns) transports its columns exactly through the gates and W and sketches only the layer's fresh part with 24 probes (8 top eigenvectors + 16 Gaussian) by a two-pass range finder and projection, then re-cuts to rank 32. On network 35 it kept -2.84% of the full form's -3.45%, but on the hard network 651 it gave -2.0% (lean) against -10.6% for an ideal rank-32 cut: 24 random probes do not find the fresh part's directions (flat spectrum), more probes cost more than they earn (48 law applications are 2.7% of bill before any conversion), the single-pass Nystrom core is unstable, and on the 50 held-out ranks it gave -0.41% (two passes) and +4.50% (one pass from layer 6). Withdrawn as a port form.

Recorded measurement:

    35: rank 32 lean -2.84%, complete law -2.73%, rank 64 with 40 probes -3.71%, rank 16 -1.02%, without two-sided cleanup -3.14%; 651: lean -2.0%, complete -3.5%, rank 64 -7.2%, single pass Halko +11.6%, start at layer 6 -1.2%, 40 random probes -6.9%, probes from carried directions -8.0%, rank 16 with 24 -4.0%; ideal rank-32 cut -10.6% (-11.4% with four forms), full matrix -15.1%; 50 ranks: two passes from layer 0 -0.410% (se 0.285, 30 of 50), +1.68% worse than the vector form; one pass from layer 6 +4.497% (se 0.459, 4 of 50); forward 47 s vs 33 s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/ncar4.py](code/opus_reexam_20261001_v1/ncar4.py)
- [code/opus_reexam_20261001_v1/ncar4_v1_nystrom_unstable.py](code/opus_reexam_20261001_v1/ncar4_v1_nystrom_unstable.py)
- [code/opus_reexam_20261001_v1/e44_sketch_fresh.sh](code/opus_reexam_20261001_v1/e44_sketch_fresh.sh)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)

<details>
<summary>Additional programs and dependencies (28)</summary>

- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py](code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py)
- [code/opus_reexam_20261001_v1/k31c.py](code/opus_reexam_20261001_v1/k31c.py)
- [code/opus_reexam_20261001_v1/k31diag.py](code/opus_reexam_20261001_v1/k31diag.py)
- [code/opus_reexam_20261001_v1/k31l.py](code/opus_reexam_20261001_v1/k31l.py)
- [code/opus_reexam_20261001_v1/k31m.py](code/opus_reexam_20261001_v1/k31m.py)
- [code/opus_reexam_20261001_v1/ncar.py](code/opus_reexam_20261001_v1/ncar.py)
- [code/opus_reexam_20261001_v1/ncar3.py](code/opus_reexam_20261001_v1/ncar3.py)
- [code/opus_reexam_20261001_v1/ncar5.py](code/opus_reexam_20261001_v1/ncar5.py)
- [code/opus_reexam_20261001_v1/ncar6.py](code/opus_reexam_20261001_v1/ncar6.py)
- [code/opus_reexam_20261001_v1/physicist/f4.py](code/opus_reexam_20261001_v1/physicist/f4.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry.py](code/opus_reexam_20261001_v1/physicist/ncarry.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L261](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L261)
- [TEAM_LOG_2026-10-01_to_10-22.md#L283](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L283)
- [TEAM_LOG_2026-10-01_to_10-22.md#L286](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L286)
- [TEAM_LOG_2026-10-01_to_10-22.md#L299](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L299)
- [TEAM_LOG_2026-10-01_to_10-22.md#L302](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L302)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L531](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L531)

<details>
<summary>All result files and records (3)</summary>

- [rank0651.log](rank0651.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-rank-32-sketched-norm-coupling-carrier-two-passes-lower) · [Copy provenance](../../../PROVENANCE.json)

