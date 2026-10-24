# Refitting after fourth-order changes yields no reproducible improvement

2026-10-02T07:03Z · experiment · rejected · measured

A refit of all learned arrays and dials on the card with the column shift and per-unit kurtosis term (strengths trainable, exact gradient through the star basis) was prepared and certified; its 3 x 4-thread pool was refused by the permission system for interfering with other workloads and launched at 2 x 4 after the box was cleared. One large step of the learned arrays along the mean of 36 exact network gradients was +0.21% where first order priced -0.5%, so the arrays have no room on this card; the optimiser was stopped after 3 updates and its pool became the trainer's evaluation pool for dial line searches, which are about ten times faster.

Recorded measurement:

    forward parity of the run M card with the lead's arms: -7.432% / -3.155% vs -7.43 / -3.16 (networks 35 / 100); gradient 73-74 s (2 workers x 4 threads, about 8 updates an hour), update 1 at 454 s; learned-array step: +0.21% (se 0.12), +2.7% at three times, +20% at eight times (optimum about -0.02%); first 14 gradients: K4L strength t -0.3 (0.75 optimal), b source t +4.2, bscale t +3.7, X1 t -2.9, F4 age 1 t -2.3

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)

<details>
<summary>Additional programs and dependencies (34)</summary>

- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/ncar.py](code/opus_reexam_20261001_v1/ncar.py)
- [code/opus_reexam_20261001_v1/ncar4.py](code/opus_reexam_20261001_v1/ncar4.py)
- [code/opus_reexam_20261001_v1/ncar5.py](code/opus_reexam_20261001_v1/ncar5.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar6_audit_frozen_s3d.py](code/opus_reexam_20261001_v1/physicist/ncar6_audit_frozen_s3d.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_fold_frozen_s2w.py](code/opus_reexam_20261001_v1/physicist/ncar_fold_frozen_s2w.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry.py](code/opus_reexam_20261001_v1/physicist/ncarry.py)
- [code/opus_reexam_20261001_v1/physicist/pbase_frozen_s4a.py](code/opus_reexam_20261001_v1/physicist/pbase_frozen_s4a.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508.py](code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508y.py](code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508y.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1339.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1339.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402x.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402x.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402y.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402y.py)
- [code/opus_reexam_20261001_v1/trainer/m_linesearch.py](code/opus_reexam_20261001_v1/trainer/m_linesearch.py)
- [code/opus_reexam_20261001_v1/trainer/master.py](code/opus_reexam_20261001_v1/trainer/master.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/opus_reexam_20261001_v1/trainer/supervisor.py](code/opus_reexam_20261001_v1/trainer/supervisor.py)
- [code/opus_reexam_20261001_v1/trainer/worker.py](code/opus_reexam_20261001_v1/trainer/worker.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L101](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L101)
- [TEAM_LOG_2026-10-01_to_10-22.md#L126](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L126)
- [TEAM_LOG_2026-10-01_to_10-22.md#L127](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L127)
- [TEAM_LOG_2026-10-01_to_10-22.md#L138](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L138)
- [TEAM_LOG_2026-10-01_to_10-22.md#L148](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L148)
- [TEAM_LOG_2026-10-01_to_10-22.md#L152](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L152)
- [TEAM_LOG_2026-10-01_to_10-22.md#L153](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L153)
- [TEAM_LOG_2026-10-01_to_10-22.md#L164](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L164)
- [TEAM_LOG_2026-10-01_to_10-22.md#L165](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L165)
- [TEAM_LOG_2026-10-01_to_10-22.md#L169](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L169)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L448](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L448)
- [RESULT.md#L384](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md#L384)

<details>
<summary>All result files and records (4)</summary>

- [progress.txt](progress.txt) · log
- [RESULT.md](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-refitting-after-fourth-order-changes-yields-no-reproducible-improvement) · [Copy provenance](../../../PROVENANCE.json)

