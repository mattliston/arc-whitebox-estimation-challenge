# Least-squares correction refits improve unseen error by 0.17–0.39%; richer readouts fit noise

2026-10-01T22:27Z · experiment · rejected · measured

The final-layer learned correction was refitted by exact least squares on growing sets of training networks. Gains on unseen networks stayed below half a percent and vanished on the 40 validation networks; readouts with 128 hidden units, 126 extra inputs or pairwise products (up to 720 numbers) only fit noise. A converged refit on all 641 networks was estimated at about 1% on fresh nets.

Recorded measurement:

    71 fit nets: fit -0.48%, unseen -0.17% (se 0.10), VAL-A +0.07% (se 0.15); richer readouts fit -0.7..-1.3%, unseen 0..+0.5%; 140 nets: -0.24% (se 0.06) on 100 unseen, +0.02% (se 0.11) VAL-A; 420 nets: -0.26% (card's 65 numbers), -0.39% (720-number readout) on 100 held-out; estimate converged on 641: about 1% fresh, valid40 about 0.3% less

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/trainer/probe_capture.py](code/opus_reexam_20261001_v1/trainer/probe_capture.py)
- [code/opus_reexam_20261001_v1/trainer/probe_fit.py](code/opus_reexam_20261001_v1/trainer/probe_fit.py)
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

<details>
<summary>Additional programs and dependencies (32)</summary>

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
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L30](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L30)
- [TEAM_LOG_2026-10-01_to_10-22.md#L35](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L35)
- [TEAM_LOG_2026-10-01_to_10-22.md#L45](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L45)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-least-squares-correction-refits-improve-unseen-error-0-170) · [Copy provenance](../../../PROVENANCE.json)

