# Gain-normalized propagation produces 39.4e-9 error against 16.3e-9 laboratory error

2026-10-01T20:15Z · experiment · rejected · measured

Because a bias-free ReLU network is homogeneous, the collective gain can be divided out exactly at every layer (u = x / G, G = m.x/|m|^2) instead of being tracked through third- and fourth-order cumulants. The formula test looked promising (in frame B the mean formula without any fourth cumulant matched the laboratory formula with the exact fourth cumulant), but Monte Carlo showed the normalised state is not Gaussian, the re-normalisation needs the third cumulant contracted with the mean direction (one extra full product per layer), and a hook-level frame-B card with 15 oracle fourth-order scalars gave 39.4e-9 against 16.3e-9 for the laboratory card. Parked; what survives is that the gain sector is harmless and the fitted birth corrections are worth about a quarter of the physics-only error.

Recorded measurement:

    frame-B formula test layer 15: 6.3e-9 vs 4.8e-9 noise floor (laboratory without fourth cumulant 24e-9); normalised mean excess kurtosis +0.012 to +0.019 at layers 6-15; covariance response eps 1.0-1.7 x var(t); closed loop learned off: laboratory 16.3e-9 (pure physics 22.1e-9), frame B 39.4e-9 (pure 39.7e-9); carried covariance error frame B 0.54 / 0.83 / 1.11% vs 0.49 / 0.69 / 0.80% at layers 6 / 10 / 12; births without fitted corrections +31%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/e9_frameB_formula.py](code/opus_reexam_20261001_v1/e9_frameB_formula.py)
- [code/opus_reexam_20261001_v1/e10_frameB_mc.py](code/opus_reexam_20261001_v1/e10_frameB_mc.py)
- [code/opus_reexam_20261001_v1/e10_analyze.py](code/opus_reexam_20261001_v1/e10_analyze.py)
- [code/opus_reexam_20261001_v1/e11_frameB_card.py](code/opus_reexam_20261001_v1/e11_frameB_card.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)

<details>
<summary>Additional programs and dependencies (15)</summary>

- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
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

</details>

## Evidence

- [SESSION_NOTES_2026-10-01_to_10-03.md#L148](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L148)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L167](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L167)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L185](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L185)

<details>
<summary>All result files and records (3)</summary>

- [e10_r35.log](e10_r35.log) · log
- [e11_rank0035.json](e11_rank0035.json) · result
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-gain-normalized-propagation-produces-39-4e-9-error-against) · [Copy provenance](../../../PROVENANCE.json)

