# Fourth-order all-hub and age-four extensions: no transferable gain on the deployed estimator

2026-10-20T06:11:43Z · experiment · rejected · measured

All-hub age-one fourth-order terms were inconsistent across two research networks. The apparent age-four gain was already represented by the deployed estimator ages-three-and-four family.

Recorded measurement:

    - [fable-lead] 2026-10-20T06:11:43Z K4L ALL-HUBS / AGE-4 CLOSED: torch k4all=1 (age-1 part from all hubs) -0.23% / +0.21% (nets 35/100) = nothing; K4L age 4 -0.8% / -1.4% in torch but the native card already carries K4L ages 3-4 at the inner layers (K4L34 family at its fitted optimum, TEAM_LOG 10-19 17:53Z). Next native oracle: the K22 plane (truth via the pair map's k22_override; the card's own K22 x2 / x3) at L10-14.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/physicist/e16b_f4b.py](code/opus_reexam_20261001_v1/physicist/e16b_f4b.py)
- [code/opus_reexam_20261001_v1/physicist/f4.py](code/opus_reexam_20261001_v1/physicist/f4.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/k31l_audit.py](code/opus_reexam_20261001_v1/physicist/k31l_audit.py)
- [code/opus_reexam_20261001_v1/physicist/ncar6_audit.py](code/opus_reexam_20261001_v1/physicist/ncar6_audit.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank.py](code/opus_reexam_20261001_v1/physicist/ncar_bank.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_fold.py](code/opus_reexam_20261001_v1/physicist/ncar_fold.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry.py](code/opus_reexam_20261001_v1/physicist/ncarry.py)
- [code/opus_reexam_20261001_v1/physicist/ntor.py](code/opus_reexam_20261001_v1/physicist/ntor.py)
- [code/opus_reexam_20261001_v1/physicist/pbase.py](code/opus_reexam_20261001_v1/physicist/pbase.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)
- [code/opus_reexam_20261001_v1/physicist/so2.py](code/opus_reexam_20261001_v1/physicist/so2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6140](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6140)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-fourth-order-all-hub-age-four-extensions-no-transferable) · [Copy provenance](../../../PROVENANCE.json)

