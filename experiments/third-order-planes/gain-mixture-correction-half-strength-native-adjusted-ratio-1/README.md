# Gain-mixture correction at half strength: native adjusted ratio 1.0580 with 4/40 wins

2026-09-23T01:40Z · experiment · rejected · measured

Adding half of the missing common-gain third-cumulant term worsened the native score; a heads-only refit recovered only a small part of the stronger correction loss.

Recorded measurement:

    2026-09-22 18:40 porter: parity OK. native g0 r35 = bank bit-exact (1.648017257382719e-08); native/torch final MSE g.5 r35 -0.08% r100 -0.17%, g.85 r35 +0.28% r100 +0.36% (r100 g0 baseline already -0.18%); layer traces: fix coefficient matches to float precision, gap = card fidelity. Starting paired VALID40 of plain g0.5 vs bank.
    2026-09-22 18:53 porter: paired VALID40 plain g0.5 (bank params) vs bank: raw_ratio 1.0579, adj_ratio 1.0580, adj_wins 4/40, util 0.18887 vs 0.18886 (fable_porter_v1/paired_valid40_bfix_g05_plain.json). Tracing r37 selections next (native/torch +0.96% at g0.5).
    2026-09-22 19:07 porter: VALID40 pooled parity native/torch: g0 1.00048, g0.5 1.00018 (per-net sd 0.26%/0.39%, max 0.61%/0.99%); fix effect native 1.0579 vs torch 1.0583. Port verdict: faithful. RESULT: fable_porter_v1/RESULT.md. Eval service: bash fable_porter_v1/eval_refit.sh <pt> <gamma> <tag>.
    2026-09-22 20:05 analyst: DONE (fable_analyst_v1/RESULT.md). b fix closed in all modes: physical per-unit state worsens vs MC truth; no learned consumer responsible (head even compensates); heads-only refit proxy recovers 4% of a 32% loss (VALID8 2.386e-8 vs control 1.893e-8); the TRUE plane is 5-9x worse via star transport (noise control neutral). Levers: per-unit v/k3/k4 + carried covariance, not the plane.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/eval_refit.sh](code/fable_porter_v1/eval_refit.sh)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/fable_porter_v1/build_bfix_variant.py](code/fable_porter_v1/build_bfix_variant.py)
- [code/fable_porter_v1/build_refit_candidate.py](code/fable_porter_v1/build_refit_candidate.py)
- [code/fable_porter_v1/cand_bfix_g05.py](code/fable_porter_v1/cand_bfix_g05.py)
- [code/fable_refit_v1/export_params.py](code/fable_refit_v1/export_params.py)
- [code/fable_refit_v1/paired_eval.py](code/fable_refit_v1/paired_eval.py)
- [code/fable_refit_v1/score_variant.py](code/fable_refit_v1/score_variant.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L3](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L3)
- [TEAM_LOG_2026-09-22_to_09-26.md#L10](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L10)
- [TEAM_LOG_2026-09-22_to_09-26.md#L13](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L13)
- [TEAM_LOG_2026-09-22_to_09-26.md#L16](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L16)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [paired_valid40_bfix_g05_plain.json](paired_valid40_bfix_g05_plain.json) · result
- [RESULT.md](RESULT.md) · record
- [RESULT-1a40d4c3.md](RESULT-1a40d4c3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-gain-mixture-correction-half-strength-native-adjusted-ratio-1) · [Copy provenance](../../../PROVENANCE.json)

