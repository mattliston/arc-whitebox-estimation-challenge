# Truth-directed cumulant corrections worsen raw error: +68%, +105%, +116% and +3.1%

2026-10-20T00:51:00Z · experiment · rejected · measured

Every arm that moved the card's cumulant inputs toward Monte Carlo truth exploded the final MSE: the layer 8-14 k3 diagonal scaled by the truth-optimal 1.08, the truth-fitted per-unit k3 shape (full or centred), the terminal-layer k3 correction evaluated out of sample on fresh networks, and the centred per-unit k4 shape. Small-step line searches along the same directions (terminal strength 0.03-0.10, k3 shape 0.02-0.10, k4 shape 0.1-0.5, a sixth-order k3^2 Gram-Charlier mean term, a terminal per-unit k4 direction, k3 diagonal x0.99/x1.01) found nothing beyond noise and a razor-sharp cliff on the positive side; the clean terminal k3 scale (0.98/1.02/0.95, z-tilt) is at its optimum.

Recorded measurement:

    k4shape (28 ranks): raw +3.14% (2/28 lower), held-out +2.68%; k3 diag x1.08 at L8-14: raw +68% (0/28); k3shape_f +105%; k3shape_c +1400%; terminal k3 correction on fresh nets (64M truths): strength 1.0 +116..132%, 0.5 +26..31%, scale 1.5 +346..398%; parabola f(s) = -12 s + 128 s^2 -> s* 0.047 at -0.28% predicted; measured s=0.03 per net -0.28/-0.14/-0.01/+0.63/+0.09/-0.28 (mean ~0), s=0.06 +0.25%; k3shape 0.02 +0.14%; k4shape small ~0 (4 nets); k3sq S 0.25/0.5/1.0 = +0.05/+0.20/+0.77% (first seed 0.5 -0.03%, 1.0 +0.29%, 2.0 +2.0%); k4t E 0.5 -0.01% (4/6), -0.5 +0.36%, 1.5 +1.09% (optimum ~0.27 for -0.05%); k3 diag x0.99 +0.10%, x1.01 +2%; clean terminal scale 0.98 +0.5..+1.0%, 1.02 +0.3..+0.6%, 0.95 +3.3..+4.9%, tilt -0.0007 +0.8..+1.0%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_k4shape_sci_v1.py](code/candidate_k4shape_sci_v1.py)
- [code/candidate_k3s_p08_sci_v1.py](code/candidate_k3s_p08_sci_v1.py)
- [code/candidate_k3shape_f_sci_v1.py](code/candidate_k3shape_f_sci_v1.py)
- [code/candidate_k3shape_c_sci_v1.py](code/candidate_k3shape_c_sci_v1.py)
- [code/candidate_k3l15_sci_v1.py](code/candidate_k3l15_sci_v1.py)
- [code/candidate_k3sq_sci_v1.py](code/candidate_k3sq_sci_v1.py)
- [code/candidate_k4t_sci_v1.py](code/candidate_k4t_sci_v1.py)
- [code/candidate_k3l15dir2_sci_v1.py](code/candidate_k3l15dir2_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6047](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6047)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6055](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6055)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6057](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6057)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6059](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6059)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6061](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6061)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6069](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6069)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6093](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6093)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6109](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6109)

<details>
<summary>All result files and records (7)</summary>

- [RESULT32_v1.json](RESULT32_v1.json) · result
- [RESULT32_v1-8b1f0755.json](RESULT32_v1-8b1f0755.json) · result
- [eval_k3l15_fresh_v1.log](eval_k3l15_fresh_v1.log) · log
- [eval_k3l15_small_v1.log](eval_k3l15_small_v1.log) · log
- [eval_k3l15clean_v1.log](eval_k3l15clean_v1.log) · log
- [eval_k4shape_small_v1.log](eval_k4shape_small_v1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-truth-directed-cumulant-corrections-worsen-raw-error-68-105) · [Copy provenance](../../../PROVENANCE.json)

