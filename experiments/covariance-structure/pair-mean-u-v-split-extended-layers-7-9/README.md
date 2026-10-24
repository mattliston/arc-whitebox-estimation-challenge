# Pair-mean u/v split extended to layers 7-9: bill -0.38%, MSE -0.09% on 8 seeds, +0.91% on 30 store networks, dropped

2026-10-16T18:33:17Z · experiment · rejected · measured

UVP79 extends the u/v pair-mean split (pair means carried on the leading r_u = 256 frame coordinates, in the estimator submitted as #336412 (S2tr) at layers 10-15) to layers 7-9 by setting _LCR_LAYERS = (7..14) and _UVP_ONLY_LAYERS = _LCR_LAYERS + (15,). It passed the 8-seed bar with a wide draw and a real bill cut, then failed on 30 store networks. A tighter r_u 192 at layers 7-9 was also killed (+2.3% MSE on 8 seeds). The three-exact-cohort join schedule bet (COH3) was skipped because of the known +10.7% join-timing cliff.

Recorded measurement:

    UVP79 (10380144) S8: xMSE -0.087% (SE .52, 6/8 lower), bill -0.383% (seed-1 cold -0.468%), xnet -0.470%, aggregate -0.350%; per seed -0.24 -1.02 +1.51 -1.67 -1.61 -0.28 +2.52 +0.09; per layer L10-12 +0.4..+0.7, L14 -0.82. Store n30: xMSE +0.913% (SE .28, 9/30 lower), bill -0.32, xnet +0.59%. Warm-judge requests +517 / +587. uvp79r192 (b98b9d48): warm bill -0.367% more, requests +24, S8 xMSE +2.3%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_ladder2h_v1/cand_s2tr_l2h_uvp79_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_uvp79_v1.py)
- [code/opus_kinkoracle_v1/cand_decomp_uvp79_v1.py](code/opus_kinkoracle_v1/cand_decomp_uvp79_v1.py)
- [code/opus_solo_v1/cand_uvp79r192.py](code/opus_solo_v1/cand_uvp79r192.py)
- [code/opus_ladder2h_v1/cand_s2tr_l2h_coh3_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_coh3_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4428](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4428)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4438](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4438)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4440](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4440)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4444](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4444)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4452](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4452)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4453](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4453)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4455](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4455)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4460](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4460)

<details>
<summary>All result files and records (3)</summary>

- [L2H_uvp79_v1.json](L2H_uvp79_v1.json) · result
- [GRADE_uvp79r192.txt](GRADE_uvp79r192.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-pair-mean-u-v-split-extended-layers-7-9) · [Copy provenance](../../../PROVENANCE.json)

