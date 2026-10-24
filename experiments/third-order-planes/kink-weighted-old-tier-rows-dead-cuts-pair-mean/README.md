# Kink-weighted old-tier rows, dead cuts and pair-mean ordering hurt final error on eight seeds

2026-10-16T17:41:28Z · experiment · rejected · measured

Generalising option B to other selections with a constant kept count: ordering the u-leg frame coordinates by kink-weighted energy at joins 10/13 passed the 8-seed bar (fallback only, since it is the order half of B); boosting old-tier tail rows at layers 12-14 by a next-layer kink proxy (Gaussian ReLU post moments) displaced the rows the OTT score keeps and lost more with larger beta; choosing the dead cut at layers 12-14 by the same proxy lost. The proxy is blind to the off-diagonal covariance where the error lives.

Recorded measurement:

    ukord_b1 (2d97d6bd): xMSE -0.254% (SE .254), bill +0.007%, xnet -0.248%, 5/8 lower; ukord_b2 -0.113%, 3/8. ottk_b1 (ef5a7463): xMSE +0.218% (SE .18, 4/8 lower); ottk_b2 +0.504% (SE .34, 3/8). dcutk_b1 (9f9072e0): +0.099% (SE .046, 2/8 lower); dcutk_b2 +0.210% (SE .116).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_kinksel_v1/build_ukord_v1.py](code/opus_kinksel_v1/build_ukord_v1.py)
- [code/opus_kinksel_v1/build_ottk_v1.py](code/opus_kinksel_v1/build_ottk_v1.py)
- [code/opus_kinksel_v1/build_dcutk_v1.py](code/opus_kinksel_v1/build_dcutk_v1.py)
- [code/opus_kinksel_v1/grade_vs_jord_v1.py](code/opus_kinksel_v1/grade_vs_jord_v1.py)
- [code/opus_kinksel_v1/cand_s2trj_ukord_b1_v1.py](code/opus_kinksel_v1/cand_s2trj_ukord_b1_v1.py)
- [code/opus_grow_v1/cand_s2tr_grow_jord_v1.py](code/opus_grow_v1/cand_s2tr_grow_jord_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4367](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4367)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4382](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4382)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4389](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4389)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4392](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4392)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4404](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4404)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-kink-weighted-old-tier-rows-dead-cuts-pair-mean) · [Copy provenance](../../../PROVENANCE.json)

