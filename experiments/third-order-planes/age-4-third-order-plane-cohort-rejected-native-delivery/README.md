# Age-4 third-order plane cohort rejected: native delivery 0.25/0.28 and positive adjusted change

2026-10-13T19:23Z · experiment · rejected · measured

The fourth cohort age in the (3,1) plane fold helps in PyTorch on top of skew- and kurtosis-corrected transport gate (gk) (and -0.157% at n = 24 on top of C), but natively it costs 0.12% of metered FLOPs at any strength, the full-layer route delivers only D 0.25 / 0.28, and the L11-14 band nets +0.056% on the native ten.

Recorded measurement:

    gk+a4k31s6 n = 24 -0.172% (se 0.049); s1.0 vs 0.6 -0.098%; band b11 native -0.27 / +0.13 / -0.02%; T147 (h) at 24 -0.157% (se 0.049, 17/24); band L11-14 native ten MSE +0.028% + bill +0.028%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/scout_native/fragment_k31a4_v1.py](code/scout_native/fragment_k31a4_v1.py)
- [code/scout_native/fragment_k31a4_b1114_v1.py](code/scout_native/fragment_k31a4_b1114_v1.py)
- [code/physicist/t147_table.py](code/physicist/t147_table.py)
- [code/engineer/a22_cutsim/n6y_cutsim_v2.py](code/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [code/engineer/a22_cutsim/pbase_cutsim_v1.py](code/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [code/engineer/a22_cutsim/torch_cutsim_s46_v11.py](code/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [code/engineer/a22_cutsim/torch_push_native_v2.py](code/engineer/a22_cutsim/torch_push_native_v2.py)
- [code/engineer/a22_cutsim/torch_rowcuts_v15.py](code/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [code/physicist/ncar_bank_frozen_s2s.py](code/physicist/ncar_bank_frozen_s2s.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1690](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1690)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1692](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1692)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1704](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1704)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1721](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1721)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1734](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1734)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1766](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1766)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1871](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1871)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1872](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1872)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1898](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1898)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1900](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1900)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2180](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2180)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2198](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2198)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2230](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2230)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2231](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2231)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2241](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2241)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2243](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2243)

<details>
<summary>All result files and records (93)</summary>

- [allowed_seeds.txt](allowed_seeds.txt) · result
- [ph147_z_r7100584.w0.json](ph147_z_r7100584.w0.json) · result
- [ph147h_r7100584.w1.json](ph147h_r7100584.w1.json) · result
- [ph147h_r7100585.w2.json](ph147h_r7100585.w2.json) · result
- [ph147h_r7100586.w0.json](ph147h_r7100586.w0.json) · result
- [ph147h_r7100587.w1.json](ph147h_r7100587.w1.json) · result
- [ph147h_r7100588.w2.json](ph147h_r7100588.w2.json) · result
- [ph147h_r7100589.w0.json](ph147h_r7100589.w0.json) · result
- [ph147h_r7100590.w1.json](ph147h_r7100590.w1.json) · result
- [ph147h_r7100591.w2.json](ph147h_r7100591.w2.json) · result
- [ph147i_r7100584.w0.json](ph147i_r7100584.w0.json) · result
- [ph147i_r7100585.w1.json](ph147i_r7100585.w1.json) · result
- [ph147i_r7100586.w2.json](ph147i_r7100586.w2.json) · result
- [ph147i_r7100587.w0.json](ph147i_r7100587.w0.json) · result
- [ph147i_r7100588.w1.json](ph147i_r7100588.w1.json) · result
- [ph147i_r7100589.w2.json](ph147i_r7100589.w2.json) · result
- [ph147i_r7100590.w0.json](ph147i_r7100590.w0.json) · result
- [ph147i_r7100591.w1.json](ph147i_r7100591.w1.json) · result
- [ph147j_r7100584.w2.json](ph147j_r7100584.w2.json) · result
- [ph147j_r7100585.w2.json](ph147j_r7100585.w2.json) · result
- [ph147j_r7100586.w2.json](ph147j_r7100586.w2.json) · result
- [ph147j_r7100587.w2.json](ph147j_r7100587.w2.json) · result
- [ph147j_r7100588.w2.json](ph147j_r7100588.w2.json) · result
- [ph147j_r7100589.w2.json](ph147j_r7100589.w2.json) · result
- [ph147j_r7100590.w2.json](ph147j_r7100590.w2.json) · result
- [ph147j_r7100591.w2.json](ph147j_r7100591.w2.json) · result
- [ph147k_r7100584.w2.json](ph147k_r7100584.w2.json) · result
- [ph147k_r7100585.w2.json](ph147k_r7100585.w2.json) · result
- [ph147k_r7100586.w2.json](ph147k_r7100586.w2.json) · result
- [ph147k_r7100587.w2.json](ph147k_r7100587.w2.json) · result
- [ph147k_r7100588.w2.json](ph147k_r7100588.w2.json) · result
- [ph147k_r7100589.w2.json](ph147k_r7100589.w2.json) · result
- [ph147k_r7100590.w2.json](ph147k_r7100590.w2.json) · result
- [ph147k_r7100591.w2.json](ph147k_r7100591.w2.json) · result
- [ph147xh_r7100592.w10.json](ph147xh_r7100592.w10.json) · result
- [ph147xh_r7100593.w11.json](ph147xh_r7100593.w11.json) · result
- [ph147xh_r7100594.w12.json](ph147xh_r7100594.w12.json) · result
- [ph147xh_r7100595.w10.json](ph147xh_r7100595.w10.json) · result
- [ph147xh_r7100596.w11.json](ph147xh_r7100596.w11.json) · result
- [ph147xh_r7100597.w12.json](ph147xh_r7100597.w12.json) · result
- [ph147xh_r7100598.w10.json](ph147xh_r7100598.w10.json) · result
- [ph147xh_r7100599.w11.json](ph147xh_r7100599.w11.json) · result
- [ph147xh_r7100600.w12.json](ph147xh_r7100600.w12.json) · result
- [ph147xh_r7100601.w10.json](ph147xh_r7100601.w10.json) · result
- [ph147xh_r7100602.w11.json](ph147xh_r7100602.w11.json) · result
- [ph147xh_r7100603.w12.json](ph147xh_r7100603.w12.json) · result
- [ph147xh_r7100604.w10.json](ph147xh_r7100604.w10.json) · result
- [ph147xh_r7100605.w11.json](ph147xh_r7100605.w11.json) · result
- [ph147xh_r7100606.w12.json](ph147xh_r7100606.w12.json) · result
- [ph147xh_r7100607.w10.json](ph147xh_r7100607.w10.json) · result
- [ph147_z_r7100584.json](ph147_z_r7100584.json) · result
- [ph147h_r7100584.json](ph147h_r7100584.json) · result
- [ph147h_r7100585.json](ph147h_r7100585.json) · result
- [ph147h_r7100586.json](ph147h_r7100586.json) · result
- [ph147h_r7100587.json](ph147h_r7100587.json) · result
- [ph147h_r7100588.json](ph147h_r7100588.json) · result
- [ph147h_r7100589.json](ph147h_r7100589.json) · result
- [ph147h_r7100590.json](ph147h_r7100590.json) · result
- [ph147h_r7100591.json](ph147h_r7100591.json) · result
- [ph147i_r7100584.json](ph147i_r7100584.json) · result
- [ph147i_r7100585.json](ph147i_r7100585.json) · result
- [ph147i_r7100586.json](ph147i_r7100586.json) · result
- [ph147i_r7100587.json](ph147i_r7100587.json) · result
- [ph147i_r7100588.json](ph147i_r7100588.json) · result
- [ph147i_r7100589.json](ph147i_r7100589.json) · result
- [ph147i_r7100590.json](ph147i_r7100590.json) · result
- [ph147i_r7100591.json](ph147i_r7100591.json) · result
- [ph147xh_r7100592.json](ph147xh_r7100592.json) · result
- [ph147xh_r7100593.json](ph147xh_r7100593.json) · result
- [ph147xh_r7100594.json](ph147xh_r7100594.json) · result
- [ph147xh_r7100595.json](ph147xh_r7100595.json) · result
- [ph147xh_r7100596.json](ph147xh_r7100596.json) · result
- [ph147xh_r7100597.json](ph147xh_r7100597.json) · result
- [ph147xh_r7100598.json](ph147xh_r7100598.json) · result
- [ph147xh_r7100599.json](ph147xh_r7100599.json) · result
- [ph147xh_r7100600.json](ph147xh_r7100600.json) · result
- [ph147xh_r7100601.json](ph147xh_r7100601.json) · result
- [ph147xh_r7100602.json](ph147xh_r7100602.json) · result
- [ph147xh_r7100603.json](ph147xh_r7100603.json) · result
- [ph147xh_r7100604.json](ph147xh_r7100604.json) · result
- [ph147xh_r7100605.json](ph147xh_r7100605.json) · result
- [ph147xh_r7100606.json](ph147xh_r7100606.json) · result
- [ph147xh_r7100607.json](ph147xh_r7100607.json) · result
- [worker_b0.log](worker_b0.log) · log
- [worker_b1.log](worker_b1.log) · log
- [worker_b2.log](worker_b2.log) · log
- [worker_w0.log](worker_w0.log) · log
- [worker_w1.log](worker_w1.log) · log
- [worker_w10.log](worker_w10.log) · log
- [worker_w11.log](worker_w11.log) · log
- [worker_w12.log](worker_w12.log) · log
- [worker_w2.log](worker_w2.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Skew- and kurtosis-corrected transport gate adopted: -0.348% pooled MSE on 24 fresh networks](../../../records/research/2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-age-4-third-order-plane-cohort-rejected-native-delivery) · [Copy provenance](../../../PROVENANCE.json)

