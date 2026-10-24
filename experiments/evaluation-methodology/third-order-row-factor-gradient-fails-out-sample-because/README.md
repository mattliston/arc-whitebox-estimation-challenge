# Third-order row-factor gradient fails out of sample because gradient replay freezes carrier states

2026-10-13T17:10Z · experiment · rejected · measured

The strongest co-adaptation signal C created was the row factor third-order row-factor coefficient (c0) (t -4.74 on 64 gradients), but +0.10 and +0.15 at L8-14 fail out of sample; gradient runs treat the carrier modules as layer constants, so a dial acting through the carriers' state gets a biased gradient while evals measure the full effect.

Recorded measurement:

    (e) +0.05 on 8 -0.204% (se 0.135); (f) +0.10 all-24 +0.044% (se 0.163), OOS +0.168%; (g) +0.15 all-24 +0.376%; native route D 0.961 / 0.966 at +0.10

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/physicist/t146_table.py](code/physicist/t146_table.py)
- [code/scout_native/fragment_k31rx_010_v1.py](code/scout_native/fragment_k31rx_010_v1.py)
- [code/engineer/a22_cutsim/n6y_cutsim_v2.py](code/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [code/engineer/a22_cutsim/pbase_cutsim_v1.py](code/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [code/engineer/a22_cutsim/torch_cutsim_s46_v11.py](code/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [code/engineer/a22_cutsim/torch_push_native_v2.py](code/engineer/a22_cutsim/torch_push_native_v2.py)
- [code/engineer/a22_cutsim/torch_rowcuts_v15.py](code/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [code/physicist/ncar_bank_frozen_s2s.py](code/physicist/ncar_bank_frozen_s2s.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2150](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2150)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2151](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2151)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2158](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2158)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2160](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2160)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2166](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2166)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2169](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2169)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2172](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2172)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2174](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2174)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2175](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2175)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2176](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2176)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2179](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2179)

<details>
<summary>All result files and records (153)</summary>

- [allowed_seeds.txt](allowed_seeds.txt) · result
- [ph145_ctl_r7100584.w0.json](ph145_ctl_r7100584.w0.json) · result
- [ph145a_r7100584.w1.json](ph145a_r7100584.w1.json) · result
- [ph145a_r7100585.w2.json](ph145a_r7100585.w2.json) · result
- [ph145a_r7100586.w0.json](ph145a_r7100586.w0.json) · result
- [ph145a_r7100587.w1.json](ph145a_r7100587.w1.json) · result
- [ph145a_r7100588.w2.json](ph145a_r7100588.w2.json) · result
- [ph145a_r7100589.w0.json](ph145a_r7100589.w0.json) · result
- [ph145a_r7100590.w1.json](ph145a_r7100590.w1.json) · result
- [ph145a_r7100591.w2.json](ph145a_r7100591.w2.json) · result
- [ph145b_r7100584.w0.json](ph145b_r7100584.w0.json) · result
- [ph145b_r7100585.w1.json](ph145b_r7100585.w1.json) · result
- [ph145b_r7100586.w2.json](ph145b_r7100586.w2.json) · result
- [ph145b_r7100587.w0.json](ph145b_r7100587.w0.json) · result
- [ph145b_r7100588.w1.json](ph145b_r7100588.w1.json) · result
- [ph145b_r7100589.w2.json](ph145b_r7100589.w2.json) · result
- [ph145b_r7100590.w0.json](ph145b_r7100590.w0.json) · result
- [ph145b_r7100591.w1.json](ph145b_r7100591.w1.json) · result
- [ph145bz_ph146e_r7100584.w2.json](ph145bz_ph146e_r7100584.w2.json) · result
- [ph145bz_ph146e_r7100585.w0.json](ph145bz_ph146e_r7100585.w0.json) · result
- [ph145bz_ph146e_r7100586.w1.json](ph145bz_ph146e_r7100586.w1.json) · result
- [ph145bz_ph146e_r7100587.w2.json](ph145bz_ph146e_r7100587.w2.json) · result
- [ph145bz_ph146e_r7100588.w0.json](ph145bz_ph146e_r7100588.w0.json) · result
- [ph145bz_ph146e_r7100589.w1.json](ph145bz_ph146e_r7100589.w1.json) · result
- [ph145bz_ph146e_r7100590.w2.json](ph145bz_ph146e_r7100590.w2.json) · result
- [ph145bz_ph146e_r7100591.w0.json](ph145bz_ph146e_r7100591.w0.json) · result
- [ph145bz_ph146f_r7100584.w1.json](ph145bz_ph146f_r7100584.w1.json) · result
- [ph145bz_ph146f_r7100585.w2.json](ph145bz_ph146f_r7100585.w2.json) · result
- [ph145bz_ph146f_r7100586.w0.json](ph145bz_ph146f_r7100586.w0.json) · result
- [ph145bz_ph146f_r7100587.w1.json](ph145bz_ph146f_r7100587.w1.json) · result
- [ph145bz_ph146f_r7100588.w2.json](ph145bz_ph146f_r7100588.w2.json) · result
- [ph145bz_ph146f_r7100589.w0.json](ph145bz_ph146f_r7100589.w0.json) · result
- [ph145bz_ph146f_r7100590.w1.json](ph145bz_ph146f_r7100590.w1.json) · result
- [ph145bz_ph146f_r7100591.w2.json](ph145bz_ph146f_r7100591.w2.json) · result
- [ph145bz_ph146g_r7100584.w1.json](ph145bz_ph146g_r7100584.w1.json) · result
- [ph145bz_ph146g_r7100585.w0.json](ph145bz_ph146g_r7100585.w0.json) · result
- [ph145bz_ph146g_r7100586.w1.json](ph145bz_ph146g_r7100586.w1.json) · result
- [ph145bz_ph146g_r7100587.w2.json](ph145bz_ph146g_r7100587.w2.json) · result
- [ph145bz_ph146g_r7100588.w0.json](ph145bz_ph146g_r7100588.w0.json) · result
- [ph145bz_ph146g_r7100589.w1.json](ph145bz_ph146g_r7100589.w1.json) · result
- [ph145bz_ph146g_r7100590.w0.json](ph145bz_ph146g_r7100590.w0.json) · result
- [ph145bz_ph146g_r7100591.w2.json](ph145bz_ph146g_r7100591.w2.json) · result
- [ph145bz_ph146xf_r7100592.w2.json](ph145bz_ph146xf_r7100592.w2.json) · result
- [ph145bz_ph146xf_r7100593.w1.json](ph145bz_ph146xf_r7100593.w1.json) · result
- [ph145bz_ph146xf_r7100594.w0.json](ph145bz_ph146xf_r7100594.w0.json) · result
- [ph145bz_ph146xf_r7100595.w2.json](ph145bz_ph146xf_r7100595.w2.json) · result
- [ph145bz_ph146xf_r7100596.w1.json](ph145bz_ph146xf_r7100596.w1.json) · result
- [ph145bz_ph146xf_r7100597.w0.json](ph145bz_ph146xf_r7100597.w0.json) · result
- [ph145bz_ph146xf_r7100598.w2.json](ph145bz_ph146xf_r7100598.w2.json) · result
- [ph145bz_ph146xf_r7100599.w1.json](ph145bz_ph146xf_r7100599.w1.json) · result
- [ph145bz_ph146xf_r7100600.w0.json](ph145bz_ph146xf_r7100600.w0.json) · result
- [ph145bz_ph146xf_r7100601.w2.json](ph145bz_ph146xf_r7100601.w2.json) · result
- [ph145bz_ph146xf_r7100602.w1.json](ph145bz_ph146xf_r7100602.w1.json) · result
- [ph145bz_ph146xf_r7100603.w2.json](ph145bz_ph146xf_r7100603.w2.json) · result
- [ph145bz_ph146xf_r7100604.w0.json](ph145bz_ph146xf_r7100604.w0.json) · result
- [ph145bz_ph146xf_r7100605.w1.json](ph145bz_ph146xf_r7100605.w1.json) · result
- [ph145bz_ph146xf_r7100606.w2.json](ph145bz_ph146xf_r7100606.w2.json) · result
- [ph145bz_ph146xf_r7100607.w0.json](ph145bz_ph146xf_r7100607.w0.json) · result
- [ph145bz_ph146xg_r7100592.w1.json](ph145bz_ph146xg_r7100592.w1.json) · result
- [ph145bz_ph146xg_r7100593.w2.json](ph145bz_ph146xg_r7100593.w2.json) · result
- [ph145bz_ph146xg_r7100594.w0.json](ph145bz_ph146xg_r7100594.w0.json) · result
- [ph145bz_ph146xg_r7100595.w1.json](ph145bz_ph146xg_r7100595.w1.json) · result
- [ph145bz_ph146xg_r7100596.w2.json](ph145bz_ph146xg_r7100596.w2.json) · result
- [ph145bz_ph146xg_r7100597.w0.json](ph145bz_ph146xg_r7100597.w0.json) · result
- [ph145bz_ph146xg_r7100598.w1.json](ph145bz_ph146xg_r7100598.w1.json) · result
- [ph145bz_ph146xg_r7100599.w2.json](ph145bz_ph146xg_r7100599.w2.json) · result
- [ph145bz_ph146xg_r7100600.w0.json](ph145bz_ph146xg_r7100600.w0.json) · result
- [ph145bz_ph146xg_r7100601.w1.json](ph145bz_ph146xg_r7100601.w1.json) · result
- [ph145bz_ph146xg_r7100602.w2.json](ph145bz_ph146xg_r7100602.w2.json) · result
- [ph145bz_ph146xg_r7100603.w0.json](ph145bz_ph146xg_r7100603.w0.json) · result
- [ph145bz_ph146xg_r7100604.w1.json](ph145bz_ph146xg_r7100604.w1.json) · result
- [ph145bz_ph146xg_r7100605.w2.json](ph145bz_ph146xg_r7100605.w2.json) · result
- [ph145bz_ph146xg_r7100606.w0.json](ph145bz_ph146xg_r7100606.w0.json) · result
- [ph145bz_ph146xg_r7100607.w1.json](ph145bz_ph146xg_r7100607.w1.json) · result
- [ph145_ctl_r7100584.json](ph145_ctl_r7100584.json) · result
- [ph145a_r7100584.json](ph145a_r7100584.json) · result
- [ph145a_r7100585.json](ph145a_r7100585.json) · result
- [ph145a_r7100586.json](ph145a_r7100586.json) · result
- [ph145a_r7100587.json](ph145a_r7100587.json) · result
- [ph145a_r7100588.json](ph145a_r7100588.json) · result
- [ph145a_r7100589.json](ph145a_r7100589.json) · result
- [ph145a_r7100590.json](ph145a_r7100590.json) · result
- [ph145a_r7100591.json](ph145a_r7100591.json) · result
- [ph145b_r7100584.json](ph145b_r7100584.json) · result
- [ph145b_r7100585.json](ph145b_r7100585.json) · result
- [ph145b_r7100586.json](ph145b_r7100586.json) · result
- [ph145b_r7100587.json](ph145b_r7100587.json) · result
- [ph145b_r7100588.json](ph145b_r7100588.json) · result
- [ph145b_r7100589.json](ph145b_r7100589.json) · result
- [ph145b_r7100590.json](ph145b_r7100590.json) · result
- [ph145b_r7100591.json](ph145b_r7100591.json) · result
- [ph146e_r7100584.json](ph146e_r7100584.json) · result
- [ph146e_r7100585.json](ph146e_r7100585.json) · result
- [ph146e_r7100586.json](ph146e_r7100586.json) · result
- [ph146e_r7100587.json](ph146e_r7100587.json) · result
- [ph146e_r7100588.json](ph146e_r7100588.json) · result
- [ph146e_r7100589.json](ph146e_r7100589.json) · result
- [ph146e_r7100590.json](ph146e_r7100590.json) · result
- [ph146e_r7100591.json](ph146e_r7100591.json) · result
- [ph146f_r7100584.json](ph146f_r7100584.json) · result
- [ph146f_r7100585.json](ph146f_r7100585.json) · result
- [ph146f_r7100586.json](ph146f_r7100586.json) · result
- [ph146f_r7100587.json](ph146f_r7100587.json) · result
- [ph146f_r7100588.json](ph146f_r7100588.json) · result
- [ph146f_r7100589.json](ph146f_r7100589.json) · result
- [ph146f_r7100590.json](ph146f_r7100590.json) · result
- [ph146f_r7100591.json](ph146f_r7100591.json) · result
- [ph146g_r7100584.json](ph146g_r7100584.json) · result
- [ph146g_r7100585.json](ph146g_r7100585.json) · result
- [ph146g_r7100586.json](ph146g_r7100586.json) · result
- [ph146g_r7100587.json](ph146g_r7100587.json) · result
- [ph146g_r7100588.json](ph146g_r7100588.json) · result
- [ph146g_r7100589.json](ph146g_r7100589.json) · result
- [ph146g_r7100590.json](ph146g_r7100590.json) · result
- [ph146g_r7100591.json](ph146g_r7100591.json) · result
- [ph146xf_r7100592.json](ph146xf_r7100592.json) · result
- [ph146xf_r7100593.json](ph146xf_r7100593.json) · result
- [ph146xf_r7100594.json](ph146xf_r7100594.json) · result
- [ph146xf_r7100595.json](ph146xf_r7100595.json) · result
- [ph146xf_r7100596.json](ph146xf_r7100596.json) · result
- [ph146xf_r7100597.json](ph146xf_r7100597.json) · result
- [ph146xf_r7100598.json](ph146xf_r7100598.json) · result
- [ph146xf_r7100599.json](ph146xf_r7100599.json) · result
- [ph146xf_r7100600.json](ph146xf_r7100600.json) · result
- [ph146xf_r7100601.json](ph146xf_r7100601.json) · result
- [ph146xf_r7100602.json](ph146xf_r7100602.json) · result
- [ph146xf_r7100603.json](ph146xf_r7100603.json) · result
- [ph146xf_r7100604.json](ph146xf_r7100604.json) · result
- [ph146xf_r7100605.json](ph146xf_r7100605.json) · result
- [ph146xf_r7100606.json](ph146xf_r7100606.json) · result
- [ph146xf_r7100607.json](ph146xf_r7100607.json) · result
- [ph146xg_r7100592.json](ph146xg_r7100592.json) · result
- [ph146xg_r7100593.json](ph146xg_r7100593.json) · result
- [ph146xg_r7100594.json](ph146xg_r7100594.json) · result
- [ph146xg_r7100595.json](ph146xg_r7100595.json) · result
- [ph146xg_r7100596.json](ph146xg_r7100596.json) · result
- [ph146xg_r7100597.json](ph146xg_r7100597.json) · result
- [ph146xg_r7100598.json](ph146xg_r7100598.json) · result
- [ph146xg_r7100599.json](ph146xg_r7100599.json) · result
- [ph146xg_r7100600.json](ph146xg_r7100600.json) · result
- [ph146xg_r7100601.json](ph146xg_r7100601.json) · result
- [ph146xg_r7100602.json](ph146xg_r7100602.json) · result
- [ph146xg_r7100603.json](ph146xg_r7100603.json) · result
- [ph146xg_r7100604.json](ph146xg_r7100604.json) · result
- [ph146xg_r7100605.json](ph146xg_r7100605.json) · result
- [ph146xg_r7100606.json](ph146xg_r7100606.json) · result
- [ph146xg_r7100607.json](ph146xg_r7100607.json) · result
- [worker_w0.log](worker_w0.log) · log
- [worker_w1.log](worker_w1.log) · log
- [worker_w2.log](worker_w2.log) · log
- [pernet_grads_v1.json](../../dials/kink-dead-unit-top-mode-features-fail-predict-per/pernet_grads_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [K4X_DECISION](K4X_DECISION) · record

</details>

## Related work

- ← enables: [Hub quartic ages 3-4 extended to layers 8-15: -0.513% mean-log MSE on 24 networks](../../../records/research/2026-10-13-hub-quartic-ages-3-4-extended-layers-8-15/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-third-order-row-factor-gradient-fails-out-sample-because) · [Copy provenance](../../../PROVENANCE.json)

