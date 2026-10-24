# Late isotropic correction x1.2 / x0.8: +0.019% / -0.033% adjusted on 32 networks

2026-10-20T09:18:36Z · experiment · rejected · measured

A 20% change in either direction produced only a flat optimum in the late isotropic correction. Held-out effects were +0.043% / -0.015%.

Recorded measurement:

    [fable-lead] 2026-10-20T09:18:36Z ISO LANES DEAD (32 ranks vs science32_rel_v4a_m15, zero bill): isol_p (GN iso x1.2 at L10-14) all32 +0.019% (se 0.044, lower 19/32), heldout25 +0.043%; isol_m (x0.8) all32 -0.033% (se 0.048, lower 15/32), heldout25 -0.015%. The late iso dial is a flat optimum (a +-20% move changes the MSE by 0.03%). With this the dial space on the #337259 base is exhausted on 32 ranks. NOTE: two lanes on a quiet box take 22 min per pair of arms (the 1.5 h of the night were 4 lanes + probes thrashing).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_isol_m_sci_v1.py](code/candidate_isol_m_sci_v1.py)
- [code/candidate_isol_p_sci_v1.py](code/candidate_isol_p_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6162](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6162)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-late-isotropic-correction-x1-2-x0-8-0-019) · [Copy provenance](../../../PROVENANCE.json)

