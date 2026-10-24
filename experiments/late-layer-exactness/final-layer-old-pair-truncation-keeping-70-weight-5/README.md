# Final-layer old-pair truncation keeping 70% of weight: +5.56% raw error for -0.318% FLOPs

2026-10-19T10:47:01Z · experiment · rejected · measured

Reconstructing only the heaviest 70% of the old u/v pairs for the final-layer k3 push (hooked into the bank-copy reconstruct path after a first hook on _uvp_recon never fired; 5,888 of 8,320 pairs kept) saved 0.426% of the bill in process but cost raw +5.56% on the first 16 ranks: the final-layer variance prediction is hyper-sensitive to its k3 contraction (a ~5% k3 error -> ~5.6% MSE), so no pair fraction pays (85% would still cost ~+2.5%); the young 85% cut and the combined arm were dropped unrun.

Recorded measurement:

    in-process net 7103256: 248.497 GF vs 249.561 GF = -0.426%; l15pc70b 16 ranks vs stk1: raw +5.56% (se .47, 0/16 lower), bill -0.318%; v1 hook: _L15PC_STATS calls 0, bill +0.00/+0.17/+0.04% on ranks 35/150/200 (noise); l15yc85 keeps 2,912 of 3,392 inherited young pairs, 249.100 GF (-0.185%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_l15pc70b_sci_v1.py](code/candidate_l15pc70b_sci_v1.py)
- [code/candidate_l15pc70_sci_v1.py](code/candidate_l15pc70_sci_v1.py)
- [code/candidate_l15yc85_sci_v1.py](code/candidate_l15yc85_sci_v1.py)
- [code/candidate_l15pcy_sci_v1.py](code/candidate_l15pcy_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5934](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5934)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5935](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5935)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5937](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5937)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5939](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5939)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-final-layer-old-pair-truncation-keeping-70-weight-5) · [Copy provenance](../../../PROVENANCE.json)

