# Pair-map skew and response-row regression variants fail fresh or out-of-selection checks

2026-10-13T19:54Z · experiment · rejected · measured

Scaling the pair map's local skew by 0.8 / 0.9 at L8-11 read -0.25% on the screen 8 but +0.36% / +0.03% at 24 with the out-of-selection 16 positive; the pair law's constant regressor on the native response rows read +0.44% on fresh 16; the all-row law fit itself costs +10.8% (the response-row fit is worth that much).

Recorded measurement:

    pe3 -0.2 all-24 +0.358% (OOS +0.659%); -0.1 +0.027% (OOS +0.175%); dfit 3 response rows +0.436% (se 0.278, 6/16); dfit 2 all-row control +10.784% vs C; dfit 4 vs all-row dfit 2 -4.07%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/t148_table.py](code/t148_table.py)
- [code/t150_table.py](code/t150_table.py)
- [code/t153_table.py](code/t153_table.py)
- [code/cutsim_ph153/n6y_cutsim_v2.py](code/cutsim_ph153/n6y_cutsim_v2.py)
- [code/cutsim_ph153/pbase_cutsim_v1.py](code/cutsim_ph153/pbase_cutsim_v1.py)
- [code/cutsim_ph153/torch_cutsim_s46_v11.py](code/cutsim_ph153/torch_cutsim_s46_v11.py)
- [code/cutsim_ph153/torch_push_native_v2.py](code/cutsim_ph153/torch_push_native_v2.py)
- [code/cutsim_ph153/torch_rowcuts_v15.py](code/cutsim_ph153/torch_rowcuts_v15.py)
- [code/ncar_bank_frozen_s2s.py](code/ncar_bank_frozen_s2s.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2187](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2187)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2205](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2205)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2208](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2208)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2212](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2212)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2238](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2238)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2239](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2239)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2244](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2244)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2258](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2258)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2259](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2259)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2260](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2260)

<details>
<summary>All result files and records (2)</summary>

- [complete153.json](complete153.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Native routes add pair-map skew, kurtosis, kink-plane and per-layer correction dials](../../../instruments/native-routes-add-pair-map-skew-kurtosis-kink-plane/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-pair-map-skew-response-row-regression-variants-fail-fresh) · [Copy provenance](../../../PROVENANCE.json)

