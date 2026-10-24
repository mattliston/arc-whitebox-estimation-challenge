# Outside-frame block-trace completion at layer 13: saved cubic error -16.6%, native adjusted +0.25–0.37%

2026-10-17T00:28:12Z · experiment · rejected · measured

Trace vectors outside the join frame Q (the AAR and RRR blocks, coefficient 1, final join 13, newly joined columns only) were completed with an exact block-harmonic construction that preserves the partial traces. On the saved rank-651 join it recovered 5.1% of the pushed third-order plane squared error and 16.6% of the k3 error, but the native STORE150 run was adverse, three owned ranks all adverse (mean adjusted +0.219848%), and an independent review found an ownership bug (the correction entered the bank-inherited subtraction without the bank carrying it). The repaired v2 was still adverse on 150 (adjusted +0.37493%) and the line was closed pending a new proof.

Recorded measurement:

    Saved 651: K21 squared error recovery 5.1106%, k3 16.5580%, mean-weighted k3 19.1016%, leading cov 3.9648%, next variance 2.2893%. Native STORE150: raw +.235219%, bill +.018479%, adjusted +.253741% (U 1.5037194e-8 vs 1.5001907e-8); STORE16 raw +.290042 / adjusted +.308369%; STORE250 +.078652 / +.097433%; owned 3 mean adjusted +.219848% SE .063205%. v2 (04b57d27) 150: raw +.353855%, adjusted +.37493%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_crunch_apa_v1/candidate_pubmax_fixed_control_v1.py](code/astra_crunch_apa_v1/candidate_pubmax_fixed_control_v1.py)
- [code/astra_crunch_physics_v1/block_trace_completion_v1/candidate_pbt_l13_v1.py](code/astra_crunch_physics_v1/block_trace_completion_v1/candidate_pbt_l13_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4578](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4578)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4580](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4580)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4582](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4582)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4584](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4584)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4590](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4590)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4594](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4594)

<details>
<summary>All result files and records (8)</summary>

- [PAIR150_v1.json](PAIR150_v1.json) · result
- [MATH_SOURCE_RESULT_v1.json](MATH_SOURCE_RESULT_v1.json) · result
- [NATIVE150_PREREG_v1.json](NATIVE150_PREREG_v1.json) · record
- [OWNED3_RESULT_v1.json](OWNED3_RESULT_v1.json) · result
- [REVIEW_v1.json](REVIEW_v1.json) · record
- [res_compute_control150_v1.json](res_compute_control150_v1.json) · result
- [guard_compute_control150_v1.json](guard_compute_control150_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-outside-frame-block-trace-completion-layer-13-saved-cubic) · [Copy provenance](../../../PROVENANCE.json)

