# Grouped Winograd requests with copied operands: fewer requests but no metered saving

2026-09-23T01:50Z · experiment · rejected · measured

Grouping strided-view operations preserved predictions exactly. Larger cold and newly bound frames consumed the steady-state arithmetic saving.

Recorded measurement:

    - 2026-09-22 18:50 [engineer] grouped-request Winograd kernel (cand_grouped_v2): r35 prediction BIT-IDENTICAL to bank; boundary requests warm r35 45.7k->36.7k, r100 50.0k->43.9k, cold r35 58.4k->57.2k (new frames); bill +0.39% cold/+0.18% warm (top-operand copies). Timing runs going.
    - 2026-09-22 19:40 [engineer] v5 = grouped kernel + grouped read5 + last-level copy elimination where the arena layout allows: 13/13 production shapes bit-exact, r35/r100 final MSE identical to bank; util r35 0.18883 (+0.29%), r100 0.18464 (+0.02%): the steady-state bill saving (-0.45%) is eaten by bigger cold/new-shape frames. Boundary census+timing running.
    - 2026-09-22 19:25 [engineer] DONE (3 h): no fire-ready variant. v5 (grouped requests + grouped read5 + layout-checked last-level copy elimination) is bit-exact at the boundary (5/5 output sha), warm requests 45.7k->35.5k, but util flat (r35 +0.29%, r100 +0.02%) and cold requests 58.4k->63.6k. Frame (view-binding reshape) bill is recurring per network (bank 4.0% first / 1.8% second). Details fable_engineer_v1/RESULT.md.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_grouped.py](code/build_grouped.py)
- [code/cand_bank.py](code/cand_bank.py)
- [code/cand_grouped_v2.py](code/cand_grouped_v2.py)
- [code/cand_grouped_v5.py](code/cand_grouped_v5.py)
- [code/grouped_fragment_v5.py](code/grouped_fragment_v5.py)
- [code/grouped_read5_fragment_v5.py](code/grouped_read5_fragment_v5.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L7](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L7)
- [TEAM_LOG_2026-09-22_to_09-26.md#L14](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L14)
- [TEAM_LOG_2026-09-22_to_09-26.md#L18](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L18)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-grouped-winograd-requests-copied-operands-fewer-requests-but-no) · [Copy provenance](../../../PROVENANCE.json)

