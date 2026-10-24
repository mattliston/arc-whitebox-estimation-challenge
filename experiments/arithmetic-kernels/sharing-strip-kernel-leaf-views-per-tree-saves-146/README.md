# Sharing strip-kernel leaf views per tree saves 146 / 122 index requests with identical predictions

2026-10-02T17:42Z · experiment · adopted · measured

Strip products now reuse leaf views keyed by their individual left, right and output trees rather than by the whole product. The native check preserves prediction bytes, bill and reset behavior while reducing first- and second-network index requests.

Recorded measurement:

    146 / 122 fewer index requests; replayed census 285 -> 135 per network on average; 119 product keys and 38 output trees; complete free changes -921 first / -445 other requests; bill -0.023%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/hzs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_ks_v8.py](code/fragment_ks_v8.py)
- [code/unit_views_v8.py](code/unit_views_v8.py)
- [code/sim_side_views.py](code/sim_side_views.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L400](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L400)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333607: learned norm-coupling diagonal input with two fitted steps and fewer requests](../../../submissions/333607/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-sharing-strip-kernel-leaf-views-per-tree-saves-146) · [Copy provenance](../../../PROVENANCE.json)

