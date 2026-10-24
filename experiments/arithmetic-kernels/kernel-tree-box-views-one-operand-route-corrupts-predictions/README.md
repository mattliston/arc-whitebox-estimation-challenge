# Kernel-tree box views: one operand route corrupts predictions; the other two save -0.011% FLOPs

2026-10-19T05:27:35Z · experiment · adopted · measured

Generic box views of the kernel operand trees (C side at depth 6, all sides at depths 4 and 7) were byte-identical on later ranks but produced garbage on the first network from layer 9; a bisection on rank 35 showed depth-4 and depth-7 boxes alone are exact and the C-side depth-6 boxes are the fault (pure-numpy aliasing of the C views is exact; the leaf matmul writing out= into a strided C slice on the server is not). The A/B-side boxes at depths 4 and 7 were kept as a free byte-identical part worth -0.011% bill; C-side boxes are dead.

Recorded measurement:

    v3 4062239b: ranks 150/200/250 byte-identical, ranks 259/277/287 max|dy| 1.5e-5, bill -0.03..-0.19% per rank, box tree hits 8-13/net; rank 35 raw +547,351% (padding18 mm 1->0, o18 peel 3->2 plans, kernel calls -490); bisection: depth-4 only byte-identical flops -0.043%, depth-7 only byte-identical flops unchanged; ab47 (7 ranks vs 95c): adjusted -0.370%, raw -0.112% (identical to the base), bill -0.258% vs the base -0.247% = -0.011%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_boxv3_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_boxv3_v1.py](code/candidate_boxv3_v1.py)
- [code/candidate_boxv3_ab47.py](code/candidate_boxv3_ab47.py)
- [code/boxv3_block_ab47.py](code/boxv3_block_ab47.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5908](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5908)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5909](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5909)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5911](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5911)

<details>
<summary>All result files and records (6)</summary>

- [science7_v1.log](science7_v1.log) · log
- [science7_ab47.log](science7_ab47.log) · log
- [r35_d4.log](r35_d4.log) · log
- [r35_d7.log](r35_d7.log) · log
- [r35_nodefer.log](r35_nodefer.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337061: level fusion, prebinding refit and two operand box views, graded 1.4427189e-9](../../../submissions/337061/README.md)

[Research graph](../../../graph/index.html#node=2026-10-19-kernel-tree-box-views-one-operand-route-corrupts-predictions) · [Copy provenance](../../../PROVENANCE.json)

