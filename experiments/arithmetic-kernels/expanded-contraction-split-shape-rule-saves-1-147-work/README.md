# Expanded contraction-split shape rule saves 1.147% work with rounding-level prediction changes

2026-09-25T18:39Z · experiment · adopted · measured

The wider rule adds profitable shapes beyond the prior splitting package. Its small request increase is covered by the combined candidate residual gate.

Recorded measurement:

    - 2026-09-25 11:39 lead: k-split v5w GO for S28 (engineer; ksplit_fragment_v5w.py 57656b02; replaces v4). Parity vs S24+E-73 on 35/100/37/39: bill -1.147% vs v4w's -1.084%, final-MSE ratio 1.0010 (rounding level), +125 requests/net, estimated +1% user CPU (about +0.7 s wall). No separate CPU pair; S28's residual gate and VALID40 cover it. The engineer also fixes the depth-table entries for rank-320 shapes (dim index 10 gets shallower Winograd than 384; about 0.05% at L13-only, more with the L10 cut) for S28, rounding-level gate.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/ksplit_fragment_v5w.py](code/ksplit_fragment_v5w.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1172](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1172)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332361: middle join rank 320 with refitted learned correction and matched dials](../../../submissions/332361/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-expanded-contraction-split-shape-rule-saves-1-147-work) · [Copy provenance](../../../PROVENANCE.json)

