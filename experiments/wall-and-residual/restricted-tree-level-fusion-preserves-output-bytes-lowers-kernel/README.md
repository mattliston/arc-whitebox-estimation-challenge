# Restricted tree-level fusion preserves output bytes and lowers kernel time by 13.1%

2026-10-01T20:44Z · experiment · adopted · measured

The level-fusion layer (in every bank since S24) fuses pairs of tree operations; a numpy bench showed the fused pair costs 1.54-2.86x two single ops on output trees. Un-fusing output trees (fusion kept for left-operand trees and trees up to 32M floats) is bit- and bill-identical and trades about 1,378 requests per network for 13% of kernel time. The fragment (lvfuse_sized32) entered S36 and every later file.

Recorded measurement:

    fuse_bench cycles, fused pair / two single ops: 0.86-1.01x left-operand trees, 1.08-1.64x right-operand, 1.54-2.86x output trees; un-fuse on networks 35,100: bill 0, predictions byte-identical, kernel calls +1,378 (+3.6%), projected platform kernel -13.1% (band -14.6..-10.5)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fuse_bench.py](code/fuse_bench.py)
- [code/rusage.py](code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L25](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L25)
- [TEAM_LOG_2026-10-01_to_10-22.md#L31](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L31)
- [TEAM_LOG_2026-10-01_to_10-22.md#L52](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L52)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333429: float32 arithmetic, half-zero strips, refreshed setup keys and separate tree levels](../../../submissions/333429/README.md)

[Research graph](../../../graph/index.html#node=2026-10-01-restricted-tree-level-fusion-preserves-output-bytes-lowers-kernel) · [Copy provenance](../../../PROVENANCE.json)

