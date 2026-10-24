# Exact-shape frames with restored six-level pushes preserve frozen output bytes at 23.4s native wall

2026-09-13 · experiment · rejected · measured

Restoring five-level squares and then serial six-level push products on exact-shape frames recovered the original arithmetic. The final variant was byte-identical to the frozen shared-rank384 estimator but its 40.4-44.4s boundary remained unsafe.

Recorded measurement:

    Five-level squares17.9s,util0.1993 cold; restored six-level push23.4s versus38.4s,util0.1908 cold,maxabs0.0; boundary40.4-44.4s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_wall_repair_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/variantE6_d6push.py](code/variantE6_d6push.py)
- [code/c2_frozen_copy.py](code/c2_frozen_copy.py)
- [code/profile_native.py](code/profile_native.py)
- [code/remote_series.py](code/remote_series.py)
- [code/time_ladder.sh](code/time_ladder.sh)

## Evidence

- [RECEIPTS.md#L31](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L31)

<details>
<summary>All result files and records (1)</summary>

- [RECEIPTS.md](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-exact-shape-frames-restored-six-level-pushes-preserve-frozen) · [Copy provenance](../../../PROVENANCE.json)

