# Four-level packed factor propagation reduces native wall from 38.4s to 28.2s at +4.1% cost

2026-09-13 · experiment · rejected · measured

Packed four-level Winograd products replaced serial six-level push products. The variant was held because its 43.3-44.2s production boundary was at the edge of the observed passing range.

Recorded measurement:

    Native wall38.4s to28.2s; utilisation0.1834 to0.1910; bill+4.1%; boundary43.3-44.2s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_wall_repair_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/variantA_d4push.py](code/variantA_d4push.py)
- [code/c2_frozen_copy.py](code/c2_frozen_copy.py)
- [code/profile_native.py](code/profile_native.py)

## Evidence

- [RECEIPTS.md#L9](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L9)

<details>
<summary>All result files and records (1)</summary>

- [RECEIPTS.md](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-four-level-packed-factor-propagation-reduces-native-wall-38) · [Copy provenance](../../../PROVENANCE.json)

