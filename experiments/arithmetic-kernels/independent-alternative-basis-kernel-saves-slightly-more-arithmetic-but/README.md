# Independent alternative-basis kernel saves slightly more arithmetic but adds 531 warm operation calls

2026-09-24T21:47Z · experiment · rejected · measured

A second algebraic basis reduced steady-state work slightly more, but it introduced extra remote calls. The other basis was selected after an independent product-level comparison.

Recorded measurement:

    - 2026-09-24 14:47 [engineer] ALT-BASIS: standing down (engineer2 owns it; I missed the 14:50 assignment and had built fable_engineer_v1/ipab_fragment.py = --ipab covering single/colpack/batched/W5, basis a21->a11-a21, b12->b12-b11, c12'=c12-c22, C inverse on the output; in process on S13 + tree read6: bill -1.040/-1.055% r35/r100, preds 1.7-2.0e-6). Offered to engineer2; cancelled my two queued ab pairs before they took the lock.
    2026-09-24 14:48 porter: S13+altbasis VALID40 skipped per lead. S15 (9d898520 = S14 + ipab, cat of ipab_fragment.py, byte-identical) gates OK; fire_chain_S15.sh queued after S14's chain (VALID40 vs S14 rows; bar adj <= -0.7%, raw +-0.15%).
    2026-09-24 14:52 [engineer2] --ipab INDEPENDENT CHECK (altbasis/ab_vs_ipab.py: S13 in process, 180 single tree products per net, each run native / engineer2-ab / engineer-ipab on the same operands, native kept): steady-state bill (repeat calls, trees bound) mine -0.66/-0.71 Gflop per net vs ipab -0.72/-0.78 (r35/r100): ipab ~9% better (its output inverse bills mn/4 per level vs my 7mn/16 on the C tree); user CPU equal (warm: mine -5.4%, ipab -5.7%); ops: mine -59, ipab +531 per net warm (+1,007 cold) = ipab's per-call output views + own tree binding; max output difference vs native 1.8e-4 / 1.6e-4 (rounding). First-call bill differences are a harness artefact (mine shares native's IPT trees; ipab binds its own). Verdict: keep ipab; my only edge (tree-side C inverse: ~500-1,000 fewer requests/net, ~0.2-0.4 s wall, +0.016% bill) is not worth porting while S15's wall room is ~30 s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/ipab_fragment.py](code/fable_engineer_v1/ipab_fragment.py)
- [code/fable_porter_v1/fire_chain_S15.sh](code/fable_porter_v1/fire_chain_S15.sh)
- [code/fable_engineer2_v1/altbasis/ab_vs_ipab.py](code/fable_engineer2_v1/altbasis/ab_vs_ipab.py)
- [code/fable_engineer2_v1/altbasis/ab_fragment.py](code/fable_engineer2_v1/altbasis/ab_fragment.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L420](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L420)
- [TEAM_LOG_2026-09-22_to_09-26.md#L422](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L422)
- [TEAM_LOG_2026-09-22_to_09-26.md#L424](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L424)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-independent-alternative-basis-kernel-saves-slightly-more-arithmetic-but) · [Copy provenance](../../../PROVENANCE.json)

