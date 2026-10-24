# Tree six-level reads and 64-row padding: adjusted error improves 0.948% on 40 networks

2026-09-24T21:33Z · experiment · adopted · measured

A tree implementation of six-level reads preserved predictions and reduced compute. Combining it with row padding and the generalized push path improved work while passing a paired residual gate.

Recorded measurement:

    - 2026-09-24 14:33 [engineer] S14 CANDIDATE = S13 + tree six-level reads + deadcut rows rounded to 64 + m64 (fable_engineer_v1/candS14_mem.py 9003e536; RESULT.md E-54..E-57). Paired 6 nets vs S13: flops -0.950%, user +4.52%; rounding-level predictions (max rel 2.9-3.6e-6) -> needs VALID40 vs S13 rows; projection ~2.935e-9 if raw holds. Pieces (paired 6 nets): tree read6 BIT-EXACT vs S13 (candS13r6x_mem.py a7e18ab7) flops -0.089%, user -4.90%, -1.04k ops/net; m64 on the tree kernel flops -0.914%, user +5.64% (+1.75 s/net; old kernels -0.71% for +7.97%). Gates: einsum PASS, validate PASS, client/server byte-identical to in-process. DROPPED: --ipt v4 row-capacity trees (flops -0.36%, user +18.4%), all divisible reads at six (~0), six-level squares (no product changes).
    - 2026-09-24 14:50 lead: S14 CANDIDATE (engineer E-54..E-57; candS14_mem.py 9003e536) = S13 + tree six-level reads (bit-exact) + deadcut rows rounded to 64 + m64 on the tree kernel. Paired vs S13: flops -0.950%, user CPU +4.5% (~+2 s by v3); predictions at rounding level; projected ~2.935e-9. Porter runs rebuild, smoke, VALID40 vs S13 rows (bar adjusted <= -0.7%, raw within +-0.1%) and stages; the lead fires. Dropped after measurement: ipt v4 row-capacity trees (user +18%), all divisible reads at six levels (~0 bill), six-level squares. Alt-basis Winograd (engineer2): paper -0.77% bill, wall-neutral, shadow prototype next; it then re-prices depth.
    2026-09-24 14:56 porter: S14 STAGED (VALID40 vs S13: adj -0.948% [CI -1.052, -0.844] 40/40, raw -0.013%); fire held on the residual gate (running). S15 switched to S14 + altbasis (e313ab77); its chain + residual gate queued after S14's gate.
    2026-09-24 15:03 porter: S14 residual gate (paired vs S13, 3 reps): cold delta -0.013 s (rule <= +0.03 s) | projected platform max = S13's 0.28 s + delta = 0.267 s (rule <= 0.33 s) -> RESIDUAL GATE PASS

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candS14_mem.py](code/candS14_mem.py)
- [code/candS13r6x_mem.py](code/candS13r6x_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L411](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L411)
- [TEAM_LOG_2026-09-22_to_09-26.md#L412](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L412)
- [TEAM_LOG_2026-09-22_to_09-26.md#L426](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L426)
- [TEAM_LOG_2026-09-22_to_09-26.md#L429](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L429)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332203: tree-based six-level reads, row padding, and general depth selection](../../../submissions/332203/README.md)

[Research graph](../../../graph/index.html#node=2026-09-24-tree-six-level-reads-64-row-padding-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)

