# Conservative forward-transform reuse saves 0.09% work and 376-483 requests with identical predictions

2026-09-25T09:10Z · experiment · adopted · measured

A guarded object-and-shape match reuses a prior left transform only while no other arena user can invalidate it. Native output, compute and residual checks passed.

Recorded measurement:

    - 2026-09-25 02:10 [engineer] E-73 TRANSFORM REUSE PASSES ALL GATES (RESULT.md E-73): candS24_treuse2_mem.py 32babb65 = fired S24 (30810da3) + treuse_fragment_v2.py (skip a forward-A whose operand object and (L,m,k) equal the previous one; conservative guard clears on any other arena user). Proof: in-process audit on S24, all 76 such events on 4 nets left the A region bit-identical. Gates: bit-identical 16 layers on 8 nets; bill -0.09% (-0.073..-0.113%/net); user CPU -0.02% (neutral); residual -2/-5/-7 ms, PASS; requests -0.38..-0.48k/net. Ready for the porter to stack.
    - 2026-09-25 02:11 lead: E-73 TRANSFORM REUSE PASSES (engineer; candS24_treuse2_mem.py 32babb65 = S24 + treuse_fragment_v2): bit-identical on 8 nets (15-22 reuses/net, arena proof on 4 nets), bill -0.09% (-0.073..-0.113%), user CPU -0.02% (neutral), residual -2/-5/-7 ms, requests -376..-483. Goes into S25. Rejected: pair-map lookup tables (only 1.6% of the bill, and a DQ-risk pattern). Engineer stands by; the kernel/diet lane is harvested.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candS24_treuse2_mem.py](code/candS24_treuse2_mem.py)
- [code/treuse_fragment_v2.py](code/treuse_fragment_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L940](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L940)
- [TEAM_LOG_2026-09-22_to_09-26.md#L941](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L941)
- [SUBMISSION_LEDGER.md#L885](../../../records/SUBMISSION_LEDGER.md#L885)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [SUBMISSION_LEDGER.md](../../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- shipped in: [Submission #332295: third birth-term row selection, split products, frame reuse, and learned refit](../../../submissions/332295/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-conservative-forward-transform-reuse-saves-0-09-work-376) · [Copy provenance](../../../PROVENANCE.json)

