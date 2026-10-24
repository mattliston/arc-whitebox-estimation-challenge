# Strided level-operation fusion cuts requests 9.5% but increases user CPU 7.9%

2026-09-25T07:50Z · experiment · adopted · measured

Fusing paired operations preserves predictions and work, but non-outermost strided views run slower in the server. The selected package buys residual margin using available wall headroom.

Recorded measurement:

    - 2026-09-25 00:50 lead: E-71 GROUP 1 (engineer lvfuse_fragment_v2 on S23 = candS23_lvfuse2_mem.py 89762ed3):
      - BIT-IDENTICAL on 35/100/37/39 (all 16 layers, flops identical).
      - Requests 31,532 -> 28,534/net (-3.0k, -9.5%; 12 nets; util identical).
      - Setup 1.28 -> 1.83 s at load 12, 362 bound, two-phase (S23's prebind first, then fusion with the guard's leftover time).
      - Handles peak 30.9k (cap 100k).
      - The residual gate (concurrent pairs vs S23) waits for memory (loopR3 holds ~20 GB wired).
      Order: MPS lock fix (refit_mps_v4) first, then the group-1 residual gate, then groups 2-4 (2 first: -0.3% bill), then the depth-7 pricing. S24 composes S23 + lvfuse2 (+ groups) + star-Nystrom (+ loopR3 if it passes).
    - 2026-09-25 01:25 [engineer] E-71 GROUP 1 (level-op fusion) BUILT AND GATED, NOT RECOMMENDED (RESULT.md E-71b): candS23_lvfuse2_mem.py 89762ed3 (two-phase prebind: never below S23's coverage) is bit-identical (4 nets, 16 layers, flops equal) and cuts -3.0k requests/net (-9.5%, 12 nets), residual -6/-6/-15 ms (concurrent matched gate, PASS), BUT server CPU +7.9% user / +18.6% backend (setup-calling user-CPU pair, 6 nets): numpy runs a fused strided pair op 1.5-3x slower than the two single-component ops unless the component axis is outermost. About +1.3-1.5 s platform wall per MLP for -9 ms residual. Compute-safe remainder: top-level fusion -0.75k, transform reuse -0.6k (bill -0.3%), batching -0.7k = about -2.1k (-6.6%).
    - 2026-09-25 01:26 lead: E-71 GROUP 1 COMPUTE GATE (engineer): lvfuse2 is bit-identical, -3.0k requests, residual -6/-6/-15 ms, BUT server user CPU +7.9% (backend +18.6%).
      - KERNEL LAW: a fused pair op on a non-outermost axis runs 1.5-3x slower than the two single ops on server numpy; only the top tree level fuses for free.
      - Net ~+1.3-1.5 s of platform wall/MLP for -9 ms residual: no further fusion.
      lvfuse2 stays inside S24 (bit-identical, bill-neutral, residual margin up, ~+1.4 s wall against ~25 s headroom).
      Remaining E-71: only (b) transform reuse (-0.6k requests, bill -0.3%) as a compute-gated package on S24's base; (a)/(c) skipped. The routing (+33..55 ms) is not fundable by the request diet. The depth-7 pricing must include server CPU under the kernel law.
      Also: under load (10-15) setup slows ~3x (S23 bound 131/362 before the guard); two-phase prebind is the safe pattern.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candS23_lvfuse2_mem.py](code/candS23_lvfuse2_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L830](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L830)
- [TEAM_LOG_2026-09-22_to_09-26.md#L831](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L831)
- [TEAM_LOG_2026-09-22_to_09-26.md#L832](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L832)
- [TEAM_LOG_2026-09-22_to_09-26.md#L833](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L833)
- [TEAM_LOG_2026-09-22_to_09-26.md#L834](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L834)
- [TEAM_LOG_2026-09-22_to_09-26.md#L835](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L835)
- [TEAM_LOG_2026-09-22_to_09-26.md#L836](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L836)
- [TEAM_LOG_2026-09-22_to_09-26.md#L862](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L862)
- [TEAM_LOG_2026-09-22_to_09-26.md#L863](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L863)
- [TEAM_LOG_2026-09-22_to_09-26.md#L864](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L864)
- [TEAM_LOG_2026-09-22_to_09-26.md#L865](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L865)
- [TEAM_LOG_2026-09-22_to_09-26.md#L866](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L866)
- [TEAM_LOG_2026-09-22_to_09-26.md#L867](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L867)
- [TEAM_LOG_2026-09-22_to_09-26.md#L868](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L868)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332255: joint refit, arithmetic-level fusion, and a Nyström residual-plane range finder](../../../submissions/332255/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-strided-level-operation-fusion-cuts-requests-9-5-but) · [Copy provenance](../../../PROVENANCE.json)

