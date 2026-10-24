# Cache residual creep traced to eviction finalizers; larger entry caps avoid steady-state evictions

2026-09-23T12:13Z · experiment · adopted · measured

Caps sized below the twenty-network working set caused repeated finalization costs. Raising caps to slightly above that working set removed the growing residual penalty.

Recorded measurement:

    - 2026-09-23 05:13 [engineer] S7 RESIDUAL CREEP (engineer2's gate: +4 ms nets 2-8 -> +9..+13 ms nets 13-19) = eviction finalizers once the v1 caps fill (~3k arrays freed per net x ~2.5 us), not LRU bookkeeping (constant ~1-2 ms). FIX --caps2 (caps_fragment_v3.py: caps ~1.1x the 20-net entry counts: BH_F/BH_O/PACKED/READ5_FRAME 280, BH_X 1000, A/B 560, bank O 280, READ5_CHILD 560): S7c2 = fable_engineer_v1/candS7c2_mem.py (791ea953). In-process 20 nets: arrays 75,846 at net 20 (tracks S6b uncapped 74.7k: no costly evictions); only _BH_F hits its cap (net 16), and its evictions free nothing (A/B frames stay cached); bounded near 80k beyond ~net 22. Expected residual cost: constant +2-4 ms, no creep.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/caps_fragment_v3.py](code/caps_fragment_v3.py)
- [code/candS7c2_mem.py](code/candS7c2_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L201](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L201)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332026: cheaper first-network frame binding and padded six-level products](../../../submissions/332026/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-cache-residual-creep-traced-eviction-finalizers-larger-entry-caps) · [Copy provenance](../../../PROVENANCE.json)

