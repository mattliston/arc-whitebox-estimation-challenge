# Gradual frame-cache entry caps preserve predictions with mean work increase 0.039%

2026-09-23T10:16Z · experiment · informational · measured

Incremental eviction inside existing cache insertions bounded remote-array handles without a bulk clear. The remaining cost was small rebinding work and finalizer overhead.

Recorded measurement:

    - 2026-09-23 03:16 [engineer] S6 HANDLE VALVE --caps (caps_fragment.py; gradual LRU entry caps on every frame cache, parents refresh children, child caps >= sum of parent caps, eviction one entry at a time in the put helpers; retires --lru's one-shot 20k drop per engineer2's residual finding). 20 distinct nets in one process, S5-like stack (--l15cut push5 --f32 third --evict5) with vs without --caps: arrays after net 20 61,928 vs 73,152 (flat from net 13 at 56.7-62.0k); predictions bit-identical on all 20 nets; bill identical through net 9, then +0.01..+0.28% per net, mean +0.039% over 20 (steady state ~+0.09%). Quarter-scale caps (store ~18k) cost +0.4..+1.3% per net: caps must hold several nets of working set. File for engineer2's server 20-net test: fable_engineer_v1/candStack_S5e_caps.py.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/caps_fragment.py](code/caps_fragment.py)
- [code/candStack_S5e_caps.py](code/candStack_S5e_caps.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L172](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L172)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-gradual-frame-cache-entry-caps-preserve-predictions-mean-work) · [Copy provenance](../../../PROVENANCE.json)

