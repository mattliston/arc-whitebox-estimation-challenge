# Smaller request and alias reductions preserve memory while removing about 938 warm round trips

2026-09-23T16:55Z · experiment · adopted · measured

The revised request reduction avoided retaining large array scopes. It preserved predictions, metered work and peak memory while removing repeated requests and frees.

Recorded measurement:

    - 2026-09-23 09:55 [engineer] --lean v3 on the fired S7: fable_engineer_v1/candS7L3_mem.py (23731b49) = S7 recipe + --lean. BIT-IDENTICAL to S7 9011916a on r35/r100/r37/r39 (preds, util, cached arrays); einsum/outer audit identical. Paired ABBA (run_mem.sh, nets 35 cold + 100 warm, load 15-18): deterministic: requests -717/net, free trips -251 cold / -221 warm, round trips -968 / -938, freed handles +1,434/net (fnp.split's two half-views per operand), live-array peak 2,405.3 -> 2,405.3 MB, max handles +8. Noisy: the runs flipped between two load regimes (server CPU 77-79 vs 89-91 s, wall 108-115 vs 149-154 s); within a regime, warm r100 residual -20..-43 ms, cold r35 cannot be matched; no reliable CPU/residual delta at this load. Expected on the platform: wall -0.2..-0.45 s/net (about 950 fewer round trips) and residual about -1..-3 ms (-968 trips at 5-7 us, +1,434 handle finalizers at about 2.5 us). fnp.split changes request count only (the same views; kernel arithmetic unchanged). Ready for the porter: S10 = S9 + --lean.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/candS7L3_mem.py](code/fable_engineer_v1/candS7L3_mem.py)
- [code/fable_engineer2_v1/run_mem.sh](code/fable_engineer2_v1/run_mem.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L300](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L300)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332182: upper-block covariance products, lean views, and an epoch-1 refit](../../../submissions/332182/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-smaller-request-alias-reductions-preserve-memory-while-removing-about) · [Copy provenance](../../../PROVENANCE.json)

