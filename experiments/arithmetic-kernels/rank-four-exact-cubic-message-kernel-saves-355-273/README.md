# Rank-four exact cubic-message kernel saves 355,273,664 FLOPs per network; eight-network adjusted -0.204%

2026-10-17T00:37:36Z · experiment · adopted · measured

An exact rank-4 cubic-message kernel (W3, the W^3 message product) saves a fixed 355,273,664 FLOPs per network with no accuracy change on 8 networks and was carried as a stack candidate; the exact-pubmax base with W3 and the zb4 dial set read raw -0.323% and cold adjusted -0.422% against the estimator submitted as #336412 (S2tr) on 12 networks. W3 entered the peel stack and the fired v17 lineage; a constant W3 setup offload (15 first-call requests and 241,216 FLOPs moved to 2 setup requests) was held for a later stack.

Recorded measurement:

    W3 rank4 fixed 8: pooled adjusted -0.20362% (mean -0.22903%, SE 0.14963%, 6/8); exactpubmax+W3+ZB4 e70b6232 native 12 vs S2tr: raw mean -.323191% SE .150535%, cold adjusted -.422332% SE .161639%, 8/12 wins (first 4 vs parent -.986% unrepresentative, extension 8 raw -.14144%). Offload 8ec16e0b: 15 RPC / 241,216 F -> 2 setup RPC / 0 F.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_crunch_exact_stack_v1/run4_v1.py](code/lead/runs/astra_crunch_exact_stack_v1/run4_v1.py)
- [code/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_v3.py](code/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_v3.py)
- [code/lead/runs/astra_crunch_exact_stack_v1/candidate_w3_zb4_v1.py](code/lead/runs/astra_crunch_exact_stack_v1/candidate_w3_zb4_v1.py)
- [code/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/lead/runs/astra_crunch_stack_v1/judge_stack_v1.py](code/lead/runs/astra_crunch_stack_v1/judge_stack_v1.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/w3_early_restore_v1/fragment_v1.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/w3_early_restore_v1/fragment_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4545](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4545)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4574](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4574)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4594](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4594)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4602](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4602)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4772](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4772)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4774](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4774)
- [fragment_v1.py#L1](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/w3_early_restore_v1/fragment_v1.py#L1)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-rank-four-exact-cubic-message-kernel-saves-355-273) · [Copy provenance](../../../PROVENANCE.json)

