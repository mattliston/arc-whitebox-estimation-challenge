# Shared strip views, selective tree fusion and binding-time operand views reduce requests without changing predictions

2026-10-02T16:25Z · experiment · adopted · measured

Three byte-identical request levers were measured at the boundary against the candidate generation 41 (S41) stack: the strip kernels' leaf views built by one index expression per view with shared operand views (HV: -570 on a first network / -380 warm, 156 index requests instead of 346), the first-load views of the three-request loads for the 128 most-used listed keys built as each key is bound (Rl3_bind128: -286 / +66, binding phase 17,798 requests), and un-fusing only trees above 64M floats (Rlvfs_limit64: -559 / -606 for +1.9% kernel time, about +1.5 s of wall). All three entered the S41 file; the earlier pieces (kurtosis birth by tables, faster pair-law kernels: -274 / -269, bill +0.011%) entered too.

Recorded measurement:

    HV -570 / -380 (bill equal); Rl3_bind128 -286 / +66, binding phase 17,798 (was 17,366), 228 deferred fusions (was 266), first-network bill -0.03%; Rlvfs_limit64 -559 / -606, kernel +1.9%, projected wall 80.9 / 92.4 s (was 79.4 / 90.7), binding phase 18,028, 143 fusions; K4L birth by tables + faster kernels -274 / -269, kernel x0.985, bill +0.011%; K31L4 +577 / +573; a first network costs +1,713 requests (579 first-load views, 432 leaf views, 482 once-per-process constants, 220 unlisted keys); views of all keys at bind: binding phase 19,010 (does not fit)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engineer/hzs/unit_views_v7.py](code/engineer/hzs/unit_views_v7.py)
- [code/release/variants/s41hv_33bac9db_lvfs64_l3b128/candidate.py](code/release/variants/s41hv_33bac9db_lvfs64_l3b128/candidate.py)
- [code/release/first_network_cost.py](code/release/first_network_cost.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L369](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L369)
- [TEAM_LOG_2026-10-01_to_10-22.md#L373](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L373)
- [TEAM_LOG_2026-10-01_to_10-22.md#L374](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L374)
- [TEAM_LOG_2026-10-01_to_10-22.md#L375](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L375)
- [TEAM_LOG_2026-10-01_to_10-22.md#L383](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L383)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333587: young-pair fourth-chaos plane, norm-coupling dial solve and fewer kernel requests](../../../submissions/333587/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-shared-strip-views-selective-tree-fusion-binding-time-operand) · [Copy provenance](../../../PROVENANCE.json)

