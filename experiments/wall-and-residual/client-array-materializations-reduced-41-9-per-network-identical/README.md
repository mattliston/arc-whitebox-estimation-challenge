# Client array materializations reduced from 41 to 9 per network with identical predictions

2026-10-20T01:50:00Z · experiment · adopted · derived

Following a forum report that client-side materialisations on flopscope arrays (tolist, bool, float, int, asserts) are synchronising round-trips charged to the 0.4 s residual, a census of the #337212 file found 41 per network: 26 tolist fetches in the rank-2 QR basis conditioning check and 15 bool checks (8 data-dependent slice-vs-scatter decisions, 6 young-crop live-in-tail invariants, 1 envelope check). Retiring the conditioning fetch and the invariant checks and calling the repaired basis directly leaves 9 syncs per network with bit-identical predictions on all 32 ranks; this file became the base for all releases.

Recorded measurement:

    probe on rank 35: 26 r.tolist() in _rtp_qr_basis (RQS2), 15 bool(fnp.all/any) (8 DCR, 6 YNT x2 copies, 1 SLR), 224 len() (no sync); nosync dc5c9386: 41 -> 9 syncs/net, max|dy| 0 on rank 35, flops -1,914; 32 ranks: raw/bill/adjusted 0.000% vs rel_mstack1, requests -41/net, flops -1,585; forum 18202: 166 syncs = 5/50 failures -> 50/50 after removal

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/probe_syncpoints_v1.py](code/probe_syncpoints_v1.py)
- [code/candidate_rel_nosync_v1.py](code/candidate_rel_nosync_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6095](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6095)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6103](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6103)

<details>
<summary>All result files and records (2)</summary>

- [RESULT32_v1.json](RESULT32_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-client-array-materializations-reduced-41-9-per-network-identical) · [Copy provenance](../../../PROVENANCE.json)

