# Exact repeated-view cache: 179 fewer first-network and 199 fewer warm requests

2026-10-16T01:37:55Z · experiment · adopted · measured

Caching repeated views preserved predictions and FLOPs while reducing charged calls. Synthetic probes showed that request count dominated residual cost and that wrapper or free-trip changes offered limited additional savings.

Recorded measurement:

    first-179,warm-199;free trips+121/+75;netroundtrips-58/-124;floor43.2-43.5Kinstructions/call;wrapper~4.5%;sameobjectrepeats388/warm;tree->plain>=.9MFLOP/request

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_req_diet_v1/cand_G0_gcstats_v1.py](code/lead/runs/opus_req_diet_v1/cand_G0_gcstats_v1.py)
- [code/lead/runs/opus_req_diet_v1/cand_G1_gcfreeze_v1.py](code/lead/runs/opus_req_diet_v1/cand_G1_gcfreeze_v1.py)
- [code/lead/runs/opus_req_diet_v1/cand_MMP_v1.py](code/lead/runs/opus_req_diet_v1/cand_MMP_v1.py)
- [code/lead/runs/opus_req_diet_v1/cand_RQX_v1.py](code/lead/runs/opus_req_diet_v1/cand_RQX_v1.py)
- [code/lead/runs/opus_req_diet_v1/mmp_judge_v1.py](code/lead/runs/opus_req_diet_v1/mmp_judge_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3645](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3645)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3640](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3640)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [RQX_v1_RECEIPT.json](RQX_v1_RECEIPT.json) · result
- [gc_freeze_block_v1.txt](gc_freeze_block_v1.txt) · result
- [gc_stats_block_v1.txt](gc_stats_block_v1.txt) · result
- [guard_mmp_v1.json](guard_mmp_v1.json) · result
- [guard_probe_repeat_v1.json](guard_probe_repeat_v1.json) · result
- [mm_probe_block_v1.txt](mm_probe_block_v1.txt) · result
- [mmp_2b108_v1.json](mmp_2b108_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336367: layer-10 rank 384, compact kernels and exact request cuts; seven runtime failures](../../../submissions/336367/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-exact-repeated-view-cache-179-fewer-first-network-199) · [Copy provenance](../../../PROVENANCE.json)

