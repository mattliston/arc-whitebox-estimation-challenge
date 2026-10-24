# Restoring original sixteen-channel projectors after message compression worsens Mac32 +0.2620721%

2026-10-21T00:59:13Z · experiment · rejected · measured

An audited input mismatch showed narrower messages altered data-dependent projectors. Restoring the original carrier inputs was locally exact but worsened full-network accuracy, and corrected width-four variants also failed.

Recorded measurement:

    V found a real compression input mismatch: narrower channels alter GOP2/W3 data-dependent projectors. Correctedbank8 retains original16 source/projectors/private metric, selects8outputs only afterward, denseW1 stays8; current weights unchanged. Official selectedH and state/head oracle byte-exact at n16/n32/n1024, cachehit/warm/reset checks; measured conservativedebit+.0585074%bankbill, +15warmrequests/layer. SCI6995e7db/REL2d508cf3 in V/msg8_parent16_carriers_v1; tiny5b7d2a8c/import6bd880fe. VM full32 PID47909 at00:50:26Z. OriginalMac32 priority wrapper launched00:50:30Z, began00:52:18Z; V/science32_msg8_parent16_carriers_parallel5_v1, session66436, BEST422 PINNED. Oldqueue77740+5judgespaused/drained safely, will auto-resume; no local GPU launch.
    Completed closures: corrected bank8 parent16 carriers SCI6995e7db loses +0.2620721% Mac32 and +0.133101% Linux32 (held +0.125422%); originalMac wrapper ended00:59:17Z, resumed old queue/five judges, BEST422 unpinned. Edge8 + CURRENT rank16 SCI305d8457 is -0.207929% Linux32/held -0.193720%, worse than edge8 alone -0.2140704%; do not sum components or promote. Corrected CM4 carrier-only +0.655903% adjusted/held +0.688501%; fixed first-affine + carriers +0.781722%/held +0.877294%; all32/12 guards good. Close this CM4 compression direction.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_vm_frontier_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_parent_carriers_v3.py](code/build_parent_carriers_v3.py)
- [code/check_frontier_tiny_v5.py](code/check_frontier_tiny_v5.py)
- [code/check_parent_carriers_v4.py](code/check_parent_carriers_v4.py)
- [code/fit_frontier_v1.py](code/fit_frontier_v1.py)
- [code/msg8_parent16_carriers_v1/candidate_msg8_parent16_carriers_sci_v1.py](code/msg8_parent16_carriers_v1/candidate_msg8_parent16_carriers_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6428](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6428)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6437](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6437)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [CURRENT_SOURCE7_v1.json](CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](PARALLEL32_COMPLETE_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [RESULT32_v1.json](RESULT32_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-restoring-original-sixteen-channel-projectors-after-message-compression-worsens) · [Copy provenance](../../../PROVENANCE.json)

