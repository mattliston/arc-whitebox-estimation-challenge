# Eight-channel message distillation with loader repair: -.6774163425% Mac32 versus #337393

2026-10-20T23:37:21Z · experiment · adopted · measured

Teacher distillation reduced message width and retained the runtime loader repair. The full-32 and held-out gains were negative on Mac and Linux, but four fresh networks showed a smaller -.18752697% adjusted gain.

Recorded measurement:

    ActualMac original32 V/science32_msg8_loaderfix_parallel5_v1 completed23:20:47Z: adjusted-.6774163425%SE.153604,25/32lower; held25-.6493062252%,19/25lower; raw-.2059519582%,bill-.4724379204%. Frozenv4a -1.01392954%,held-1.00141150%,sciencebarPASS. Originalmsg8Linuxunrepaired whole32-.71230084%,held-.70541343%,raw-.21121174%,bill-.50217879%; loadercost makes restoredversionlessnegative. Fresh92-95 samehostBLAS4 repairedmsg8-.18752697%SE.32111953,3/4lower; raw+.28121350%,bill-.46744780%, noJ2W-sized independentconfirmation. All8bank/candidateguardsclosed,swap0,rescue0. F/grades and A/cloud_vm_v1/results/MSG8_LOADERFIX_FRESH4_GRADE_v1.json/proof saved.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_vm_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/monitor_utilization_v2.py](code/monitor_utilization_v2.py)
- [code/monitor_utilization_v3.py](code/monitor_utilization_v3.py)
- [code/bootstrap_linux_DRAFT.sh](code/bootstrap_linux_DRAFT.sh)
- [code/eh14_gk4m_cloud_v1.py](code/eh14_gk4m_cloud_v1.py)
- [code/microedge_gk4m_cloud_v1.py](code/microedge_gk4m_cloud_v1.py)
- [code/eh14_gk4m_cloud_v3.py](code/eh14_gk4m_cloud_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6380](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6380)

<details>
<summary>All result files and records (13)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [MSG8_LOADERFIX_FRESH4_GRADE_v1.json](MSG8_LOADERFIX_FRESH4_GRADE_v1.json) · result
- [CLOUD_INITIAL_CANDIDATES_v1.json](CLOUD_INITIAL_CANDIDATES_v1.json) · result
- [LOCAL_SSH_ACCESS_v1.json](LOCAL_SSH_ACCESS_v1.json) · result
- [VM_ACCESS_CONNECTED_v1.json](VM_ACCESS_CONNECTED_v1.json) · result
- [PORTABILITY_PLAN_v1.md](PORTABILITY_PLAN_v1.md) · record
- [TRANSFER_MANIFEST_v1.json](TRANSFER_MANIFEST_v1.json) · result
- [VS_BEST_337421_v1.json](VS_BEST_337421_v1.json) · result
- [CURRENT_SOURCE7_v1.json](CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](PARALLEL32_COMPLETE_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [RESULT32_v1.json](RESULT32_v1.json) · result

</details>

## Related work

- shipped in: [Submission #337422: Message width 8 with teacher compression and loader repair](../../../submissions/337422/README.md)

[Research graph](../../../graph/index.html#node=2026-10-20-eight-channel-message-distillation-loader-repair-6774163425-mac32-versus) · [Copy provenance](../../../PROVENANCE.json)

