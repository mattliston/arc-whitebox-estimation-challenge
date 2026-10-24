# Linux native worker port verifies official source and paired controls before parallel evaluation

2026-10-20T22:04:21Z · tool · informational · measured

The additional Linux host used isolated client/server/judge environments, verified official source fingerprints and cached weights/truth. It retained a separately measured host baseline and unchanged local release qualification.

Recorded measurement:

    User VM is ADDITIONAL capacity; only copy minimal selected code/data, never move/delete laptop files. Connected arc-worker,[removed],us-central1-b,[removed] as arcworker;48vCPU/176GiB RAM/243GBdisk,Ubuntu22.04.5,Python3.10.12 versus Mac3.10.20 explicitly recorded. All files under /home/arcworker/arc-workspace; results /home/arcworker/arc-runs. 385 selected payload files2,225,092,415bytes copied plus16 portfiles181,189bytes; no privatekeys/credentials/venvbinaries copied. Three isolated envs client/server/judge,official0.12.3 source fingerprints and all32weights/64truths VERIFIED; NumPy2.2.6 server/judge,NumPy-freeclient. Guard smokePASSincludingwholegroup timeoutcleanup. LinuxpilotPID5115 bank1→bridge1→bridge2→bank2 ranks35,150,200 started22:02:52Z,not yetfullyutilized; keepVMqueuefullaftercalibration,userrequest100%utilizationwithinRAM. Pilot1CPUactive at22:03:36Z,171GiBavailable,zeroSwap. D/cloud_linux_v1 immutableport; planonVM1ab8a53e,guard2ebaa11b. Originallocalreleasegatesunchanged.
    VMcapacity isramping.48vCPU/176GiB,firstwhole32Linuxbankdoneallvalid,maxobservedgroupRSS5.444GB,zeroswap. SamehostBLAS4bankandmicroedge32ongoing,EH14RT3all32ongoing. Currentfuturequeue twoGK4=.94full32followups (microedge3fa908f4,EH14RT3variant5f2b0838), noassumedgains; sourceimportPASS. Rootlaunchedcapacityv4with5.6GBempiricalreservation/6.5GBretirementcap,19GBavailable-memoryreserve,zeroswap,.2spoll,sharedglobalceiling32; old7.9GBrecordsremainhonored. This isempiricalcharacterizedfamilyscaling,notkernelguarantee. No laptop/releaseguardchange. Earlier52–54%CPU/104–112GiBavailablewasNOTfullutilization; newmeasurementpending. Onlycopiesofselectedpayload, allLaptopfilesretained.
    VM first matched BLAS4 full32 microedge result -.202840% (held25-.226463%), distinct from original Mac -.276832%; architectures must be paired separately. Bank32 BLAS4 complete. EH14RT3 and two GK4=.94 full32 studies running. Teacher capture first7 complete, no truth-target fitting; three edge-width students12/10/8, encoder96/64 and message12/8 singles, and enc96/msg8 + enc64/msg8 joints launched under shared capacityv4. Nine compression candidates native-testing; no score gains assumed. All fitted sources copied back locally with verified hashes, remote originals retained. Shallow96/msg12 is held as dominated by teacher error/cost proxy, not declared full-network failure. Zero-runtime teacher-head repair is being prepared only to compensate compressed-state error.
    Capacity V7 normal and owned-tree timeout cleanup smokePASS23:02:11Z; A/cloud_linux_capacity_v7/SMOKE_RESULT_v1.json copied back. Scheduler0c625815/guard884199a8; new groups kernel6.5GBcap under parent/sys/fs/cgroup/arc-estimator-v5 memory.max160GiB,swap0; guard stays outside parent and enforces original19GB physical reserve. New shared36 ceiling/reservation4.8GB/BLAS4; oldv4records honored, no existing native source mutation. V5/V6 smoke failed before native and are superseded; onlyV7 native authorized. Exactpinned sudo helper only moves verified ownedchild; estimators unprivileged.

## Programs

- [code/fable_oct19_kc_v1/agent_astra_joint_v1/run_current32_parallel5_v2.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/run_current32_parallel5_v2.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/client_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/client_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/prepare_plan_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/prepare_plan_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/server_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/server_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/prepare_fresh_plan_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/prepare_fresh_plan_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/check_source_local_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/check_source_local_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/run_group_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/run_group_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/run_science_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/run_science_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/smoke_guard_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/smoke_guard_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/inspect_environment_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/inspect_environment_v1.py)

<details>
<summary>Additional programs and dependencies (20)</summary>

- [code/astra_oct20_bill_v1/cloud_linux_v1/grade_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/grade_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_linux_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/grade_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/grade_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/move_owned_child_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/move_owned_child_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/server_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/server_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py)
- [code/astra_crunch_physics_v1/guard_memory50_v1.py](code/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py](code/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py)
- [code/astra_oct18_bill_v1/run_guarded_product_census_v4.py](code/astra_oct18_bill_v1/run_guarded_product_census_v4.py)
- [code/astra_oct18_bill_v1/run_guarded_product_census_v5.py](code/astra_oct18_bill_v1/run_guarded_product_census_v5.py)
- [code/astra_oct18_lead_v1/run_current_controls25_v2.py](code/astra_oct18_lead_v1/run_current_controls25_v2.py)
- [code/astra_oct18_small_stack_qualification_v1/run_current32_m15c_fable_v1.py](code/astra_oct18_small_stack_qualification_v1/run_current32_m15c_fable_v1.py)
- [code/fable_oct19_kc_v1/compare_best_v1.py](code/fable_oct19_kc_v1/compare_best_v1.py)
- [code/fable_oct19_kc_v1/compare_relative_v1.py](code/fable_oct19_kc_v1/compare_relative_v1.py)
- [code/opus_fix336687_v1/candidate_v61j.py](code/opus_fix336687_v1/candidate_v61j.py)
- [code/opus_fix336687_v1/grade_store_v1.py](code/opus_fix336687_v1/grade_store_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6340](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6340)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6349](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6349)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6357](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6357)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6373](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6373)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SMOKE_RESULT_v1.json](SMOKE_RESULT_v1.json) · result
- [README_v1.md](README_v1.md) · record
- [MAC_SCORER_CHECK_v1.json](MAC_SCORER_CHECK_v1.json) · result
- [LOCAL_CHECK_v1.json](LOCAL_CHECK_v1.json) · result
- [MAC_SCORER_CHECK_v2.json](MAC_SCORER_CHECK_v2.json) · result
- [RUN_PLAN_v1.json](RUN_PLAN_v1.json) · result
- [BUILD_v7.json](BUILD_v7.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-10-20-linux-native-worker-port-verifies-official-source-paired-controls) · [Copy provenance](../../PROVENANCE.json)

