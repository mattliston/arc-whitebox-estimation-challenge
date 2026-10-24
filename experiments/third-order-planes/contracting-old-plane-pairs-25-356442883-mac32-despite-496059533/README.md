# Contracting old-plane pairs by 25%: +.356442883% Mac32 despite -.496059533% FLOPs

2026-10-21T03:29:44Z · experiment · rejected · measured

The pair contraction retained frame coordinates, output rows and the full bank but removed complete contracted pairs. Both Mac and Linux full-32 tests showed accuracy cost exceeded FLOP savings.

Recorded measurement:

    V-pair75 SCI0315a05f corrected interleaved tariff mean -0.569022% bill/conservative -0.490570%, PRICE_CURRENT32_v2 SHA3258b7de. Source/cache independent review PASS. Original Mac32 completed03:27:22Z: +0.356442883% adjusted/held +0.407530137%, raw +0.856693035%, bill -0.496059533%, SE .195737,13/32lower. Accuracy cost exceeded compute saving: NO bank/fresh/stack. VM PID104957 continues normally, V watcher106563. Historic OP75 pilot8 +2.8559% was uncovered as adverse prior: it cut both u/v with uncentered selection, no exact-rowmean rescue, Fv320/L10–12, so distinct but relevant. Bill common-u contraction priced mean -0.332203%/conservative -0.253885% bill; exact777a TRAIN35/150 target-free observer PID108249 launched03:26:58, jobs2/originalguard. Physical diagnostic only before any native source; no fraction sweep.
    V-pair75 CLOSED on BOTH hosts: Mac+.3564429%/held+.4075301%, VM+.377123%/held+.433415%, bill-.496060% but raw+.877481% VM. No fresh/stack; conditional receipt REJECT_FULL32. Common-u75 diagnostic CLOSED before candidate: two TRAIN captures source/prediction byte exact, full physical read error5.05–5.96% versus failedv75 3.40–3.83%, at smaller modeled -.3322% primary/-.2539% conservative bill. D/old_ucontract_v1/CLOSURE_v1.json c8a8fd5c. No fraction sweep. Bill moves to distinct algebraic reuse in paired reconstruction.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/old_ucontract_v1/watch_physical_v1.py](code/astra_oct20_bill_v1/old_ucontract_v1/watch_physical_v1.py)
- [code/astra_oct20_bill_v1/old_ucontract_v1/price_current32_v1.py](code/astra_oct20_bill_v1/old_ucontract_v1/price_current32_v1.py)
- [code/astra_oct20_bill_v1/old_ucontract_v1/run_capture_v1.py](code/astra_oct20_bill_v1/old_ucontract_v1/run_capture_v1.py)
- [code/astra_oct20_bill_v1/old_ucontract_v1/prepare_capture_v1.py](code/astra_oct20_bill_v1/old_ucontract_v1/prepare_capture_v1.py)
- [code/astra_oct20_bill_v1/old_ucontract_v1/check_observer_v1.py](code/astra_oct20_bill_v1/old_ucontract_v1/check_observer_v1.py)
- [code/astra_oct20_bill_v1/old_ucontract_v1/old_su_hook_v1.py](code/astra_oct20_bill_v1/old_ucontract_v1/old_su_hook_v1.py)
- [code/astra_oct20_bill_v1/old_ucontract_v1/helper_v1.py](code/astra_oct20_bill_v1/old_ucontract_v1/helper_v1.py)
- [code/astra_oct20_bill_v1/old_ucontract_v1/analyze_physical_v1.py](code/astra_oct20_bill_v1/old_ucontract_v1/analyze_physical_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/old_vpair_prune_v1/analyze_v1.py](code/astra_oct20_bill_v1/old_vpair_prune_v1/analyze_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6500](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6500)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6524](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6524)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PRICE_CURRENT32_v1.json](PRICE_CURRENT32_v1.json) · result
- [OBSERVER_CHECK_v1.json](OBSERVER_CHECK_v1.json) · result
- [CLOSURE_v1.json](CLOSURE_v1.json) · result
- [RECIPE_CAPTURE_v1.json](RECIPE_CAPTURE_v1.json) · result
- [PLAN_CAPTURE_v1.json](PLAN_CAPTURE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-contracting-old-plane-pairs-25-356442883-mac32-despite-496059533) · [Copy provenance](../../../PROVENANCE.json)

