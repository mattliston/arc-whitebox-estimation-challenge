# All push-peeling routes and three reads reverted to parent executors: cold failures removed

2026-10-17T05:13:59Z · experiment · adopted · measured

Original cap decisions and debits were preserved while all push peeling and three read routes used parent executors. Six dead setup families were removed; cold residual instructions fell and all five cold platform networks passed, while two warm residual failures remained.

Recorded measurement:

    1403 public setup calls and 2.920887 GF saved; saved-16 added arithmetic mean .367810%, max .528411%; cold residual instructions −7.789%, warm −.513%; rank 35 saved-truth U +.437684%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py](code/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py)
- [code/lead/runs/astra_crunch_compute_v1/hardware_repair336649_v1/price_all_push_v1.py](code/lead/runs/astra_crunch_compute_v1/hardware_repair336649_v1/price_all_push_v1.py)
- [code/lead/runs/astra_crunch_compute_v1/hardware_repair336649_v1/proof_cap_all_push_v2.py](code/lead/runs/astra_crunch_compute_v1/hardware_repair336649_v1/proof_cap_all_push_v2.py)
- [code/lead/runs/astra_crunch_compute_v1/hardware_repair336649_v1/proof_prune_all_push_v1.py](code/lead/runs/astra_crunch_compute_v1/hardware_repair336649_v1/proof_prune_all_push_v1.py)
- [code/lead/runs/astra_crunch_runtime_v1/failure336640_v1/candidate_setup_hold_v19e.py](code/lead/runs/astra_crunch_runtime_v1/failure336640_v1/candidate_setup_hold_v19e.py)
- [code/lead/runs/astra_crunch_runtime_v1/failure336649_v1/candidate_repair_v21a.py](code/lead/runs/astra_crunch_runtime_v1/failure336649_v1/candidate_repair_v21a.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py](code/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py)
- [code/release/runs/a_astra_runtime_v21a_c8f98b90/payload/estimator.py](code/release/runs/a_astra_runtime_v21a_c8f98b90/payload/estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4826](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4826)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4828](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4828)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4832](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4832)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [fragment_all_push_off_v1.py.txt](fragment_all_push_off_v1.py.txt) · code
- [ALL_PUSH_REPAIR_LEDGER_v1.json](ALL_PUSH_REPAIR_LEDGER_v1.json) · result
- [ALL_PUSH_SETUP_PRICE_v1.json](ALL_PUSH_SETUP_PRICE_v1.json) · result
- [PROOF_PRUNE_ALL_PUSH_v1.json](PROOF_PRUNE_ALL_PUSH_v1.json) · result
- [PROOF_CAP_ALL_PUSH_v2.json](PROOF_CAP_ALL_PUSH_v2.json) · result
- [GUARD_ALL_PUSH_v1.json](GUARD_ALL_PUSH_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336657: parent executors for push products and three reads: two warm residual failures](../../../submissions/336657/README.md)

[Research graph](../../../graph/index.html#node=2026-10-17-all-push-peeling-routes-three-reads-reverted-parent-executors) · [Copy provenance](../../../PROVENANCE.json)

