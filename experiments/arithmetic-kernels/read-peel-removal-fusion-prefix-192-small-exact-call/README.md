# Read-peel removal and fusion prefix 192: small exact call reduction, no 1,500-call lever

2026-10-17T08:08:10Z · experiment · rejected · measured

Remaining read peeling was disabled after unchanged cap decisions, and the historical prebind prefix 128:192 supplied eager fused views. Exact tiny checks passed, but extending the fusion suffix added setup cost with little warm credit; the integrated candidate remained held.

Recorded measurement:

    Setup net -159 calls; saved-16 mean bill +0.0447967%, mean warm calls -115.9375; integrated 447 calls / -0.232% residual; first 10 residual mean +0.436%; aged 20 scan max 18.982 ms.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py](code/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py)
- [code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/price_fusion_prefix_v1.py](code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/price_fusion_prefix_v1.py)
- [code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/price_fusion_suffix_v1.py](code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/price_fusion_suffix_v1.py)
- [code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/price_read_off_cold_v1.py](code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/price_read_off_cold_v1.py)
- [code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/price_read_off_setup_v1.py](code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/price_read_off_setup_v1.py)
- [code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/tiny_fusion192_v1.py](code/lead/runs/astra_crunch_compute_v1/runtime_repair336657_v1/tiny_fusion192_v1.py)
- [code/lead/runs/astra_crunch_runtime_v1/failure336640_v1/candidate_setup_hold_v19e.py](code/lead/runs/astra_crunch_runtime_v1/failure336640_v1/candidate_setup_hold_v19e.py)
- [code/lead/runs/astra_crunch_runtime_v1/failure336657_v1/candidate_setup_freeze_v22b.py](code/lead/runs/astra_crunch_runtime_v1/failure336657_v1/candidate_setup_freeze_v22b.py)
- [code/lead/runs/astra_crunch_runtime_v1/failure336678_v1/build_review_v23c_v1.py](code/lead/runs/astra_crunch_runtime_v1/failure336678_v1/build_review_v23c_v1.py)
- [code/lead/runs/astra_crunch_runtime_v1/failure336678_v1/candidate_gen2_read_off_fusion192_v23c.py](code/lead/runs/astra_crunch_runtime_v1/failure336678_v1/candidate_gen2_read_off_fusion192_v23c.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py](code/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4848](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4848)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4852](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4852)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4854](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4854)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [fragment_read_off_fusion192_v1.py.txt](fragment_read_off_fusion192_v1.py.txt) · code
- [HANDOFF_READ_OFF_FUSION192_v1.json](HANDOFF_READ_OFF_FUSION192_v1.json) · record
- [CLOSURE_FUSION_SUFFIX_v1.json](CLOSURE_FUSION_SUFFIX_v1.json) · result
- [TINY_FUSION192_v1.json](TINY_FUSION192_v1.json) · result
- [FUSION_SUFFIX_PRICE_v1.json](FUSION_SUFFIX_PRICE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-read-peel-removal-fusion-prefix-192-small-exact-call) · [Copy provenance](../../../PROVENANCE.json)

