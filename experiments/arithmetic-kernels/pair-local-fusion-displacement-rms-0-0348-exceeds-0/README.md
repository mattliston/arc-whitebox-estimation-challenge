# Pair-local fusion: displacement RMS 0.0348 exceeds the 0.001 exactness cap

2026-10-14T00:02:56Z · experiment · rejected · measured

The pair local fusion (PLF) kernel fragment passed every setup, no-op, counter and bill gate of its native qualification but changed the predictions far beyond the 0.001 displacement cap, so its measured credit was not treated as a scientific pass. A post-failure source study found a strict-channel variant saving about 0.036 to 0.038% and an unchanged-Gaussian variant saving about 0.18% of the bill; both were parked without a native retest.

Recorded measurement:

    displacement/RMSE cap 0.001 failed: maximum 0.03475665, final 0.02681963; measured credit 662,265,856 FLOPs; post-failure toys: unchanged-Gaussian 15Q+52H (~0.177-0.185%), strict-channel 3Q+32H (~0.036-0.038%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_pair_local_fusion_v1/finite_plf_qualification_v1.py](code/lead/runs/astra_pair_local_fusion_v1/finite_plf_qualification_v1.py)
- [code/lead/runs/astra_pair_local_fusion_v1/build_qualification_sources_v2.py](code/lead/runs/astra_pair_local_fusion_v1/build_qualification_sources_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_gnn_energy_v1/native_energy_price_judge_UNVALIDATED_v1.py](code/lead/runs/astra_gnn_energy_v1/native_energy_price_judge_UNVALIDATED_v1.py)
- [code/lead/runs/astra_pair_local_fusion_v1/fragment_plf_unvalidated_v1.py](code/lead/runs/astra_pair_local_fusion_v1/fragment_plf_unvalidated_v1.py)
- [code/lead/runs/astra_pair_local_fusion_v1/fragment_plf_unvalidated_v2.py](code/lead/runs/astra_pair_local_fusion_v1/fragment_plf_unvalidated_v2.py)
- [code/lead/runs/astra_pair_local_fusion_v1/native_plf_qualification_judge_v1.py](code/lead/runs/astra_pair_local_fusion_v1/native_plf_qualification_judge_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/astra_compute_v1_arm/candidate.py](code/scout_native/astra_compute_v1_arm/candidate.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2431](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2431)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2439](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2439)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-pair-local-fusion-displacement-rms-0-0348-exceeds-0) · [Copy provenance](../../../PROVENANCE.json)

