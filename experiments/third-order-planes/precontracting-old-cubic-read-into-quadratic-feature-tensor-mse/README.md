# Precontracting the old cubic read into a quadratic feature tensor: MSE -0.48% for bill +1.28%, net +0.79%, rejected

2026-10-14T04:01:42Z · experiment · rejected · measured

The 'capacity' arm precontracts the old-bank cubic read into a symmetric quadratic feature tensor instead of materialising its bank (a cheaper expression of the rank-384 reconstruction than the rank-320 current file). Its native price was 1.28% more bill than the current file (0.32 to 0.33% cheaper than the full rank-384 expression) and the fresh screen did not pay for it.

Recorded measurement:

    price on current rank-320: log bill +1.275887042%; incremental vs full384 -0.316794409% / -0.334556313%; fresh 32 (16 pairs): MSE log -0.482307753%, bill +1.275887042%, net +0.793579289% (SE 0.343349190%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/lead/runs/astra_energy_e1_standalone_v1/finite_e1_science_UNVALIDATED_v2.py](code/lead/runs/astra_energy_e1_standalone_v1/finite_e1_science_UNVALIDATED_v2.py)
- [code/lead/runs/astra_energy_e1_standalone_v1/finite_e1_smoke_UNVALIDATED_v3.py](code/lead/runs/astra_energy_e1_standalone_v1/finite_e1_smoke_UNVALIDATED_v3.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/build_recon_science_held_astra_v1.py](code/scout_card/capacity_recon_prefix_held_astra_v1/build_recon_science_held_astra_v1.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/build_sources_astra_v1.py](code/scout_card/capacity_recon_prefix_held_astra_v1/build_sources_astra_v1.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/candidate_full384_UNVALIDATED_astra_v1.py](code/scout_card/capacity_recon_prefix_held_astra_v1/candidate_full384_UNVALIDATED_astra_v1.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/candidate_recon_prefix_off_UNVALIDATED_astra_v2.py](code/scout_card/capacity_recon_prefix_held_astra_v1/candidate_recon_prefix_off_UNVALIDATED_astra_v2.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/candidate_recon_prefix_on_UNVALIDATED_astra_v2.py](code/scout_card/capacity_recon_prefix_held_astra_v1/candidate_recon_prefix_on_UNVALIDATED_astra_v2.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/finite_recon_science_UNVALIDATED_astra_v1.py](code/scout_card/capacity_recon_prefix_held_astra_v1/finite_recon_science_UNVALIDATED_astra_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/scout_card/capacity_recon_prefix_held_astra_v1/finite_recon_science_UNVALIDATED_astra_v2.py](code/scout_card/capacity_recon_prefix_held_astra_v1/finite_recon_science_UNVALIDATED_astra_v2.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/finite_recon_smoke_UNVALIDATED_astra_v2.py](code/scout_card/capacity_recon_prefix_held_astra_v1/finite_recon_smoke_UNVALIDATED_astra_v2.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/fragment_full384_current_bridge_UNVALIDATED_astra_v1.py](code/scout_card/capacity_recon_prefix_held_astra_v1/fragment_full384_current_bridge_UNVALIDATED_astra_v1.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/fragment_recon_prefix_UNVALIDATED_astra_v1.py](code/scout_card/capacity_recon_prefix_held_astra_v1/fragment_recon_prefix_UNVALIDATED_astra_v1.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/fragment_recon_prefix_UNVALIDATED_astra_v2.py](code/scout_card/capacity_recon_prefix_held_astra_v1/fragment_recon_prefix_UNVALIDATED_astra_v2.py)
- [code/scout_card/capacity_recon_prefix_held_astra_v1/native_recon_price_judge_UNVALIDATED_astra_v1.py](code/scout_card/capacity_recon_prefix_held_astra_v1/native_recon_price_judge_UNVALIDATED_astra_v1.py)
- [code/scout_card/fragment_capacity_ub384_on_UNVALIDATED_astra_v1_v2.py](code/scout_card/fragment_capacity_ub384_on_UNVALIDATED_astra_v1_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2552](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2552)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2565](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2565)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2568](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2568)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2569](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2569)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2613](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2613)

<details>
<summary>All result files and records (2)</summary>

- [SOURCE_COST_SCREEN_v2.json](SOURCE_COST_SCREEN_v2.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-precontracting-old-cubic-read-into-quadratic-feature-tensor-mse) · [Copy provenance](../../../PROVENANCE.json)

