# Exact depth peeling of the layer-6 read body: 0.73% bill uncapped, 0.52% under a 750-call budget, 0.42% as routed

2026-10-17T01:20:47Z · experiment · adopted · measured

Peeling the fringes of the main layer-6 Smirnov read body one tree level shallower (exact arithmetic, existing lower tree reused) was priced on full public metadata: 0.730479% of the whole bill on the matched 8 networks at +2,022.5 wrapper calls; a call-budget frontier gave 0.524923% under a strict 750 extra calls per network, and the same 79 cuts repriced on the estimator submitted as #336412 (S2tr) fringes 0.698318% uncapped, 0.671213% in layer order. Measured on the 16 actual routes the peel saved 0.423227% (0.568078% with W3), 0.25 points below the historic projection. The peel shipped in the v17 lineage, and its 18 push routes were later disabled in the hardware repair v21a (+0.367810% bill) when cold networks ran out of time.

Recorded measurement:

    Matched 8: shallow-only 0.681481% at +1,746.5 calls; hybrid 0.465024% at +786; original 0.730479% at +2,022.5. Budget frontier 0/250/500/750/1000 extra calls: 0.061439/0.300450/0.423502/0.555854/0.617075%. Strict 750: 0.52492333%, mean 697.25 / worst 750 calls. S2tr cap750 v2: uncapped 0.698318%, layer-order replay 0.671213% (mean extra 534.75 / max 746). SWAP audit: mean 24 saving 0.004451%. Actual 16: peel mean 1.040723463 Gops, + W3 .355273664 G = 1.395997127 G/network; normalized 0.423227244% / 0.568078023%; historic 0.6712132% overstated by 0.24798596 pp.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_actual16_tariff_v1/recover_p40_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_actual16_tariff_v1/recover_p40_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_actual16_tariff_v1/replay_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_actual16_tariff_v1/replay_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/boundary_client_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/boundary_client_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/boundary_run_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/boundary_run_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/combine_science8_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/combine_science8_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/grade_predictions_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/grade_predictions_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/run_science7_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/run_science7_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/run_session_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_boundary_v1/run_session_v1.py)

<details>
<summary>Additional programs and dependencies (22)</summary>

- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_sm40_v1/price_peel_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/peel_sm40_v1/price_peel_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_s2tr_v1/fragment_swap_nonpeel_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_s2tr_v1/fragment_swap_nonpeel_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_s2tr_v1/price_swap_residual_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_s2tr_v1/price_swap_residual_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_s2tr_v1/replay_actual_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_s2tr_v1/replay_actual_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_s2tr_v1/reprice_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_s2tr_v1/reprice_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_tailprice_v1/frontier_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_tailprice_v1/frontier_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_tailprice_v1/materialize_strict750_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_tailprice_v1/materialize_strict750_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_tailprice_v1/price_tails_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_tailprice_v1/price_tails_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_tailprice_v1/verify_fullprice_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_peel_tailprice_v1/verify_fullprice_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_s2tr_fullstack_v1/aggregate_fixed16_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_s2tr_fullstack_v1/aggregate_fixed16_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_s2tr_fullstack_v1/judge_fullstack_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_s2tr_fullstack_v1/judge_fullstack_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_s2tr_fullstack_v1/run_root8_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_s2tr_fullstack_v1/run_root8_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py](code/opus_reexam_20261001_v1/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py)
- [code/opus_reexam_20261001_v1/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_reexam_20261001_v1/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4582](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4582)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4586](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4586)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4588](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4588)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4592](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4592)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4596](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4596)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4598](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4598)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4602](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4602)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4645](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4645)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4646](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4646)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4648](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4648)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4650](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4650)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4826](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4826)

<details>
<summary>All result files and records (33)</summary>

- [ASSESSMENT_v1.json](ASSESSMENT_v1.json) · result
- [FRONTIER_v1.json](FRONTIER_v1.json) · result
- [DEPLOY_STRICT750_v1.json](DEPLOY_STRICT750_v1.json) · config
- [DEPLOY_S2TR_CAP750_v2.json](DEPLOY_S2TR_CAP750_v2.json) · config
- [SWAP_RESIDUAL_v1.json](SWAP_RESIDUAL_v1.json) · result
- [ACTUAL16_WARM_TARIFF_v1.json](../../../records/research/2026-10-17-depth-peeling-exact-cubic-messages-four-coefficients-sixteen-network/ACTUAL16_WARM_TARIFF_v1.json) · result
- [DISPATCH_matched8_mean_0_v1.json](DISPATCH_matched8_mean_0_v1.json) · result
- [DISPATCH_matched8_mean_250_v1.json](DISPATCH_matched8_mean_250_v1.json) · result
- [DISPATCH_matched8_mean_500_v1.json](DISPATCH_matched8_mean_500_v1.json) · result
- [DISPATCH_matched8_mean_750_v1.json](DISPATCH_matched8_mean_750_v1.json) · result
- [DISPATCH_matched8_mean_1000_v1.json](DISPATCH_matched8_mean_1000_v1.json) · result
- [DISPATCH_matched8_each_0_v1.json](DISPATCH_matched8_each_0_v1.json) · result
- [DISPATCH_matched8_each_250_v1.json](DISPATCH_matched8_each_250_v1.json) · result
- [DISPATCH_matched8_each_500_v1.json](DISPATCH_matched8_each_500_v1.json) · result
- [DISPATCH_matched8_each_750_v1.json](DISPATCH_matched8_each_750_v1.json) · result
- [DISPATCH_matched8_each_1000_v1.json](DISPATCH_matched8_each_1000_v1.json) · result
- [res_16_v1.json](res_16_v1.json) · result
- [res_35_v1.json](res_35_v1.json) · result
- [res_150_v1.json](res_150_v1.json) · result
- [res_200_v1.json](res_200_v1.json) · result
- [res_250_v1.json](res_250_v1.json) · result
- [res_259_v1.json](res_259_v1.json) · result
- [res_277_v1.json](res_277_v1.json) · result
- [res_287_v1.json](res_287_v1.json) · result
- [res_293_v1.json](res_293_v1.json) · result
- [rows.jsonl](rows.jsonl) · result
- [rows-3ea4727d.jsonl](rows-3ea4727d.jsonl) · result
- [rows-a95a679b.jsonl](rows-a95a679b.jsonl) · result
- [rows-6b8eec08.jsonl](rows-6b8eec08.jsonl) · result
- [rows-a96bce18.jsonl](rows-a96bce18.jsonl) · result
- [rows-705e0f1f.jsonl](rows-705e0f1f.jsonl) · result
- [rows-da90ffb6.jsonl](rows-da90ffb6.jsonl) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-exact-depth-peeling-layer-6-read-body-0-73) · [Copy provenance](../../../PROVENANCE.json)

