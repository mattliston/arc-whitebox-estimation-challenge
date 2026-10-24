# Float32 reductions and live-row operations lower charged work by 0.669% with rounding-level error changes

2026-10-01T22:18Z · experiment · adopted · measured

Five bill levers without any kernel change were measured in process against the parent: the two first-chaos tail traces in float32 (they had widened float32 operands to float64 at the 2x rate), the join's Nystrom core by one float32 solve R K^-1 R' = Q[rows]^-1 R', the birth update in float32, the mean-correction features' nine plane reductions with float32 accumulation, and the live rows taken as a slice of the pushed rows where the dead-side cut is active (byte-identical). Together bill -0.669%; all but the slice are rounding level. They entered the S36 stack.

Recorded measurement:

    bill -0.669% on both networks 35 and 100 (261,203,647,823 and 261,865,881,366 flops); T+C alone -0.441 / -0.438%; D -0.02% byte-identical; tail traces 943 Mflop per network at the 2x rate, birth update 850 Mflop; cond(Q[rows]) 4.6 and 6.2, core equal to the parent's to 5e-8; rounding parity in situ: tail traces <= 2.2e-6 relative, birth base 4.9e-8, plane delta 1.4e-6, feature matrix 7.9e-8, join core 4-5e-8

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/engineer/stack/build.py](code/opus_reexam_20261001_v1/engineer/stack/build.py)
- [code/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D.py](code/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D.py)
- [code/opus_reexam_20261001_v1/engineer/stack/parity_probe.py](code/opus_reexam_20261001_v1/engineer/stack/parity_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a10_bd32/fragment_bd32.py](code/opus_reexam_20261001_v1/engineer/a10_bd32/fragment_bd32.py)
- [code/opus_reexam_20261001_v1/engineer/a11_bare_views/analyze_binds.py](code/opus_reexam_20261001_v1/engineer/a11_bare_views/analyze_binds.py)
- [code/opus_reexam_20261001_v1/engineer/a11_bare_views/fragment_bvp.py](code/opus_reexam_20261001_v1/engineer/a11_bare_views/fragment_bvp.py)
- [code/opus_reexam_20261001_v1/engineer/a12_live_slice/fragment_dcr.py](code/opus_reexam_20261001_v1/engineer/a12_live_slice/fragment_dcr.py)
- [code/opus_reexam_20261001_v1/engineer/a17_tanh/fragment_ets.py](code/opus_reexam_20261001_v1/engineer/a17_tanh/fragment_ets.py)
- [code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3.py](code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3.py)
- [code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3o.py](code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3o.py)
- [code/opus_reexam_20261001_v1/engineer/a29_e2/fragment_e2_v1.py](code/opus_reexam_20261001_v1/engineer/a29_e2/fragment_e2_v1.py)

<details>
<summary>Additional programs and dependencies (23)</summary>

- [code/opus_reexam_20261001_v1/engineer/a31_star/fragment_sx_v1.py](code/opus_reexam_20261001_v1/engineer/a31_star/fragment_sx_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a32_c2k/fragment_c2k_v1.py](code/opus_reexam_20261001_v1/engineer/a32_c2k/fragment_c2k_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/fragment_ss_v3.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/fragment_ss_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a35_join/fragment_jr_v1.py](code/opus_reexam_20261001_v1/engineer/a35_join/fragment_jr_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a40_setup/fragment_cs_v1.py](code/opus_reexam_20261001_v1/engineer/a40_setup/fragment_cs_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bk3_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bk3_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bsk_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bsk_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_gsk_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_gsk_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v2.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v1.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v2.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a43_readout/fragment_ro_v1.py](code/opus_reexam_20261001_v1/engineer/a43_readout/fragment_ro_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a45_l15/fragment_l15_v1.py](code/opus_reexam_20261001_v1/engineer/a45_l15/fragment_l15_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a47_kc/fragment_kc_v1.py](code/opus_reexam_20261001_v1/engineer/a47_kc/fragment_kc_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py](code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py](code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a55_u15/fragment_u15_v1.py](code/opus_reexam_20261001_v1/engineer/a55_u15/fragment_u15_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqs_v1.py](code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqs_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a60_psdm/fragment_psdm_v1.py](code/opus_reexam_20261001_v1/engineer/a60_psdm/fragment_psdm_v1.py)
- [code/opus_reexam_20261001_v1/engineer/buildlib.py](code/opus_reexam_20261001_v1/engineer/buildlib.py)
- [code/opus_reexam_20261001_v1/engineer/edits_lib.py](code/opus_reexam_20261001_v1/engineer/edits_lib.py)
- [code/opus_reexam_20261001_v1/engineer/hzs/fragment_gen_v7.py](code/opus_reexam_20261001_v1/engineer/hzs/fragment_gen_v7.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L43](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L43)
- [TEAM_LOG_2026-10-01_to_10-22.md#L60](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L60)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333429: float32 arithmetic, half-zero strips, refreshed setup keys and separate tree levels](../../../submissions/333429/README.md)

[Research graph](../../../graph/index.html#node=2026-10-01-float32-reductions-live-row-operations-lower-charged-work-0) · [Copy provenance](../../../PROVENANCE.json)

