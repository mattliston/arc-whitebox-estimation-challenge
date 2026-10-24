# Store dial sweep of 39 zero-bill arms (gate, skew, kurtosis, birth, plane scales, late grows): nothing beyond noise

2026-10-16T21:50:38Z · experiment · rejected · measured

Twelve zero-bill dial arms and later late-layer grows were run on the 32 store networks on the estimator submitted as #336412 (S2tr) point (gate transport GK4 0.4/0.8/1.0/1.2, skew and kurtosis scales, birth-fix 0.119, SS beta, NP strength, pe34, k4l34, learned-edge scale, dead cut and floor variants, old-tier tail at layers 13-14, read-prefix grows). Early reads such as the GK4 1.0 revert (-0.375% on 7 ranks) faded to noise as ranks accumulated; the pooled table of 39 arms held no MSE-side arm beyond noise, so the remaining gap was bill-only and the sweep was stopped as a 0.2%-class exercise at the user request.

Recorded measurement:

    n32: ssb_085 xnet -0.062 (SE .07, 19/32) null; gk4_04 +0.008 null; ss0 +0.11 worse; uvp_320 dead knob (L15 r_u hard-coded 256). ott30_1314 n31 +0.18 FAIL; gk4_08 n17 -0.11 (fading); dc_325 n13 +0.02; gne_12_1012 n23 0.00; dc40_1314 n24 +0.01; dcv_0015 n15 -0.05. gk4_10 (GK4 1.0): n7 xMSE -0.375% (SE .092, 6/7 lower) -> n18 -0.07 (0.08); gk4_12 n8 -0.08; bfix119 n12 -0.23 (0.17).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_solo_v1/sweep_feeder.py](code/opus_solo_v1/sweep_feeder.py)
- [code/opus_kinkoracle_v1/cand_decomp_gk4_10_v1.py](code/opus_kinkoracle_v1/cand_decomp_gk4_10_v1.py)
- [code/opus_kinkoracle_v1/cand_decomp_gsk_09_v1.py](code/opus_kinkoracle_v1/cand_decomp_gsk_09_v1.py)
- [code/opus_kinkoracle_v1/cand_decomp_bsk_030_v1.py](code/opus_kinkoracle_v1/cand_decomp_bsk_030_v1.py)
- [code/opus_kinkoracle_v1/cand_s2tr_kinkoracle_v2.py](code/opus_kinkoracle_v1/cand_s2tr_kinkoracle_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4491](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4491)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4498](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4498)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4502](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4502)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4506](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4506)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4507](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4507)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4511](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4511)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4513](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4513)

<details>
<summary>All result files and records (2)</summary>

- [SWEEP_RESULTS.md](SWEEP_RESULTS.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Four fixed coefficients for gate, birth, skew and star terms improve 32-network raw error by 0.295%](../four-fixed-coefficients-gate-birth-skew-star-terms-improve/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-store-dial-sweep-39-zero-bill-arms-gate-skew) · [Copy provenance](../../../PROVENANCE.json)

