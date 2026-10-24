# Zero strip-push gain floor rejected: -0.051% FLOPs costs +7.5%/+2.9% kernel CPU

2026-10-11T07:50Z · experiment · rejected · measured

In the submitted candidate generation 47 (S47) configuration (release fragments rebuilt in the engineer's lane, byte for byte 55b951a6) the seven-level strip-push rule with its gain floor at 0 instead of seven-level full-packet strip push (HZ7R)'s 3% was measured on networks 651 / 378: metered FLOPs -0.075% / -0.026% (pooled -0.051%) with local kernel CPU +7.5% / +2.9% (50.0 -> 53.8 s, 49.8 -> 51.2 s) and, once the persistent covariance-sandwich scratch (XRQS) scratch in the same file is subtracted, about +1,000 / +320 charged operations; the final-layer MSE moved -0.16% / -0.06% at rounding level. With #335340 already at 77.6 s mean kernel time and 93.1 s max wall, +3..+8% of kernel time for -0.05% of metered FLOPs was judged a bad trade and the 3% floor stayed.

Recorded measurement:

    bill -0.075% / -0.026% (pooled -0.051%); kernel CPU +7.5% / +2.9% (50.0 -> 53.8 s, 49.8 -> 51.2 s); zero floor alone +1,000 / +320 charged ops for -0.072% / -0.023%; MSE -0.16% / -0.06% rounding level; #335340 kernel 77.6 s mean (S46p 70.2 s), wall max 93.1 s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engineer/stack/rl/s47s_safe/candidate.py](code/engineer/stack/rl/s47s_safe/candidate.py)
- [code/engineer/stack/rl/s47r0s_safe/candidate.py](code/engineer/stack/rl/s47r0s_safe/candidate.py)
- [code/engineer/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@jmNd_r2_HC@jmNdh_BT@kbef3e_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__jmNd.py](code/engineer/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@jmNd_r2_HC@jmNdh_BT@kbef3e_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__jmNd.py)
- [code/engineer/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@jmNd_r2_HC@jmNdh_BT@kdeb4d_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__jmNd.py](code/engineer/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@jmNd_r2_HC@jmNdh_BT@kdeb4d_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__jmNd.py)
- [code/release/variants/fragments/l3_bind128.py](code/release/variants/fragments/l3_bind128.py)
- [code/release/variants/fragments/lvfs_limit_off.py](code/release/variants/fragments/lvfs_limit_off.py)
- [code/release/variants/fragments/prices_zero.py](code/release/variants/fragments/prices_zero.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1043](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1043)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1044](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1044)

<details>
<summary>All result files and records (3)</summary>

- [build.json](build.json) · result
- [build-d11b193d.json](build-d11b193d.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-zero-strip-push-gain-floor-rejected-0-051-flops) · [Copy provenance](../../../PROVENANCE.json)

