# True-inner-size covariance tiles rejected: -0.433% FLOPs costs 6,537 additional first-network requests

2026-10-10T03:26Z · experiment · rejected · measured

For the covariance-sandwich dead cut at gate floor 1e-3 three kernels for the cut product were priced on network 651 against the skeleton: the generic symmetric path (-0.186% metered FLOPs, -1,223 requests), the padded specialised 1,024 product (-0.316% metered FLOPs, -18 requests, request-neutral) and tiles at the inner size (-0.433% metered FLOPs but +6,537 requests on the first network from lazy binds of the new tree keys, viable only with those keys bound at setup). Floors above 1e-3 were probed too: 2e-3 adds one cut layer (L12 960 -> 896, C_pre error 1.9e-4) and 5e-3 cuts layers 7-14 at errors 3-8e-4, so 1e-3 stayed.

Recorded measurement:

    generic -0.186% / -1,223 requests; padded -0.316% / -18 requests, prediction within the rounding draw (-0.18%); tiles -0.433% bill, +6,537 first-network requests; floor 2e-3 C_pre error 1.9e-4; 5e-3 errors 3-8e-4

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/a44_cov/fragment_dcv_v3.py](code/a44_cov/fragment_dcv_v3.py)
- [code/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@d6_r2_HC@s2_BT@bdcv3p_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__law_dials_d6.py](code/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@d6_r2_HC@s2_BT@bdcv3p_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__law_dials_d6.py)
- [code/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@d6_r2_HC@s2_BT@bdcv3t_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__law_dials_d6.py](code/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@d6_r2_HC@s2_BT@bdcv3t_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__law_dials_d6.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L536](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L536)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-true-inner-size-covariance-tiles-rejected-0-433-flops) · [Copy provenance](../../../PROVENANCE.json)

