# Final-layer head-column hook corrected and smoke-tested: 25 added requests and 0.3 Mflop

2026-10-10T02:23Z · tool · informational · measured

The native final-layer learned correction is executed through the same encoder as the inner layers, so the new fourth-order input must enter its bias wrapper. The first implementation targeted the wrong function; the corrected hook ran with random weights and was explicitly a smoke test, not an accuracy result.

Recorded measurement:

    +25 requests, +0.3 Mflop on network 651 with random weights; smoke test only

## Programs

- [code/opus_reexam_20261001_v1/engineer/a42_head/debug_hcf.py](code/opus_reexam_20261001_v1/engineer/a42_head/debug_hcf.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/dump_columns.py](code/opus_reexam_20261001_v1/engineer/a42_head/dump_columns.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v1.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v2.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v1.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v2.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v3.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/judge_ten.py](code/opus_reexam_20261001_v1/engineer/a42_head/judge_ten.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/parity_hc.py](code/opus_reexam_20261001_v1/engineer/a42_head/parity_hc.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/run_dump.sh](code/opus_reexam_20261001_v1/engineer/a42_head/run_dump.sh)
- [code/opus_reexam_20261001_v1/engineer/a42_head/unit_hc.py](code/opus_reexam_20261001_v1/engineer/a42_head/unit_hc.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/unit_hc_v2.py](code/opus_reexam_20261001_v1/engineer/a42_head/unit_hc_v2.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/opus_reexam_20261001_v1/engineer/a42_head/unit_hcf.py](code/opus_reexam_20261001_v1/engineer/a42_head/unit_hcf.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/weights_from_trainer.py](code/opus_reexam_20261001_v1/engineer/a42_head/weights_from_trainer.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [code/opus_reexam_20261001_v1/engineer/quick.py](code/opus_reexam_20261001_v1/engineer/quick.py)
- [code/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv3t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv7c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@d6_r2_HC@dummy7_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__law_dials_d6.py](code/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv3t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv7c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@d6_r2_HC@dummy7_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__law_dials_d6.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L484](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L484)

<details>
<summary>All result files and records (45)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [HEAD_INPUT_C_DEFINITION.md](../../records/research/2026-10-02-left-factor-row-size-input-rejected-0-16-bill/HEAD_INPUT_C_DEFINITION.md) · result
- [columns_dummy7_651.log](columns_dummy7_651.log) · log
- [columns_dummy7_651_guard.json](columns_dummy7_651_guard.json) · result
- [b6dummy.json](b6dummy.json) · result
- [b7all.json](b7all.json) · result
- [b7c2.json](b7c2.json) · result
- [b7c3.json](b7c3.json) · result
- [b7cheap.json](b7cheap.json) · result
- [c17p.json](c17p.json) · result
- [dummy7.json](dummy7.json) · result
- [dummyf1.json](dummyf1.json) · result
- [dummyf3.json](dummyf3.json) · result
- [jmCnq.json](jmCnq.json) · result
- [jmCnq_f15.json](jmCnq_f15.json) · result
- [jmDh.json](jmDh.json) · result
- [jmDh_f15.json](jmDh_f15.json) · result
- [jmGh.json](jmGh.json) · result
- [jmGh_f15.json](jmGh_f15.json) · result
- [jmHh.json](jmHh.json) · result
- [jmHh_f15.json](jmHh_f15.json) · result
- [jmIfh.json](jmIfh.json) · result
- [jmIfh_f15.json](jmIfh_f15.json) · result
- [jmNdh.json](jmNdh.json) · result
- [jmNdh_f15.json](jmNdh_f15.json) · result
- [jmS50_v1nh.json](jmS50_v1nh.json) · result
- [jmS50_v1nh_f15.json](jmS50_v1nh_f15.json) · result
- [jmS51_v1h.json](jmS51_v1h.json) · result
- [jmS51_v1h_f15.json](jmS51_v1h_f15.json) · result
- [jmS53_v1h.json](jmS53_v1h.json) · result
- [jmS53_v1h_f15.json](jmS53_v1h_f15.json) · result
- [jmS54_v1h.json](jmS54_v1h.json) · result
- [jmS54_v1h_f15.json](jmS54_v1h_f15.json) · result
- [jmS56_v1h.json](jmS56_v1h.json) · result
- [jmS56_v1h_f15.json](jmS56_v1h_f15.json) · result
- [jmS56abA1h.json](jmS56abA1h.json) · result
- [jmS56abA1h_f15.json](jmS56abA1h_f15.json) · result
- [jmS56abA2h.json](jmS56abA2h.json) · result
- [jmS56abA2h_f15.json](jmS56abA2h_f15.json) · result
- [s1.json](s1.json) · result
- [s2.json](s2.json) · result
- [s2all.json](s2all.json) · result
- [s2v2.json](s2v2.json) · result
- [t45z.json](t45z.json) · result
- [t45z_f15.json](t45z_f15.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-10-10-final-layer-head-column-hook-corrected-smoke-tested-25) · [Copy provenance](../../PROVENANCE.json)

