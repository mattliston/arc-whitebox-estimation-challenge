# Native tail-repair capture and projection helpers pass seven exact normalization fixtures

2026-10-11T00:44:54Z · tool · informational · measured

Source and vector fixtures certified disabled and zero-strength identity, exact diagonal preservation and signed-overlap handling for the tail-repair helper. This certificate covered the helper only; full-network export binding, gradients and release-cost checks remained unrun.

Recorded measurement:

    seven normalization cases maxabs0; Projection526336 FLOPs/3requests; capture30654/43 includes original stub diagonal arithmetic, so not a full-network delta

## Programs

- [code/opus_reexam_20261001_v1/engineer/a59_repair_heads/native_vector_fixture_v2.py](code/opus_reexam_20261001_v1/engineer/a59_repair_heads/native_vector_fixture_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_repair_heads_v1.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_repair_heads_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/engineer/a59_repair_heads/fixtures_v2/native_disabled_source.py](code/opus_reexam_20261001_v1/engineer/a59_repair_heads/fixtures_v2/native_disabled_source.py)
- [code/opus_reexam_20261001_v1/engineer/a59_repair_heads/fixtures_v2/native_nonzero_source.py](code/opus_reexam_20261001_v1/engineer/a59_repair_heads/fixtures_v2/native_nonzero_source.py)
- [code/opus_reexam_20261001_v1/engineer/a59_repair_heads/fixtures_v2/native_zero_source.py](code/opus_reexam_20261001_v1/engineer/a59_repair_heads/fixtures_v2/native_zero_source.py)
- [code/opus_reexam_20261001_v1/engineer/a59_repair_heads/fragment_repair_heads_v2.py](code/opus_reexam_20261001_v1/engineer/a59_repair_heads/fragment_repair_heads_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a59_repair_heads/repair_head_build_v2.py](code/opus_reexam_20261001_v1/engineer/a59_repair_heads/repair_head_build_v2.py)
- [code/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@jmIf_r2_HC@jmIfh_BT@kef1e7_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__jmIf.py](code/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@jmIf_r2_HC@jmIfh_BT@kef1e7_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__jmIf.py)

<details>
<summary>Additional programs and dependencies (8)</summary>

- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/t87_torch_repair_heads_v1.py](code/opus_reexam_20261001_v1/physicist/t87_torch_repair_heads_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L866](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L866)
- [TEAM_LOG_2026-10-01_to_10-22.md#L868](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L868)
- [TEAM_LOG_2026-10-01_to_10-22.md#L871](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L871)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SOURCE_BUILD_RECEIPT_v2.json](SOURCE_BUILD_RECEIPT_v2.json) · result
- [SOURCE_AND_VECTOR_GATES_v1.json](SOURCE_AND_VECTOR_GATES_v1.json) · result
- [T87_PREPARATION_COMPLETE_v1.json](T87_PREPARATION_COMPLETE_v1.json) · result
- [t87_vector_native_receipt_v2.json](t87_vector_native_receipt_v2.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-10-11-native-tail-repair-capture-projection-helpers-pass-seven-exact) · [Copy provenance](../../PROVENANCE.json)

