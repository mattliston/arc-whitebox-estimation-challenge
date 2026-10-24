# Pair-mean rank 320 rejected natively: -0.92% MSE, +1.26% FLOPs and +0.33% score

2026-10-11T17:35Z · experiment · rejected · measured

The whole gain of an exact pair-mean read sits in coordinates 256-320 in PyTorch, but natively pair-mean read rank (r_u) 320 costs +1.26% of metered FLOPs for -0.92% MSE on 8 gate networks and failed its pre-declared < -0.3% bar; r_u 272 is an unaligned width (+4.64% metered FLOPs).

Recorded measurement:

    torch 320 -0.89% (se 0.26, 13 of 16), 288 -0.60%, 384 = 512 bit-identical; native 8 networks MSE -0.92%, bill +1.26%, score +0.33% (se 0.43), +429 ops

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engineer/a22_cutsim/n6y_cutsim_v2.py](code/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [code/engineer/a22_cutsim/pbase_cutsim_v1.py](code/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [code/engineer/a22_cutsim/torch_cutsim_s46_v11.py](code/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [code/engineer/a22_cutsim/torch_push_native_v2.py](code/engineer/a22_cutsim/torch_push_native_v2.py)
- [code/engineer/a22_cutsim/torch_rowcuts_v15.py](code/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [code/engineer/a62_modules/cand_s47e_lu272.py](code/engineer/a62_modules/cand_s47e_lu272.py)
- [code/engineer/a62_modules/cand_s47e_lu288.py](code/engineer/a62_modules/cand_s47e_lu288.py)
- [code/engineer/a62_modules/cand_s47e_lu320.py](code/engineer/a62_modules/cand_s47e_lu320.py)
- [code/engineer/a62_modules/cand_s47e_lu384.py](code/engineer/a62_modules/cand_s47e_lu384.py)
- [code/physicist/ncar_bank_frozen_s2s.py](code/physicist/ncar_bank_frozen_s2s.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1116](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1116)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1120](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1120)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1122](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1122)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1125](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1125)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1133](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1133)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1134](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1134)

<details>
<summary>All result files and records (229)</summary>

- [allowed_seeds.txt](allowed_seeds.txt) · result
- [t105_7100136_RU272.w0.json](t105_7100136_RU272.w0.json) · result
- [t105_7100136_RU288.w1.json](t105_7100136_RU288.w1.json) · result
- [t105_7100136_RU320.w0.json](t105_7100136_RU320.w0.json) · result
- [t105_7100136_RU320SS16.w2.json](t105_7100136_RU320SS16.w2.json) · result
- [t105_7100136_RU320SS8.w0.json](t105_7100136_RU320SS8.w0.json) · result
- [t105_7100136_RU384.w1.json](t105_7100136_RU384.w1.json) · result
- [t105_7100136_RU512.w2.json](t105_7100136_RU512.w2.json) · result
- [t105_7100137_RU272.w2.json](t105_7100137_RU272.w2.json) · result
- [t105_7100137_RU288.w0.json](t105_7100137_RU288.w0.json) · result
- [t105_7100137_RU320.w0.json](t105_7100137_RU320.w0.json) · result
- [t105_7100137_RU320SS16.w1.json](t105_7100137_RU320SS16.w1.json) · result
- [t105_7100137_RU320SS8.w2.json](t105_7100137_RU320SS8.w2.json) · result
- [t105_7100137_RU384.w1.json](t105_7100137_RU384.w1.json) · result
- [t105_7100137_RU512.w2.json](t105_7100137_RU512.w2.json) · result
- [t105_7100138_RU272.w1.json](t105_7100138_RU272.w1.json) · result
- [t105_7100138_RU288.w2.json](t105_7100138_RU288.w2.json) · result
- [t105_7100138_RU320.w0.json](t105_7100138_RU320.w0.json) · result
- [t105_7100138_RU320SS16.w0.json](t105_7100138_RU320SS16.w0.json) · result
- [t105_7100138_RU320SS8.w1.json](t105_7100138_RU320SS8.w1.json) · result
- [t105_7100138_RU384.w1.json](t105_7100138_RU384.w1.json) · result
- [t105_7100138_RU512.w2.json](t105_7100138_RU512.w2.json) · result
- [t105_7100139_RU272.w0.json](t105_7100139_RU272.w0.json) · result
- [t105_7100139_RU288.w1.json](t105_7100139_RU288.w1.json) · result
- [t105_7100139_RU320.w0.json](t105_7100139_RU320.w0.json) · result
- [t105_7100139_RU320SS16.w2.json](t105_7100139_RU320SS16.w2.json) · result
- [t105_7100139_RU320SS8.w0.json](t105_7100139_RU320SS8.w0.json) · result
- [t105_7100139_RU384.w1.json](t105_7100139_RU384.w1.json) · result
- [t105_7100139_RU512.w2.json](t105_7100139_RU512.w2.json) · result
- [t105_7100140_RU272.w2.json](t105_7100140_RU272.w2.json) · result
- [t105_7100140_RU288.w0.json](t105_7100140_RU288.w0.json) · result
- [t105_7100140_RU320.w0.json](t105_7100140_RU320.w0.json) · result
- [t105_7100140_RU320SS16.w1.json](t105_7100140_RU320SS16.w1.json) · result
- [t105_7100140_RU320SS8.w2.json](t105_7100140_RU320SS8.w2.json) · result
- [t105_7100140_RU384.w1.json](t105_7100140_RU384.w1.json) · result
- [t105_7100140_RU512.w2.json](t105_7100140_RU512.w2.json) · result
- [t105_7100141_RU272.w1.json](t105_7100141_RU272.w1.json) · result
- [t105_7100141_RU288.w2.json](t105_7100141_RU288.w2.json) · result
- [t105_7100141_RU320.w0.json](t105_7100141_RU320.w0.json) · result
- [t105_7100141_RU320SS16.w0.json](t105_7100141_RU320SS16.w0.json) · result
- [t105_7100141_RU320SS8.w1.json](t105_7100141_RU320SS8.w1.json) · result
- [t105_7100141_RU384.w1.json](t105_7100141_RU384.w1.json) · result
- [t105_7100141_RU512.w2.json](t105_7100141_RU512.w2.json) · result
- [t105_7100142_RU272.w0.json](t105_7100142_RU272.w0.json) · result
- [t105_7100142_RU288.w1.json](t105_7100142_RU288.w1.json) · result
- [t105_7100142_RU320.w0.json](t105_7100142_RU320.w0.json) · result
- [t105_7100142_RU320SS16.w2.json](t105_7100142_RU320SS16.w2.json) · result
- [t105_7100142_RU320SS8.w0.json](t105_7100142_RU320SS8.w0.json) · result
- [t105_7100142_RU384.w1.json](t105_7100142_RU384.w1.json) · result
- [t105_7100142_RU512.w2.json](t105_7100142_RU512.w2.json) · result
- [t105_7100143_RU272.w2.json](t105_7100143_RU272.w2.json) · result
- [t105_7100143_RU288.w0.json](t105_7100143_RU288.w0.json) · result
- [t105_7100143_RU320.w0.json](t105_7100143_RU320.w0.json) · result
- [t105_7100143_RU320SS16.w1.json](t105_7100143_RU320SS16.w1.json) · result
- [t105_7100143_RU320SS8.w2.json](t105_7100143_RU320SS8.w2.json) · result
- [t105_7100143_RU384.w1.json](t105_7100143_RU384.w1.json) · result
- [t105_7100143_RU512.w2.json](t105_7100143_RU512.w2.json) · result
- [t105_7100144_RU272.w1.json](t105_7100144_RU272.w1.json) · result
- [t105_7100144_RU288.w2.json](t105_7100144_RU288.w2.json) · result
- [t105_7100144_RU320.w0.json](t105_7100144_RU320.w0.json) · result
- [t105_7100144_RU320SS16.w0.json](t105_7100144_RU320SS16.w0.json) · result
- [t105_7100144_RU320SS8.w1.json](t105_7100144_RU320SS8.w1.json) · result
- [t105_7100144_RU384.w1.json](t105_7100144_RU384.w1.json) · result
- [t105_7100144_RU512.w2.json](t105_7100144_RU512.w2.json) · result
- [t105_7100145_RU272.w0.json](t105_7100145_RU272.w0.json) · result
- [t105_7100145_RU288.w1.json](t105_7100145_RU288.w1.json) · result
- [t105_7100145_RU320.w0.json](t105_7100145_RU320.w0.json) · result
- [t105_7100145_RU320SS16.w2.json](t105_7100145_RU320SS16.w2.json) · result
- [t105_7100145_RU320SS8.w0.json](t105_7100145_RU320SS8.w0.json) · result
- [t105_7100145_RU384.w1.json](t105_7100145_RU384.w1.json) · result
- [t105_7100145_RU512.w2.json](t105_7100145_RU512.w2.json) · result
- [t105_7100146_RU272.w2.json](t105_7100146_RU272.w2.json) · result
- [t105_7100146_RU288.w0.json](t105_7100146_RU288.w0.json) · result
- [t105_7100146_RU320.w0.json](t105_7100146_RU320.w0.json) · result
- [t105_7100146_RU320SS16.w1.json](t105_7100146_RU320SS16.w1.json) · result
- [t105_7100146_RU320SS8.w2.json](t105_7100146_RU320SS8.w2.json) · result
- [t105_7100146_RU384.w1.json](t105_7100146_RU384.w1.json) · result
- [t105_7100146_RU512.w2.json](t105_7100146_RU512.w2.json) · result
- [t105_7100147_RU272.w1.json](t105_7100147_RU272.w1.json) · result
- [t105_7100147_RU288.w2.json](t105_7100147_RU288.w2.json) · result
- [t105_7100147_RU320.w0.json](t105_7100147_RU320.w0.json) · result
- [t105_7100147_RU320SS16.w0.json](t105_7100147_RU320SS16.w0.json) · result
- [t105_7100147_RU320SS8.w1.json](t105_7100147_RU320SS8.w1.json) · result
- [t105_7100147_RU384.w1.json](t105_7100147_RU384.w1.json) · result
- [t105_7100147_RU512.w2.json](t105_7100147_RU512.w2.json) · result
- [t105_7100148_RU272.w0.json](t105_7100148_RU272.w0.json) · result
- [t105_7100148_RU288.w1.json](t105_7100148_RU288.w1.json) · result
- [t105_7100148_RU320.w0.json](t105_7100148_RU320.w0.json) · result
- [t105_7100148_RU320SS16.w2.json](t105_7100148_RU320SS16.w2.json) · result
- [t105_7100148_RU320SS8.w0.json](t105_7100148_RU320SS8.w0.json) · result
- [t105_7100148_RU384.w1.json](t105_7100148_RU384.w1.json) · result
- [t105_7100148_RU512.w2.json](t105_7100148_RU512.w2.json) · result
- [t105_7100149_RU272.w2.json](t105_7100149_RU272.w2.json) · result
- [t105_7100149_RU288.w0.json](t105_7100149_RU288.w0.json) · result
- [t105_7100149_RU320.w0.json](t105_7100149_RU320.w0.json) · result
- [t105_7100149_RU320SS16.w1.json](t105_7100149_RU320SS16.w1.json) · result
- [t105_7100149_RU320SS8.w2.json](t105_7100149_RU320SS8.w2.json) · result
- [t105_7100149_RU384.w1.json](t105_7100149_RU384.w1.json) · result
- [t105_7100149_RU512.w2.json](t105_7100149_RU512.w2.json) · result
- [t105_7100150_RU272.w1.json](t105_7100150_RU272.w1.json) · result
- [t105_7100150_RU288.w2.json](t105_7100150_RU288.w2.json) · result
- [t105_7100150_RU320.w0.json](t105_7100150_RU320.w0.json) · result
- [t105_7100150_RU320SS16.w0.json](t105_7100150_RU320SS16.w0.json) · result
- [t105_7100150_RU320SS8.w1.json](t105_7100150_RU320SS8.w1.json) · result
- [t105_7100150_RU384.w1.json](t105_7100150_RU384.w1.json) · result
- [t105_7100150_RU512.w2.json](t105_7100150_RU512.w2.json) · result
- [t105_7100151_RU272.w0.json](t105_7100151_RU272.w0.json) · result
- [t105_7100151_RU288.w1.json](t105_7100151_RU288.w1.json) · result
- [t105_7100151_RU320.w0.json](t105_7100151_RU320.w0.json) · result
- [t105_7100151_RU320SS16.w2.json](t105_7100151_RU320SS16.w2.json) · result
- [t105_7100151_RU320SS8.w0.json](t105_7100151_RU320SS8.w0.json) · result
- [t105_7100151_RU384.w1.json](t105_7100151_RU384.w1.json) · result
- [t105_7100151_RU512.w2.json](t105_7100151_RU512.w2.json) · result
- [t105_7100136_RU272.json](t105_7100136_RU272.json) · result
- [t105_7100136_RU288.json](t105_7100136_RU288.json) · result
- [t105_7100136_RU320.json](t105_7100136_RU320.json) · result
- [t105_7100136_RU320SS16.json](t105_7100136_RU320SS16.json) · result
- [t105_7100136_RU320SS8.json](t105_7100136_RU320SS8.json) · result
- [t105_7100136_RU384.json](t105_7100136_RU384.json) · result
- [t105_7100136_RU512.json](t105_7100136_RU512.json) · result
- [t105_7100137_RU272.json](t105_7100137_RU272.json) · result
- [t105_7100137_RU288.json](t105_7100137_RU288.json) · result
- [t105_7100137_RU320.json](t105_7100137_RU320.json) · result
- [t105_7100137_RU320SS16.json](t105_7100137_RU320SS16.json) · result
- [t105_7100137_RU320SS8.json](t105_7100137_RU320SS8.json) · result
- [t105_7100137_RU384.json](t105_7100137_RU384.json) · result
- [t105_7100137_RU512.json](t105_7100137_RU512.json) · result
- [t105_7100138_RU272.json](t105_7100138_RU272.json) · result
- [t105_7100138_RU288.json](t105_7100138_RU288.json) · result
- [t105_7100138_RU320.json](t105_7100138_RU320.json) · result
- [t105_7100138_RU320SS16.json](t105_7100138_RU320SS16.json) · result
- [t105_7100138_RU320SS8.json](t105_7100138_RU320SS8.json) · result
- [t105_7100138_RU384.json](t105_7100138_RU384.json) · result
- [t105_7100138_RU512.json](t105_7100138_RU512.json) · result
- [t105_7100139_RU272.json](t105_7100139_RU272.json) · result
- [t105_7100139_RU288.json](t105_7100139_RU288.json) · result
- [t105_7100139_RU320.json](t105_7100139_RU320.json) · result
- [t105_7100139_RU320SS16.json](t105_7100139_RU320SS16.json) · result
- [t105_7100139_RU320SS8.json](t105_7100139_RU320SS8.json) · result
- [t105_7100139_RU384.json](t105_7100139_RU384.json) · result
- [t105_7100139_RU512.json](t105_7100139_RU512.json) · result
- [t105_7100140_RU272.json](t105_7100140_RU272.json) · result
- [t105_7100140_RU288.json](t105_7100140_RU288.json) · result
- [t105_7100140_RU320.json](t105_7100140_RU320.json) · result
- [t105_7100140_RU320SS16.json](t105_7100140_RU320SS16.json) · result
- [t105_7100140_RU320SS8.json](t105_7100140_RU320SS8.json) · result
- [t105_7100140_RU384.json](t105_7100140_RU384.json) · result
- [t105_7100140_RU512.json](t105_7100140_RU512.json) · result
- [t105_7100141_RU272.json](t105_7100141_RU272.json) · result
- [t105_7100141_RU288.json](t105_7100141_RU288.json) · result
- [t105_7100141_RU320.json](t105_7100141_RU320.json) · result
- [t105_7100141_RU320SS16.json](t105_7100141_RU320SS16.json) · result
- [t105_7100141_RU320SS8.json](t105_7100141_RU320SS8.json) · result
- [t105_7100141_RU384.json](t105_7100141_RU384.json) · result
- [t105_7100141_RU512.json](t105_7100141_RU512.json) · result
- [t105_7100142_RU272.json](t105_7100142_RU272.json) · result
- [t105_7100142_RU288.json](t105_7100142_RU288.json) · result
- [t105_7100142_RU320.json](t105_7100142_RU320.json) · result
- [t105_7100142_RU320SS16.json](t105_7100142_RU320SS16.json) · result
- [t105_7100142_RU320SS8.json](t105_7100142_RU320SS8.json) · result
- [t105_7100142_RU384.json](t105_7100142_RU384.json) · result
- [t105_7100142_RU512.json](t105_7100142_RU512.json) · result
- [t105_7100143_RU272.json](t105_7100143_RU272.json) · result
- [t105_7100143_RU288.json](t105_7100143_RU288.json) · result
- [t105_7100143_RU320.json](t105_7100143_RU320.json) · result
- [t105_7100143_RU320SS16.json](t105_7100143_RU320SS16.json) · result
- [t105_7100143_RU320SS8.json](t105_7100143_RU320SS8.json) · result
- [t105_7100143_RU384.json](t105_7100143_RU384.json) · result
- [t105_7100143_RU512.json](t105_7100143_RU512.json) · result
- [t105_7100144_RU272.json](t105_7100144_RU272.json) · result
- [t105_7100144_RU288.json](t105_7100144_RU288.json) · result
- [t105_7100144_RU320.json](t105_7100144_RU320.json) · result
- [t105_7100144_RU320SS16.json](t105_7100144_RU320SS16.json) · result
- [t105_7100144_RU320SS8.json](t105_7100144_RU320SS8.json) · result
- [t105_7100144_RU384.json](t105_7100144_RU384.json) · result
- [t105_7100144_RU512.json](t105_7100144_RU512.json) · result
- [t105_7100145_RU272.json](t105_7100145_RU272.json) · result
- [t105_7100145_RU288.json](t105_7100145_RU288.json) · result
- [t105_7100145_RU320.json](t105_7100145_RU320.json) · result
- [t105_7100145_RU320SS16.json](t105_7100145_RU320SS16.json) · result
- [t105_7100145_RU320SS8.json](t105_7100145_RU320SS8.json) · result
- [t105_7100145_RU384.json](t105_7100145_RU384.json) · result
- [t105_7100145_RU512.json](t105_7100145_RU512.json) · result
- [t105_7100146_RU272.json](t105_7100146_RU272.json) · result
- [t105_7100146_RU288.json](t105_7100146_RU288.json) · result
- [t105_7100146_RU320.json](t105_7100146_RU320.json) · result
- [t105_7100146_RU320SS16.json](t105_7100146_RU320SS16.json) · result
- [t105_7100146_RU320SS8.json](t105_7100146_RU320SS8.json) · result
- [t105_7100146_RU384.json](t105_7100146_RU384.json) · result
- [t105_7100146_RU512.json](t105_7100146_RU512.json) · result
- [t105_7100147_RU272.json](t105_7100147_RU272.json) · result
- [t105_7100147_RU288.json](t105_7100147_RU288.json) · result
- [t105_7100147_RU320.json](t105_7100147_RU320.json) · result
- [t105_7100147_RU320SS16.json](t105_7100147_RU320SS16.json) · result
- [t105_7100147_RU320SS8.json](t105_7100147_RU320SS8.json) · result
- [t105_7100147_RU384.json](t105_7100147_RU384.json) · result
- [t105_7100147_RU512.json](t105_7100147_RU512.json) · result
- [t105_7100148_RU272.json](t105_7100148_RU272.json) · result
- [t105_7100148_RU288.json](t105_7100148_RU288.json) · result
- [t105_7100148_RU320.json](t105_7100148_RU320.json) · result
- [t105_7100148_RU320SS16.json](t105_7100148_RU320SS16.json) · result
- [t105_7100148_RU320SS8.json](t105_7100148_RU320SS8.json) · result
- [t105_7100148_RU384.json](t105_7100148_RU384.json) · result
- [t105_7100148_RU512.json](t105_7100148_RU512.json) · result
- [t105_7100149_RU272.json](t105_7100149_RU272.json) · result
- [t105_7100149_RU288.json](t105_7100149_RU288.json) · result
- [t105_7100149_RU320.json](t105_7100149_RU320.json) · result
- [t105_7100149_RU320SS16.json](t105_7100149_RU320SS16.json) · result
- [t105_7100149_RU320SS8.json](t105_7100149_RU320SS8.json) · result
- [t105_7100149_RU384.json](t105_7100149_RU384.json) · result
- [t105_7100149_RU512.json](t105_7100149_RU512.json) · result
- [t105_7100150_RU272.json](t105_7100150_RU272.json) · result
- [t105_7100150_RU288.json](t105_7100150_RU288.json) · result
- [t105_7100150_RU320.json](t105_7100150_RU320.json) · result
- [t105_7100150_RU320SS16.json](t105_7100150_RU320SS16.json) · result
- [t105_7100150_RU320SS8.json](t105_7100150_RU320SS8.json) · result
- [t105_7100150_RU384.json](t105_7100150_RU384.json) · result
- [t105_7100150_RU512.json](t105_7100150_RU512.json) · result
- [t105_7100151_RU272.json](t105_7100151_RU272.json) · result
- [t105_7100151_RU288.json](t105_7100151_RU288.json) · result
- [t105_7100151_RU320.json](t105_7100151_RU320.json) · result
- [t105_7100151_RU320SS16.json](t105_7100151_RU320SS16.json) · result
- [t105_7100151_RU320SS8.json](t105_7100151_RU320SS8.json) · result
- [t105_7100151_RU384.json](t105_7100151_RU384.json) · result
- [t105_7100151_RU512.json](t105_7100151_RU512.json) · result
- [worker_w0.log](worker_w0.log) · log
- [worker_w1.log](worker_w1.log) · log
- [worker_w2.log](worker_w2.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Three native approximations explain 91% of the native-versus-research error gap](../../../records/research/2026-10-10-three-native-approximations-explain-91-native-versus-research-error/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-pair-mean-rank-320-rejected-natively-0-92-mse) · [Copy provenance](../../../PROVENANCE.json)

