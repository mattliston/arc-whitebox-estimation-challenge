# Exact final-layer row and push dead cuts rejected: MSE changes 0.00%/-0.01% and +0.02%

2026-10-11T17:40:14Z · experiment · rejected · measured

On 16 seeds of the S47e moment-closure estimator at the 32-network ridge-re-sized cut-simulated refit (Nd) point under the v11 cut simulator (the PyTorch moment-closure estimator with the native cuts reproduced), restoring the final-layer row cut to exact (l15 99) or loosening it to 4.0 changed final error by 0.00% / -0.01% (se 0.01) and making the push dead cut exact (dead 99) changed it +0.02% (se 0.12). The cuts cost nothing in error, so un-cutting them (the audit's exact-l15 metered FLOPs +0.98%) would be pure metered FLOPs loss.

Recorded measurement:

    l15 99 and 4.0 = 0.00% / -0.01% (se 0.01); dead 99 +0.02% (se 0.12); the audit's exact-l15 bill +0.98%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engineer/a22_cutsim/n6y_cutsim_v2.py](code/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [code/engineer/a22_cutsim/pbase_cutsim_v1.py](code/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [code/engineer/a22_cutsim/torch_cutsim_s46_v11.py](code/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [code/engineer/a22_cutsim/torch_push_native_v2.py](code/engineer/a22_cutsim/torch_push_native_v2.py)
- [code/engineer/a22_cutsim/torch_rowcuts_v15.py](code/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [code/physicist/ncar_bank_frozen_s2s.py](code/physicist/ncar_bank_frozen_s2s.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1135](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1135)

<details>
<summary>All result files and records (100)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [t106_7100136_DEADX.w2.json](t106_7100136_DEADX.w2.json) · record
- [t106_7100136_L15X.w0.json](t106_7100136_L15X.w0.json) · record
- [t106_7100136_L15_40.w1.json](t106_7100136_L15_40.w1.json) · record
- [t106_7100137_DEADX.w2.json](t106_7100137_DEADX.w2.json) · record
- [t106_7100137_L15X.w0.json](t106_7100137_L15X.w0.json) · record
- [t106_7100137_L15_40.w1.json](t106_7100137_L15_40.w1.json) · record
- [t106_7100138_DEADX.w2.json](t106_7100138_DEADX.w2.json) · record
- [t106_7100138_L15X.w0.json](t106_7100138_L15X.w0.json) · record
- [t106_7100138_L15_40.w1.json](t106_7100138_L15_40.w1.json) · record
- [t106_7100139_DEADX.w2.json](t106_7100139_DEADX.w2.json) · record
- [t106_7100139_L15X.w0.json](t106_7100139_L15X.w0.json) · record
- [t106_7100139_L15_40.w1.json](t106_7100139_L15_40.w1.json) · record
- [t106_7100140_DEADX.w2.json](t106_7100140_DEADX.w2.json) · record
- [t106_7100140_L15X.w0.json](t106_7100140_L15X.w0.json) · record
- [t106_7100140_L15_40.w1.json](t106_7100140_L15_40.w1.json) · record
- [t106_7100141_DEADX.w2.json](t106_7100141_DEADX.w2.json) · record
- [t106_7100141_L15X.w0.json](t106_7100141_L15X.w0.json) · record
- [t106_7100141_L15_40.w1.json](t106_7100141_L15_40.w1.json) · record
- [t106_7100142_DEADX.w2.json](t106_7100142_DEADX.w2.json) · record
- [t106_7100142_L15X.w0.json](t106_7100142_L15X.w0.json) · record
- [t106_7100142_L15_40.w1.json](t106_7100142_L15_40.w1.json) · record
- [t106_7100143_DEADX.w2.json](t106_7100143_DEADX.w2.json) · record
- [t106_7100143_L15X.w0.json](t106_7100143_L15X.w0.json) · record
- [t106_7100143_L15_40.w1.json](t106_7100143_L15_40.w1.json) · record
- [t106_7100144_DEADX.w2.json](t106_7100144_DEADX.w2.json) · record
- [t106_7100144_L15X.w0.json](t106_7100144_L15X.w0.json) · record
- [t106_7100144_L15_40.w1.json](t106_7100144_L15_40.w1.json) · record
- [t106_7100145_DEADX.w2.json](t106_7100145_DEADX.w2.json) · record
- [t106_7100145_L15X.w0.json](t106_7100145_L15X.w0.json) · record
- [t106_7100145_L15_40.w1.json](t106_7100145_L15_40.w1.json) · record
- [t106_7100146_DEADX.w2.json](t106_7100146_DEADX.w2.json) · record
- [t106_7100146_L15X.w0.json](t106_7100146_L15X.w0.json) · record
- [t106_7100146_L15_40.w1.json](t106_7100146_L15_40.w1.json) · record
- [t106_7100147_DEADX.w2.json](t106_7100147_DEADX.w2.json) · record
- [t106_7100147_L15X.w0.json](t106_7100147_L15X.w0.json) · record
- [t106_7100147_L15_40.w1.json](t106_7100147_L15_40.w1.json) · record
- [t106_7100148_DEADX.w2.json](t106_7100148_DEADX.w2.json) · record
- [t106_7100148_L15X.w0.json](t106_7100148_L15X.w0.json) · record
- [t106_7100148_L15_40.w1.json](t106_7100148_L15_40.w1.json) · record
- [t106_7100149_DEADX.w2.json](t106_7100149_DEADX.w2.json) · record
- [t106_7100149_L15X.w0.json](t106_7100149_L15X.w0.json) · record
- [t106_7100149_L15_40.w1.json](t106_7100149_L15_40.w1.json) · record
- [t106_7100150_DEADX.w2.json](t106_7100150_DEADX.w2.json) · record
- [t106_7100150_L15X.w0.json](t106_7100150_L15X.w0.json) · record
- [t106_7100150_L15_40.w1.json](t106_7100150_L15_40.w1.json) · record
- [t106_7100151_DEADX.w2.json](t106_7100151_DEADX.w2.json) · record
- [t106_7100151_L15X.w0.json](t106_7100151_L15X.w0.json) · record
- [t106_7100151_L15_40.w1.json](t106_7100151_L15_40.w1.json) · record
- [t106_7100136_DEADX.json](t106_7100136_DEADX.json) · result
- [t106_7100136_L15X.json](t106_7100136_L15X.json) · result
- [t106_7100136_L15_40.json](t106_7100136_L15_40.json) · result
- [t106_7100137_DEADX.json](t106_7100137_DEADX.json) · result
- [t106_7100137_L15X.json](t106_7100137_L15X.json) · result
- [t106_7100137_L15_40.json](t106_7100137_L15_40.json) · result
- [t106_7100138_DEADX.json](t106_7100138_DEADX.json) · result
- [t106_7100138_L15X.json](t106_7100138_L15X.json) · result
- [t106_7100138_L15_40.json](t106_7100138_L15_40.json) · result
- [t106_7100139_DEADX.json](t106_7100139_DEADX.json) · result
- [t106_7100139_L15X.json](t106_7100139_L15X.json) · result
- [t106_7100139_L15_40.json](t106_7100139_L15_40.json) · result
- [t106_7100140_DEADX.json](t106_7100140_DEADX.json) · result
- [t106_7100140_L15X.json](t106_7100140_L15X.json) · result
- [t106_7100140_L15_40.json](t106_7100140_L15_40.json) · result
- [t106_7100141_DEADX.json](t106_7100141_DEADX.json) · result
- [t106_7100141_L15X.json](t106_7100141_L15X.json) · result
- [t106_7100141_L15_40.json](t106_7100141_L15_40.json) · result
- [t106_7100142_DEADX.json](t106_7100142_DEADX.json) · result
- [t106_7100142_L15X.json](t106_7100142_L15X.json) · result
- [t106_7100142_L15_40.json](t106_7100142_L15_40.json) · result
- [t106_7100143_DEADX.json](t106_7100143_DEADX.json) · result
- [t106_7100143_L15X.json](t106_7100143_L15X.json) · result
- [t106_7100143_L15_40.json](t106_7100143_L15_40.json) · result
- [t106_7100144_DEADX.json](t106_7100144_DEADX.json) · result
- [t106_7100144_L15X.json](t106_7100144_L15X.json) · result
- [t106_7100144_L15_40.json](t106_7100144_L15_40.json) · result
- [t106_7100145_DEADX.json](t106_7100145_DEADX.json) · result
- [t106_7100145_L15X.json](t106_7100145_L15X.json) · result
- [t106_7100145_L15_40.json](t106_7100145_L15_40.json) · result
- [t106_7100146_DEADX.json](t106_7100146_DEADX.json) · result
- [t106_7100146_L15X.json](t106_7100146_L15X.json) · result
- [t106_7100146_L15_40.json](t106_7100146_L15_40.json) · result
- [t106_7100147_DEADX.json](t106_7100147_DEADX.json) · result
- [t106_7100147_L15X.json](t106_7100147_L15X.json) · result
- [t106_7100147_L15_40.json](t106_7100147_L15_40.json) · result
- [t106_7100148_DEADX.json](t106_7100148_DEADX.json) · result
- [t106_7100148_L15X.json](t106_7100148_L15X.json) · result
- [t106_7100148_L15_40.json](t106_7100148_L15_40.json) · result
- [t106_7100149_DEADX.json](t106_7100149_DEADX.json) · result
- [t106_7100149_L15X.json](t106_7100149_L15X.json) · result
- [t106_7100149_L15_40.json](t106_7100149_L15_40.json) · result
- [t106_7100150_DEADX.json](t106_7100150_DEADX.json) · result
- [t106_7100150_L15X.json](t106_7100150_L15X.json) · result
- [t106_7100150_L15_40.json](t106_7100150_L15_40.json) · result
- [t106_7100151_DEADX.json](t106_7100151_DEADX.json) · result
- [t106_7100151_L15X.json](t106_7100151_L15X.json) · result
- [t106_7100151_L15_40.json](t106_7100151_L15_40.json) · result
- [worker_w0.log](worker_w0.log) · log
- [worker_w1.log](worker_w1.log) · log
- [worker_w2.log](worker_w2.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-exact-final-layer-row-push-dead-cuts-rejected-mse) · [Copy provenance](../../../PROVENANCE.json)

