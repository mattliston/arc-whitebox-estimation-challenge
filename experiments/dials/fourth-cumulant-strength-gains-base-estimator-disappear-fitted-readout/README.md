# Fourth-cumulant strength gains on the base estimator disappear on the fitted readout

2026-10-21T20:43:08Z · experiment · rejected · measured

A small fourth-cumulant coefficient increase improved the base file, but the same change worsened the combined readout on both native platforms. Nearby strength and band variants and the plane coefficient decrease were also measured; the base benefit was not additive.

Recorded measurement:

    - 2026-10-21T20:43:08Z [fable-lead] DIAL HUNT (teammate, 49bd base, vs global ref): k4lc_up x1.05 -0.144% (held25 -0.179) = the only live dial; k4lc_up2 x1.10 -0.033 (worse); np_dn -0.096 / np_dn2 -0.075 (flat, held25 -0.06/-0.08); k4band_up -0.031, k4band_dn +0.150, k4band3_up +0.196 (worse). Stacked k4lc x1.05 onto hgfh+cap4+8e9 clean (hgfh_c4s8_k4lc105_clean): VM32 16/32, fresh running, Mac32+gates queued. #337617 passed setup (grading, 5/100, max wall 103 s); #337619 queued.
    - 2026-10-21T20:51:38Z [fable-lead] hgfh_c4s8_k4lc105_clean (K4L_C x1.05 on the hg+fh+cap4+8e9 clean stack) VM32 vs 49bd: all32 -0.442% held25 -0.538% raw -0.729% bill +0.289%; fresh12 -0.592%. vs the plain hgfh cap4s8e9 (XSG-on -0.620/-0.678, raw -0.826, fresh -0.637): raw WORSE by 0.1%, adjusted worse by ~0.1% after the XSG-off bill (+0.077): the k4lc dial does NOT transfer onto the readout stack (co-adaptation). CLOSED. Fire candidate stays hgfh cap4s8e9 clean (#337619 queued); fallback hgfh cap3 clean (1966e86ecaf8). Score hunt exhausted: features, strength, dials, margin tails all closed.
    - 2026-10-21T20:55:09Z [fable-lead] Mac32 hgfh_c4s8_k4lc105_clean vs 49bd: all32 -0.367% held25 -0.419% raw -0.654% bill +0.289% (vs plain hgfh cap4s8e9 -0.423/-0.524, bill +0.211): confirms k4lc does not transfer; CLOSED. #337617 42/100 graded clean (max wall 107.3 s); #337619 queued 16 min.
    Sweep result detail: - 2026-10-21T19:38:02Z | f21_np_up | _NP_STRENGTH = (0.0, 1.8250736189933134, 1.472262585730081, 1.274543548278647, 1.3240929144838185, 1.2441616412100303, 1.2441616412100303, 1.2441616412100303, 1 | ref first128_paid_nocensus_native32_v1 | all32 adj +0.083% (se 0.077) | held25 adj +0.089% (se 0.084) | wins 12/32 | raw +0.083% | bill -0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +10.43% | backend +10.71% | residual +9.30% | max-rank wall +15.17% | slow10 wall +8.79% req +0.00% adj -0.134% | slow10ctl ratio 1.088 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_np_up | sha256 da7d30c3c1cff828aa4f094fa173bff3e9584dd4fc23b2ece6fc79fe448d4ecb | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_np_up.py
    - 2026-10-21T19:39:06Z | f21_np_dn | _NP_STRENGTH = (0.0, 1.7187586508966153, 1.3864997166584256, 1.200298293039114, 1.2469612883973824, 1.1716862057997373, 1.1716862057997373, 1.1716862057997373,  | ref first128_paid_nocensus_native32_v1 | all32 adj -0.096% (se 0.083) | held25 adj -0.062% (se 0.100) | wins 19/32 | raw -0.094% | bill -0.002% | requests +0.00% | puts +0.00% | put_bytes_f4 -0.00% | wall +9.75% | backend +10.08% | residual +6.87% | max-rank wall +18.31% | slow10 wall +15.48% req +0.00% adj -0.047% | slow10ctl ratio 1.155 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_np_dn | sha256 6c4e17d7f3b884d9a6fa77a7f96a1e0a84696fdb44c29103954463c44d06a912 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_np_dn.py
    - 2026-10-21T19:47:14Z | f21_k4band_up | _S19_K4BAND = (0.9924165946610191, 0.9910213769857896, 0.9709508938573927) | ref first128_paid_nocensus_native32_v1 | all32 adj -0.031% (se 0.134) | held25 adj +0.023% (se 0.155) | wins 14/32 | raw -0.031% | bill -0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +10.10% | backend +10.37% | residual +8.19% | max-rank wall +15.72% | slow10 wall +7.94% req +0.00% adj -0.050% | slow10ctl ratio 1.079 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4band_up | sha256 e4e5015bde8527cbaf420b3d34440098e0a359560980b67bd4b3b72f653818cd | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4band_up.py
    - 2026-10-21T19:48:19Z | f21_k4band_dn | _S19_K4BAND = (0.9727647809053552, 0.9713971913029027, 0.9517241434839789) | ref first128_paid_nocensus_native32_v1 | all32 adj +0.150% (se 0.154) | held25 adj +0.102% (se 0.192) | wins 13/32 | raw +0.150% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +9.59% | backend +9.92% | residual +6.14% | max-rank wall +17.87% | slow10 wall +15.44% req +0.00% adj +0.071% | slow10ctl ratio 1.154 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4band_dn | sha256 a4b0fd5c88e3c983f2b2fa8e0c96dd23b7ee9241412bd6b3e7081a857cc9949a | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4band_dn.py
    - 2026-10-21T19:54:39Z | f21_k4lc_dn | k4lc_dn | RUN FAILED (COMPLETE.json successful=false; setup or group failure, see /home/arcworker/arc-runs/f21_k4lc_dn/part01.log) | CLOSED
    - 2026-10-21T20:01:02Z | f21_k4lc_up | _K4L_C = (0.855039891950189, 0.7103163656163689) | ref first128_paid_nocensus_native32_v1 | all32 adj -0.144% (se 0.095) | held25 adj -0.179% (se 0.115) | wins 17/32 | raw -0.144% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +51.08% | backend +49.00% | residual +39.52% | max-rank wall +91.07% | slow10 wall +16.13% req +0.00% adj -0.227% | slow10ctl ratio 1.161 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4lc_up | sha256 b2cac9a3d1e0b38683e8685985978b10cda7bfa85efda444a2b7b96fce82dca2 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4lc_up.py
    - 2026-10-21T20:38:39Z | f21_np_dn2 | _NP_STRENGTH = (0.0, 1.6656011668482664, 1.3436182821225982, 1.1631756654193475, 1.2083954753541644, 1.1354484880945908, 1.1354484880945908, 1.1354484880945908, | ref first128_paid_nocensus_native32_v1 | all32 adj -0.075% (se 0.090) | held25 adj -0.080% (se 0.108) | wins 16/32 | raw -0.075% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +58.52% | backend +58.26% | residual +47.51% | max-rank wall +85.37% | slow10 wall +41.53% req +0.00% adj -0.057% | slow10ctl ratio 1.415 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_np_dn2 | sha256 45a29638f078a7c6e27300b12b5b813c6a9602ec0d2bd5aebde8749e98269910 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_np_dn2.py
    - 2026-10-21T20:39:43Z | f21_k4lc_up2 | _K4L_C = (0.8957560772811505, 0.7441409544552436) | ref first128_paid_nocensus_native32_v1 | all32 adj -0.033% (se 0.077) | held25 adj -0.029% (se 0.086) | wins 17/32 | raw -0.033% | bill -0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +62.93% | backend +62.74% | residual +50.19% | max-rank wall +86.69% | slow10 wall +53.74% req +0.00% adj +0.056% | slow10ctl ratio 1.537 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4lc_up2 | sha256 033a490af8652e7e6e84e82bcea142e9bb2e85daac0d597d370f3f96bde787cd | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4lc_up2.py
    - 2026-10-21T20:42:34Z | f21_k4band3_up | _S19_K4BAND = (0.9825906877831871, 0.9812092841443462, 0.9805642690440995) | ref first128_paid_nocensus_native32_v1 | all32 adj +0.196% (se 0.081) | held25 adj +0.231% (se 0.096) | wins 13/32 | raw +0.196% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +63.04% | backend +62.77% | residual +49.58% | max-rank wall +86.39% | slow10 wall +53.71% req +0.00% adj +0.408% | slow10ctl ratio 1.537 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4band3_up | sha256 80abfbca3123559d60a5b045933641e8b8cf7fc4971d3ac5c4c03fa29e08a766 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4band3_up.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_np_up.py](code/cand_np_up.py)
- [code/cand_np_dn.py](code/cand_np_dn.py)
- [code/cand_k4band_up.py](code/cand_k4band_up.py)
- [code/cand_k4band_dn.py](code/cand_k4band_dn.py)
- [code/cand_k4lc_up.py](code/cand_k4lc_up.py)
- [code/cand_np_dn2.py](code/cand_np_dn2.py)
- [code/cand_k4lc_up2.py](code/cand_k4lc_up2.py)
- [code/cand_k4band3_up.py](code/cand_k4band3_up.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7551](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7551)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7553](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7553)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7554](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7554)
- [VM_SWEEP_RESULTS_2026-10-21.md#L81](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L81)
- [VM_SWEEP_RESULTS_2026-10-21.md#L82](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L82)
- [VM_SWEEP_RESULTS_2026-10-21.md#L83](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L83)
- [VM_SWEEP_RESULTS_2026-10-21.md#L84](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L84)
- [VM_SWEEP_RESULTS_2026-10-21.md#L85](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L85)
- [VM_SWEEP_RESULTS_2026-10-21.md#L86](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L86)
- [VM_SWEEP_RESULTS_2026-10-21.md#L89](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L89)
- [VM_SWEEP_RESULTS_2026-10-21.md#L90](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L90)
- [VM_SWEEP_RESULTS_2026-10-21.md#L91](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L91)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-fourth-cumulant-strength-gains-base-estimator-disappear-fitted-readout) · [Copy provenance](../../../PROVENANCE.json)

