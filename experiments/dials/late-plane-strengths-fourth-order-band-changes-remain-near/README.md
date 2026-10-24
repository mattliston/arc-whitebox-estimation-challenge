# Late plane strengths and fourth-order band changes remain near zero or worsen the deployed source

2026-10-21T21:14:59Z · experiment · rejected · measured

The completed downward fourth-order coefficient rerun and late-layer plane strength changes were small relative to their standard errors. The third-band decrease was effectively score-neutral.

Recorded measurement:

    Reference: first128_paid_nocensus_native32_v1 (#337506 science); per-line concurrent controls apply.
    - 2026-10-21T21:14:59Z | f21_k4lc_dn_b | _K4L_C x0.95 (re-run of the guard-killed k4lc_dn) | ref first128_paid_nocensus_native32_v1 | all32 adj -0.076% (se 0.096) | held25 adj -0.035% (se 0.107) | wins 16/32 | raw -0.076% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +59.90% | backend +59.83% | residual +47.31% | max-rank wall +79.51% | slow10 wall +52.80% req +0.00% adj -0.191% | slow10ctl ratio 1.528 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4lc_dn_b | sha256 10285bb02f9011e80abee2c763a1688b3534bc1ea10717377343c8ea3a06c665 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4lc_dn.py
    - 2026-10-21T21:15:02Z | f21_np_late_up | _NP_STRENGTH L10-14 x1.05 | ref first128_paid_nocensus_native32_v1 | all32 adj -0.031% (se 0.052) | held25 adj +0.041% (se 0.055) | wins 15/32 | raw -0.031% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +43.14% | backend +43.13% | residual +33.44% | max-rank wall +77.81% | slow10 wall +51.98% req +0.00% adj -0.010% | slow10ctl ratio 1.520 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_np_late_up | sha256 c03eeeed79f3eb2fe44ca4f6411b9953df64f01e1d1936e33121a1275f49c9c1 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_np_late_up.py
    - 2026-10-21T21:15:05Z | f21_np_late_dn | _NP_STRENGTH L10-14 x0.95 | ref first128_paid_nocensus_native32_v1 | all32 adj -0.037% (se 0.047) | held25 adj -0.040% (se 0.057) | wins 19/32 | raw -0.037% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +20.70% | backend +20.69% | residual +13.00% | max-rank wall +58.92% | slow10 wall +47.08% req +0.00% adj -0.059% | slow10ctl ratio 1.471 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_np_late_dn | sha256 00277c58265b1baa8dca4e7a26768b2b3a392c1b775ae0da6f08707fbecc0415 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_np_late_dn.py
    - 2026-10-21T21:15:08Z | f21_k4band3_dn | _S19_K4BAND band3 x0.98 | ref first128_paid_nocensus_native32_v1 | all32 adj +0.001% (se 0.083) | held25 adj +0.028% (se 0.096) | wins 14/32 | raw +0.001% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +53.04% | backend +52.92% | residual +42.19% | max-rank wall +87.09% | slow10 wall +51.33% req +0.00% adj -0.207% | slow10ctl ratio 1.513 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4band3_dn | sha256 796968e128ee1b6fdbb08bd52ff0fc40222b7a404b84f4fc75f2d0f9ac8d529a | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4band3_dn.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_k4lc_dn.py](code/cand_k4lc_dn.py)
- [code/cand_np_late_up.py](code/cand_np_late_up.py)
- [code/cand_np_late_dn.py](code/cand_np_late_dn.py)
- [code/cand_k4band3_dn.py](code/cand_k4band3_dn.py)

## Evidence

- [VM_SWEEP_RESULTS_2026-10-21.md#L92](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L92)
- [VM_SWEEP_RESULTS_2026-10-21.md#L93](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L93)
- [VM_SWEEP_RESULTS_2026-10-21.md#L94](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L94)
- [VM_SWEEP_RESULTS_2026-10-21.md#L95](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L95)

<details>
<summary>All result files and records (1)</summary>

- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-late-plane-strengths-fourth-order-band-changes-remain-near) · [Copy provenance](../../../PROVENANCE.json)

