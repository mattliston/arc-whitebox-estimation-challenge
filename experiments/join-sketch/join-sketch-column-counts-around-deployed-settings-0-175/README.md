# Join-sketch column counts around the deployed settings: +0.175%, +0.011% and +0.110% adjusted

2026-10-21T14:49:55Z · experiment · rejected · measured

Increasing or decreasing the joined-plane sketch column counts failed to improve adjusted score. The tighter 1024/768/768 schedule was also measured against a concurrent control.

Recorded measurement:

    Reference: first128_paid_nocensus_native32_v1 (#337506 science); per-line concurrent controls apply.
    - 2026-10-21T14:49:55Z | f21_kc_loose | _JSK_KC {7:1792,10:1280,13:1280} (one step looser) | all32 adj +0.175% (se 0.097) | held25 adj +0.240% (se 0.112) | wins 10/32 | raw -0.006% | bill +0.181% | wall +7.40% | requests +0.02% | puts +0.24% | max-rank wall +16.52% | slow10 wall -1.08% req +0.09% adj +0.211% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_kc_loose | sha256 e53789d3d4e81e0314c6fae0c31bb0c375d778aa75d1d19126805a9da98a0454 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_kc_loose.py
    - 2026-10-21T14:49:58Z | f21_kc_tight | _JSK_KC {7:1280,10:768,13:768} (one step tighter) | all32 adj +0.011% (se 0.131) | held25 adj +0.029% (se 0.139) | wins 15/32 | raw +0.104% | bill -0.093% | wall +7.54% | requests -0.01% | puts +0.07% | max-rank wall +18.53% | slow10 wall +15.68% req +0.07% adj -0.131% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_kc_tight | sha256 5926e3f44e658048ef1d357774f144cff7f5545fb8739f94eb5aa7a910a24cb3 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_kc_tight.py
    - 2026-10-21T18:21:43Z | f21_kc_1024_768 | _JSK_KC = {7: 1024, 10: 768, 13: 768} | ref f21_ctl_kc_1024_768 | all32 adj +0.110% (se 0.140) | held25 adj +0.160% (se 0.140) | wins 14/32 | raw +0.280% | bill -0.169% | requests +0.02% | puts +0.02% | put_bytes_f4 -0.04% | wall +5.44% | backend +5.60% | residual +4.23% | max-rank wall +143.81% | slow10 wall -1.27% req +0.10% adj +0.168% | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_kc_1024_768 | sha256 4415254cfb16128c0af3d2b914473dd5398b1ff917b903e593c9f8a75e6039a6 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_kc_1024_768.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_kc_loose.py](code/cand_kc_loose.py)
- [code/cand_kc_tight.py](code/cand_kc_tight.py)
- [code/cand_kc_1024_768.py](code/cand_kc_1024_768.py)

## Evidence

- [VM_SWEEP_RESULTS_2026-10-21.md#L7](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L7)
- [VM_SWEEP_RESULTS_2026-10-21.md#L8](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L8)
- [VM_SWEEP_RESULTS_2026-10-21.md#L65](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L65)

<details>
<summary>All result files and records (1)</summary>

- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-join-sketch-column-counts-around-deployed-settings-0-175) · [Copy provenance](../../../PROVENANCE.json)

