# Grouped reads restricted to initial or later rank: +0.179% and +0.531% adjusted

2026-10-21T16:20:22Z · experiment · rejected · measured

Restricting grouped strided-read plans by the rank key produced a worse adjusted tradeoff than the adopted fixed per-prediction read cap.

Recorded measurement:

    Reference: first128_paid_nocensus_native32_v1 (#337506 science); per-line concurrent controls apply.
    - 2026-10-21T16:20:22Z | f21_sm40_rank0only | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) == 4 and k[3] != 0)} | ref f21_ctl_sm40_rank0only | all32 adj +0.179% (se 0.058) | held25 adj +0.215% (se 0.072) | wins 12/32 | raw -0.002% | bill +0.181% | requests +0.12% | puts -0.72% | put_bytes_f4 -0.60% | wall +4.98% | backend +4.48% | residual +4.22% | max-rank wall +26.28% | slow10 wall -2.94% req +0.20% adj +0.175% | sm40 calls 3.96875 vs 6.125 | byte-identical n (2/32) | VM run /home/arcworker/arc-runs/f21_sm40_rank0only | sha256 92cf747e6e2430e044005fbf112991467f1c1ddaa7fb6d90d042da89e0972cbd | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_rank0only.py
    - 2026-10-21T16:31:20Z | f21_sm40_lateonly | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) == 4 and k[3] == 0)} | ref f21_ctl_sm40_lateonly | all32 adj +0.531% (se 0.090) | held25 adj +0.558% (se 0.102) | wins 5/32 | raw -0.068% | bill +0.600% | requests +0.11% | puts -0.40% | put_bytes_f4 +0.33% | wall -12.42% | backend -12.62% | residual -1.69% | max-rank wall -5.16% | slow10 wall -16.64% req +0.50% adj +0.455% | sm40 calls 2.15625 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_sm40_lateonly | sha256 30817559f8c86e8c8d3b0251a127d5a811a9218f354915509c8b7f01d5c307b4 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_lateonly.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_sm40_rank0only.py](code/cand_sm40_rank0only.py)
- [code/cand_sm40_lateonly.py](code/cand_sm40_lateonly.py)

## Evidence

- [VM_SWEEP_RESULTS_2026-10-21.md#L44](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L44)
- [VM_SWEEP_RESULTS_2026-10-21.md#L47](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L47)

<details>
<summary>All result files and records (1)</summary>

- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-grouped-reads-restricted-initial-or-later-rank-0-179) · [Copy provenance](../../../PROVENANCE.json)

