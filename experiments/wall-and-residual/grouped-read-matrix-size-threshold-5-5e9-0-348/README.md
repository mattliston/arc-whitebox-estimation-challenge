# Grouped-read matrix-size threshold 5.5e9: +0.348% adjusted under repeated paired measurement

2026-10-21T17:34:49Z · experiment · rejected · measured

Dropping grouped-read plans whose matrix dimension product reaches 5.5e9 reduced runtime at increased metered cost. The repeat paired comparison retained the same adjusted result, which was too expensive for the desired source.

Recorded measurement:

    Reference: first128_paid_nocensus_native32_v1 (#337506 science); per-line concurrent controls apply.
    - 2026-10-21T17:34:49Z | f21_sm40_small | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) >= 3 and k[0] * k[1] * k[2] >= 5.5e9)} | ref f21_ctl_sm40_small | all32 adj +0.348% (se 0.089) | held25 adj +0.388% (se 0.103) | wins 8/32 | raw -0.086% | bill +0.435% | requests +0.21% | puts -0.42% | put_bytes_f4 -0.07% | wall -21.73% | backend -22.02% | residual -12.18% | max-rank wall +3.43% | slow10 wall -22.50% req +0.57% adj +0.399% | sm40 calls 2.90625 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_sm40_small | sha256 b988e20102cf5c40f375d6f75e29bc9221b5554deadcf65a0c7cb1ad05088e1b | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_small.py
    - 2026-10-21T18:33:26Z | f21_sm40_small2 | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) >= 3 and k[0] * k[1] * k[2] >= 5.5e9)} | ref f21_ctl_sm40_small2 | all32 adj +0.348% (se 0.089) | held25 adj +0.388% (se 0.103) | wins 8/32 | raw -0.086% | bill +0.435% | requests +0.21% | puts -0.42% | put_bytes_f4 -0.07% | wall -5.60% | backend -6.07% | residual +0.75% | max-rank wall -0.68% | slow10 wall -4.62% req +0.57% adj +0.399% | slow10ctl ratio 0.940 | sm40 calls 2.90625 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_sm40_small2 | sha256 b988e20102cf5c40f375d6f75e29bc9221b5554deadcf65a0c7cb1ad05088e1b | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_small.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_sm40_small.py](code/cand_sm40_small.py)

## Evidence

- [VM_SWEEP_RESULTS_2026-10-21.md#L59](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L59)
- [VM_SWEEP_RESULTS_2026-10-21.md#L69](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L69)

<details>
<summary>All result files and records (1)</summary>

- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-grouped-read-matrix-size-threshold-5-5e9-0-348) · [Copy provenance](../../../PROVENANCE.json)

