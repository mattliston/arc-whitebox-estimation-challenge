# Carrier age singleton and square-depth cap 5 are byte-identical no-ops in the deployed estimator

2026-10-21T17:37:45Z · experiment · rejected · measured

The two parameter overrides produced identical predictions and FLOPs for all 32 networks. Neither changed the effective deployed path.

Recorded measurement:

    Reference: first128_paid_nocensus_native32_v1 (#337506 science); per-line concurrent controls apply.
    - 2026-10-21T17:37:45Z | f21_hcf_1 | _HCF_AGES = (1.0,) | ref f21_ctl_hcf_1 | all32 adj +0.000% (se 0.000) | held25 adj +0.000% (se 0.000) | wins 0/32 | raw +0.000% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall -2.61% | backend -2.65% | residual -2.88% | max-rank wall +5.97% | slow10 wall +4.19% req +0.00% adj +0.000% | sm40 calls 6.125 vs 6.125 | byte-identical y (32/32) | VM run /home/arcworker/arc-runs/f21_hcf_1 | sha256 a5fe6d7f546ef4d967211864550a6cbe34ee18b13bc7d55e0578eab66c7ebc19 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_hcf_1.py
    - 2026-10-21T17:52:18Z | f21_sqcap5 | _SQUARE_DEPTH_CAP = 5 | ref f21_ctl_sqcap5 | all32 adj +0.000% (se 0.000) | held25 adj +0.000% (se 0.000) | wins 0/32 | raw +0.000% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +23.30% | backend +23.17% | residual +15.75% | max-rank wall +42.43% | slow10 wall +41.14% req +0.00% adj +0.000% | sm40 calls 6.125 vs 6.125 | byte-identical y (32/32) | VM run /home/arcworker/arc-runs/f21_sqcap5 | sha256 250d22d4f8fa2e4180c9f1401f08778c4512dc55983f8bf39206d6613db7d9dc | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sqcap5.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_hcf_1.py](code/cand_hcf_1.py)
- [code/cand_sqcap5.py](code/cand_sqcap5.py)

## Evidence

- [VM_SWEEP_RESULTS_2026-10-21.md#L60](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L60)
- [VM_SWEEP_RESULTS_2026-10-21.md#L62](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L62)

<details>
<summary>All result files and records (1)</summary>

- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-carrier-age-singleton-square-depth-cap-5-byte-identical) · [Copy provenance](../../../PROVENANCE.json)

