# Cumulant carrier age reduced to one: +2.306% and +0.917% adjusted; another family fails setup

2026-10-21T17:34:28Z · experiment · rejected · measured

Reducing the retained carrier age to one worsened both completed fourth-order variants. The additional third-order/plane variant did not complete and has no native score.

Recorded measurement:

    Reference: first128_paid_nocensus_native32_v1 (#337506 science); per-line concurrent controls apply.
    - 2026-10-21T17:34:28Z | f21_np_k3l_age1 | np_k3l_age1 | RUN FAILED (COMPLETE.json successful=false; setup or group failure, see /home/arcworker/arc-runs/f21_np_k3l_age1/part01.log) | CLOSED
    - 2026-10-21T17:34:43Z | f21_f4c_age1 | _F4C_AGE = 1 ; _F4C_THETA = (0.704467345174,) ; _F4C_C = ((0.0, 0.459128468809),) ; _F4C_C_BY_BIRTH = {6: ((0.0, 0.473219284687),)} | ref f21_ctl_f4c_age1 | all32 adj +2.306% (se 0.368) | held25 adj +2.382% (se 0.452) | wins 5/32 | raw +2.492% | bill -0.182% | requests -0.39% | puts -0.70% | put_bytes_f4 -0.47% | wall -7.05% | backend -7.02% | residual -4.35% | max-rank wall +14.39% | slow10 wall -18.47% req -0.39% adj +1.678% | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_f4c_age1 | sha256 5289c02196bc3845dd0645bb2d280d76be768dc3b02c216aaa48e28a4ce7b6ce | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_f4c_age1.py
    - 2026-10-21T17:34:46Z | f21_k4l_age1 | _K4L_AGE = 1 ; _K4L_C = (0.8143237066192276,) ; _K4L34_BY_LAYER = {} | ref f21_ctl_k4l_age1 | all32 adj +0.917% (se 0.255) | held25 adj +1.032% (se 0.288) | wins 8/32 | raw +1.039% | bill -0.121% | requests -0.92% | puts -1.20% | put_bytes_f4 -0.32% | wall +22.24% | backend +21.40% | residual +16.26% | max-rank wall +44.59% | slow10 wall +17.90% req -0.91% adj +0.412% | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4l_age1 | sha256 01420161f8a5d713fb740da8c37d42faca9076120af533493a804b2adf5ca9bb | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4l_age1.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_f4c_age1.py](code/cand_f4c_age1.py)
- [code/cand_k4l_age1.py](code/cand_k4l_age1.py)

## Evidence

- [VM_SWEEP_RESULTS_2026-10-21.md#L52](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L52)
- [VM_SWEEP_RESULTS_2026-10-21.md#L57](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L57)
- [VM_SWEEP_RESULTS_2026-10-21.md#L58](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L58)

<details>
<summary>All result files and records (1)</summary>

- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-cumulant-carrier-age-reduced-one-2-306-0-917) · [Copy provenance](../../../PROVENANCE.json)

