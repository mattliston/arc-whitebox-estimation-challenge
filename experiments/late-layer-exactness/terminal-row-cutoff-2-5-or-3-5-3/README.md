# Terminal row cutoff 2.5 or 3.5: +3.147% or +0.190% adjusted on the deployed first-128 source

2026-10-21T14:35:30Z · experiment · rejected · measured

Both tightening and loosening the terminal standardized row threshold worsened adjusted score relative to the deployed threshold 3.0. The recorded runtime rescore retains load-sensitive wall measurements.

Recorded measurement:

    Reference: first128_paid_nocensus_native32_v1 (#337506 science); per-line concurrent controls apply.
    - 2026-10-21T14:35:30Z | f21_rowcut_2p5 | _L15_ROWCUT_Z 3.0 -> 2.5 | all32 adj +3.147% (se 0.308) | held25 adj +3.231% (se 0.317) | wins 0/32 | raw +3.580% | bill -0.417% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_rowcut_2p5 | sha256 82f3e26da2ef795336c1ea83045f5df70205f812903583cf1629ac057202d2c8 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_rowcut_2p5.py
    - 2026-10-21T14:35:33Z | f21_rowcut_3p5 | _L15_ROWCUT_Z 3.0 -> 3.5 | all32 adj +0.190% (se 0.060) | held25 adj +0.247% (se 0.060) | wins 6/32 | raw -0.109% | bill +0.299% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_rowcut_3p5 | sha256 72d55dc0edc5cf7a75f0107f0e3ac66707b134e6b15a26c1dff64ce42c93c432 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_rowcut_3p5.py
    - 2026-10-21T14:51:07Z | f21_rowcut_2p5 | _L15_ROWCUT_Z 3.0 -> 2.5 (re-scored with wall/requests) | all32 adj +3.147% (se 0.308) | held25 adj +3.231% (se 0.317) | wins 0/32 | raw +3.580% | bill -0.417% | wall +27.21% | requests -0.03% | puts +0.11% | max-rank wall +60.63% | slow10 wall +16.21% req -0.02% adj +3.382% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_rowcut_2p5 | sha256 82f3e26da2ef795336c1ea83045f5df70205f812903583cf1629ac057202d2c8 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_rowcut_2p5.py
    - 2026-10-21T14:51:10Z | f21_rowcut_3p5 | _L15_ROWCUT_Z 3.0 -> 3.5 (re-scored with wall/requests) | all32 adj +0.190% (se 0.060) | held25 adj +0.247% (se 0.060) | wins 6/32 | raw -0.109% | bill +0.299% | wall +28.18% | requests -0.09% | puts -0.26% | max-rank wall +62.37% | slow10 wall +18.25% req +0.02% adj +0.206% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_rowcut_3p5 | sha256 72d55dc0edc5cf7a75f0107f0e3ac66707b134e6b15a26c1dff64ce42c93c432 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_rowcut_3p5.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_rowcut_2p5.py](code/cand_rowcut_2p5.py)
- [code/cand_rowcut_3p5.py](code/cand_rowcut_3p5.py)

## Evidence

- [VM_SWEEP_RESULTS_2026-10-21.md#L5](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L5)
- [VM_SWEEP_RESULTS_2026-10-21.md#L6](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L6)
- [VM_SWEEP_RESULTS_2026-10-21.md#L9](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L9)
- [VM_SWEEP_RESULTS_2026-10-21.md#L10](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L10)

<details>
<summary>All result files and records (1)</summary>

- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-terminal-row-cutoff-2-5-or-3-5-3) · [Copy provenance](../../../PROVENANCE.json)

