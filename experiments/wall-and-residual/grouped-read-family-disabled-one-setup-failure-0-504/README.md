# Grouped-read family disabled: one setup failure and +0.504% adjusted for the alternate family

2026-10-21T15:19:23Z · experiment · rejected · measured

Disabling the primary grouped strided-read plans failed before a completed score. Disabling the alternate grouped-read family completed but added FLOPs and worsened adjusted score; the paired full result supersedes the earlier twelve-network partial.

Recorded measurement:

    Reference: first128_paid_nocensus_native32_v1 (#337506 science); per-line concurrent controls apply.
    - 2026-10-21T15:19:23Z | f21_sm40_off | sm40_off | RUN FAILED (COMPLETE.json successful=false; setup or group failure, see /home/arcworker/arc-runs/f21_sm40_off/part01.log) | CLOSED
    - 2026-10-21T15:21:16Z | f21_smx_off | _SMX_ON = False | ref f21_ctl_smx_off | all32 adj +0.504% (se 0.064) | held25 adj +0.574% (se 0.073) | wins 2/32 | raw -0.111% | bill +0.616% | requests +0.13% | puts +0.50% | put_bytes_f4 -1.63% | wall +1.43% | max-rank wall +14.83% | slow10 wall +11.69% req +0.16% adj +0.546% | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_smx_off | sha256 846843b556cf19bbe135b8546bcc0ef084a535e2af65da702e78a6a3d31c4103 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_smx_off.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_smx_off.py](code/cand_smx_off.py)

## Evidence

- [VM_SWEEP_RESULTS_2026-10-21.md#L20](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L20)
- [VM_SWEEP_RESULTS_2026-10-21.md#L22](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L22)

<details>
<summary>All result files and records (1)</summary>

- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-grouped-read-family-disabled-one-setup-failure-0-504) · [Copy provenance](../../../PROVENANCE.json)

