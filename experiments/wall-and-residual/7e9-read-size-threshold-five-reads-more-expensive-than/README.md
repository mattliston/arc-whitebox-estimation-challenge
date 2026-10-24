# A 7e9 read-size threshold and five reads are more expensive than the submitted cap-4 tail

2026-10-21T21:34:18Z · experiment · rejected · measured

Reducing the matrix-size threshold or permitting a fifth grouped read added FLOPs without a sufficient raw-score gain. The completed comparison and Mac panel rejected these alternatives; memory-interrupted Linux partials were not treated as completed evidence.

Recorded measurement:

    - 2026-10-21T21:34:18Z [fable-lead] Mac32 hgfh_c5s7e9_clean vs 49bd: all32 -0.280% held25 -0.386% raw -0.673% bill +0.396% vs the #337619 twin (hgfh cap4s8e9 XSG-on) -0.423/-0.524 raw -0.632 bill +0.211: raw -0.04 better but bill +0.185 (XSG-off +0.077 + cap5/7e9 +0.11) -> adjusted ~0.14% WORSE. cap5+7e9 tail CLOSED; cap4+8e9 stays the final tail. Gates clear anyway: hgfh_c5s7e9_clean archive af375c6ec18e, hgfh_c4s7e9_clean ffd824a61a09 (both not to be fired). Killed c5s7e9 VM32+fresh. Remaining counterplay: fh20/fh10 per-block strength (VM32 7-12/32).
    - 2026-10-21T21:28:37Z [fable-lead] TEAMMATE partials (guard-retired triple): cap4+7e9 vs cap4+8e9 (23 ranks) adj +0.101%, bill +0.137%, wall flat (slow10 1.005) -> DEAD (cap4 already limits reads; extra drops only cost bill); cap5+7e9 vs cap4+8e9 (10 ranks) adj -0.087%, raw -0.181%, bill +0.094%, wall +1.1% -> full pair re-running (ETA 21:41Z). Killed hgfh_c4s7e9_clean runs + its Mac32 rerun; hgfh_c5s7e9_clean stays (VM32+fresh, Mac32 running).
    - 2026-10-21T22:00:13Z [fable-lead] STRENGTH x1.35 (hgfh_s135_c4s8e9_clean) VM32 vs 49bd: all32 -0.572% (22/32) held25 -0.654% raw -0.859% bill +0.289% vs the clean twin -0.542/-0.600 raw -0.826: -0.03/-0.05 better (noise-level, positive). s165 VM32 complete, scoring; Mac32 s135 in window. Fresh cohorts launched for s135/s165. cap5 family closed on the full pair (+0.071%).
    Sweep result detail: - 2026-10-21T21:56:37Z | f21_sm40_cap5_size7e9 | cap5 + size7e9 (guard-retired at 10/32; re-run as _b) | CANCELLED/RETIRED (not a variant failure: lead cancellation or memory-guard retirement) | CLOSED
    - 2026-10-21T21:56:37Z | f21_sm40_cap4_size7e9 | cap4 + size7e9 (guard-retired at 23/32; partial vs ctl: adj +0.101%, bill +0.137%, wall flat) | CANCELLED/RETIRED (not a variant failure: lead cancellation or memory-guard retirement) | CLOSED
    - 2026-10-21T21:56:37Z | f21_sm40_cap4_size8e9_c | control of the 21:14Z triple (guard-retired at 23/32) | CANCELLED/RETIRED (not a variant failure: lead cancellation or memory-guard retirement) | CLOSED
    - 2026-10-21T21:54:50Z | f21_sm40_cap4_size8e9_d | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) >= 3 and k[0] * k[1] * k[2] >= 8.0e9)} ; class _Sm40CapPlans(dict): ;     def get | ref first128_paid_nocensus_native32_v1 | all32 adj +0.138% (se 0.062) | held25 adj +0.172% (se 0.071) | wins 8/32 | raw -0.060% | bill +0.198% | requests +0.13% | puts -0.70% | put_bytes_f4 -0.73% | wall +58.24% | backend +57.18% | residual +50.59% | max-rank wall +81.07% | slow10 wall +46.87% req +0.24% adj +0.150% | slow10ctl ratio 1.469 | sm40 calls 3.9375 vs 6.125 | byte-identical n (2/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap4_size8e9_d | sha256 7eb1d130dddd888bdcbcf45253006a567f1f8434e11c43c64d93c8187b40a4d5 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap4_size8e9.py
    - 2026-10-21T21:54:53Z | f21_sm40_cap5_size7e9_b | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) >= 3 and k[0] * k[1] * k[2] >= 7.0e9)} ; class _Sm40CapPlans(dict): ;     def get | ref f21_sm40_cap4_size8e9_d | all32 adj +0.071% (se 0.075) | held25 adj +0.066% (se 0.082) | wins 14/32 | raw -0.037% | bill +0.108% | requests +0.26% | puts +0.65% | put_bytes_f4 +1.31% | wall +1.05% | backend +1.15% | residual +0.84% | max-rank wall +7.69% | slow10 wall +2.49% req +0.27% adj -0.027% | slow10ctl ratio 0.995 | sm40 calls 4.28125 vs 3.9375 | byte-identical n (1/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap5_size7e9_b | sha256 47558b7a6e1e601cbd104cff89b75463cabf106f8f732e9b4dde41e18c12d68b | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap5_size7e9.py
    - 2026-10-21T21:56:37Z | f21_sm40_cap4_size8e9_e | control for cap5+8e9 | CANCELLED/RETIRED (not a variant failure) | CLOSED
    - 2026-10-21T21:56:37Z | f21_sm40_cap5_size8e9 | cap5 + size8e9 | CANCELLED/RETIRED (not a variant failure) | CLOSED

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_sm40_cap4_size8e9.py](code/cand_sm40_cap4_size8e9.py)
- [code/cand_sm40_cap5_size7e9.py](code/cand_sm40_cap5_size7e9.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7566](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7566)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7563](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7563)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7572](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7572)
- [VM_SWEEP_RESULTS_2026-10-21.md#L96](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L96)
- [VM_SWEEP_RESULTS_2026-10-21.md#L97](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L97)
- [VM_SWEEP_RESULTS_2026-10-21.md#L98](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L98)
- [VM_SWEEP_RESULTS_2026-10-21.md#L99](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L99)
- [VM_SWEEP_RESULTS_2026-10-21.md#L100](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L100)
- [VM_SWEEP_RESULTS_2026-10-21.md#L101](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L101)
- [VM_SWEEP_RESULTS_2026-10-21.md#L102](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L102)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-7e9-read-size-threshold-five-reads-more-expensive-than) · [Copy provenance](../../../PROVENANCE.json)

