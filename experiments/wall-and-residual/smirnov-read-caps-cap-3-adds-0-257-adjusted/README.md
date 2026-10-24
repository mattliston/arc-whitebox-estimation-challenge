# Smirnov read caps: cap 3 adds +0.257% adjusted cost and reduces wall time -11.0%

2026-10-21T16:05:44Z · experiment · adopted · measured

A per-prediction counter returns the fallback after a fixed number of grouped strided reads. The paired cap curve favored three reads for wall reduction per adjusted cost; four reads later combined with a matrix-size threshold.

Recorded measurement:

    - 2026-10-21T16:05:44Z [fable-lead] CAP PRICE CURVE (paired vs concurrent control, 32 ranks): cap4 adj +0.197% bill +0.180% raw +0.017% wall -5.3% resid -3.4%; cap3 adj +0.257% bill +0.359% raw -0.102% wall -11.0%; cap2 adj +0.382% bill +0.518% raw -0.135% wall -12.4% resid -3.9%; cap0 adj +0.668% bill +0.781% raw -0.112% wall -10.2% resid -1.8%. cap3 = best wall per adjusted (-11% wall: slowest public net 114.9 -> ~102 s, survives +15% burst) for +0.26%. Fits in held evaluation (16:05Z).
    - 2026-10-21T15:29:25Z [fable-lead] WALL LEVER FOUND: sm40_cap4 (tail: _Sm40CapPlans returning None after 4 Smirnov reads per predict; counter resets per predict at line 19042) paired vs concurrent control on 12 ranks: wall -9.5% (median), backend -10%, residual -4.6%, flops +0.2%, requests +0.16%, puts -0.6%; accuracy pending. Stack s1p5_cap4 built (sci d1def756 / rel ce80c06f = s1.5 refit + cap4 tail) and launched: VM32 s1p5_cap4_native32_v1 + concurrent control ctl_s1p5_cap4_native32_v1 + fresh12 (original8/extension4) at 15:26:53Z; Mac32 started 15:29Z. Teammate building cap3/cap2/cap1 pairs. Extended (squares) and extended+gate fits running unguarded BLAS4 since 15:20/15:23Z.
    - Margin lever:  tail (lead/runs/fable_oct21_final_v1/tails/sm40_cap4.py; _Sm40CapPlans dict wrapper returns None after N Smirnov young reads per predict; counter resets per predict). cap4 paired vs concurrent control: wall -9.5%, backend -10%, residual -4.6%, flops +0.2%, requests +0.16%, puts -0.6% (12 ranks); accuracy pending from teammate; cap3/cap2/cap1 pairs queued. sm40_off (setup failure) and smx_off (wall +11.6%, adj +0.5%) CLOSED.
    - 2026-10-21T15:31:56Z [fable-lead] sm40_cap4 FULL 32 paired vs concurrent control (STORE_sm40_cap4_vs_ctl_v1.json): adjusted +0.197% (SE 0.054), held25 +0.225%, raw +0.017% (accuracy unchanged, 15/32 lower), bill +0.180%, requests +0.09%, puts -0.70%, wall -5.26% (SE 1.2, 24/32), residual -3.39% (24/32), 4/32 byte-identical. Rate ~0.035% bill per 1% wall; cap0 (all reads off at predict time, setup intact) projected ~+0.55% bill for ~-15% wall.
    - Cap price curve (paired vs concurrent controls, 32 ranks): cap4 adj +0.20% (bill +0.18, raw +0.02) wall -5.3% resid -3.4%; cap3 adj +0.26% (bill +0.36, raw -0.10) wall -11.0%; cap2 adj +0.38% (bill +0.52) wall -12.4%; cap0 adj +0.67% (bill +0.78) wall -10.2%. cap3 = best margin per adjusted. Tails: lead/runs/fable_oct21_final_v1/tails/sm40_cap{4,3,2,0}.py (cap0 uses the teammate's predict-only 'retired' gate).
    Sweep result detail: - 2026-10-21T15:31:00Z | f21_sm40_cap4 | class _Sm40CapPlans(dict): ;     def get(self, key, default=None): ;         if _ASTRA_SM40_STATS.get('calls', 0) >= _SM40_CAP: ;             return None ;      | ref f21_ctl_sm40_cap4 | all32 adj +0.197% (se 0.054) | held25 adj +0.225% (se 0.067) | wins 5/32 | raw +0.017% | bill +0.180% | requests +0.09% | puts -0.70% | put_bytes_f4 -0.77% | wall -5.26% | max-rank wall +8.23% | slow10 wall -9.51% req +0.16% adj +0.168% | sm40 calls 4.0 vs 6.125 | byte-identical n (4/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap4 | sha256 40d00aed4da79e1bfb3e28e7feca4560d69a4f808521319b63cbd42fada4588c | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap4.py
    - 2026-10-21T15:52:37Z | f21_sm40_cap3 | class _Sm40CapPlans(dict): ;     def get(self, key, default=None): ;         if _ASTRA_SM40_STATS.get('calls', 0) >= _SM40_CAP: ;             return None ;      | ref f21_ctl_sm40_cap3 | all32 adj +0.257% (se 0.076) | held25 adj +0.287% (se 0.083) | wins 10/32 | raw -0.102% | bill +0.359% | requests +0.33% | puts -0.65% | put_bytes_f4 -0.28% | wall -10.97% | backend -11.18% | residual -3.58% | max-rank wall -1.12% | slow10 wall -11.78% req +0.45% adj +0.247% | sm40 calls 3.0 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap3 | sha256 d10bcbba67a63e75ba02295ee19fecd9334c9d478aeef2e66ed3dc939b4f3776 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap3.py
    - 2026-10-21T16:07:09Z | f21_sm40_cap0 | class _Sm40CapPlans(dict): ;     def get(self, key, default=None): ;         if _ASTRA_SM40_STATS.get('retired') is False and _ASTRA_SM40_STATS.get('calls', 0)  | ref f21_ctl_sm40_cap0 | all32 adj +0.668% (se 0.093) | held25 adj +0.698% (se 0.098) | wins 2/32 | raw -0.112% | bill +0.781% | requests +0.23% | puts -1.12% | put_bytes_f4 -0.28% | wall -10.24% | backend -11.08% | residual -1.81% | max-rank wall +2.09% | slow10 wall -10.60% req +0.70% adj +0.729% | sm40 calls 0.0 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap0 | sha256 86fc39c9176f8b50bc8c8ae5012cf8af3e3174315a5dc600af8248fb528fe1aa | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap0.py
    - 2026-10-21T16:07:13Z | f21_sm40_cap2 | class _Sm40CapPlans(dict): ;     def get(self, key, default=None): ;         if _ASTRA_SM40_STATS.get('calls', 0) >= _SM40_CAP: ;             return None ;      | ref f21_ctl_sm40_cap2 | all32 adj +0.382% (se 0.073) | held25 adj +0.433% (se 0.084) | wins 7/32 | raw -0.135% | bill +0.518% | requests +0.43% | puts -0.84% | put_bytes_f4 -0.19% | wall -12.42% | backend -12.84% | residual -3.86% | max-rank wall -2.83% | slow10 wall -10.41% req +0.66% adj +0.347% | sm40 calls 2.0 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap2 | sha256 89173652cfa07b9a492321b07b2a42ad1715f8159076164f80db4105364baff5 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap2.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/tails/sm40_cap4.py](code/tails/sm40_cap4.py)
- [code/tails/sm40_cap3.py](code/tails/sm40_cap3.py)
- [code/tails/sm40_cap2.py](code/tails/sm40_cap2.py)
- [code/tails/sm40_cap0.py](code/tails/sm40_cap0.py)
- [code/cand_sm40_cap4.py](code/cand_sm40_cap4.py)
- [code/cand_sm40_cap3.py](code/cand_sm40_cap3.py)
- [code/cand_sm40_cap0.py](code/cand_sm40_cap0.py)
- [code/cand_sm40_cap2.py](code/cand_sm40_cap2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7462](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7462)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7445](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7445)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7451](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7451)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7455](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7455)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7466](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7466)
- [VM_SWEEP_RESULTS_2026-10-21.md#L26](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L26)
- [VM_SWEEP_RESULTS_2026-10-21.md#L34](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L34)
- [VM_SWEEP_RESULTS_2026-10-21.md#L38](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L38)
- [VM_SWEEP_RESULTS_2026-10-21.md#L39](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L39)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

## Related work

- shipped in: [Submission #337565: 705-column gate readout at strength 1.5 and grouped-read cap 3](../../../submissions/337565/README.md)
- shipped in: [Submission #337617: Hidden-times-gate readout and cap 3 with direct gather](../../../submissions/337617/README.md)
- shipped in: [Submission #337619: 961-column on-policy readout, grouped-read cap 4 and 8e9 size threshold](../../../submissions/337619/README.md)
- shipped in: [Submission #337641: Final 961-column readout at strength 1.35 with residual-meter references removed](../../../submissions/337641/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-smirnov-read-caps-cap-3-adds-0-257-adjusted) · [Copy provenance](../../../PROVENANCE.json)

