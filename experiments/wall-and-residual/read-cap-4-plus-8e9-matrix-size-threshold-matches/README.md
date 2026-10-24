# Read cap 4 plus an 8e9 matrix-size threshold matches cap-3 slow-network margin with fewer FLOPs

2026-10-21T18:57:00Z · experiment · adopted · measured

Dropping the largest grouped-read plans and allowing four reads reduced wall and residual time with little adjusted cost. A concurrent comparison against the three-read cap retained a large runtime outlier but supported comparable slow-network margin and lower FLOPs.

Recorded measurement:

    - 2026-10-21T18:57:00Z [fable-lead] Teammate size8e9 pair (drop 9/167 SM40 plans >= 8e9) vs concurrent ctl: adj -0.021% (held25 +0.003), bill +0.067%, wall -4.69% mean, slow10ctl 0.976, residual -4.19%, 12/32 byte-identical = a free margin lever. GO given for cap4+8e9 vs cap4, then cap3+8e9 vs cap3 (concurrent pairs); dial hunt queued behind. 7e9 pair killed for memory.
    - 2026-10-21T19:53:23Z [fable-lead] TEAMMATE: cap4+size8e9 vs cap4 (concurrent): adj -0.059%, bill +0.018%, wall -5.67%, slow10 0.948, residual -5.23% -> combined vs no cap: slow10 0.923x0.948=0.875 (cap3 0.866) at bill +0.18-0.06% vs cap3's +0.36%: cap4+8e9 = cap3 margin for ~0.14% less -> NEW MARGIN TAIL (tails/sm40_cap4_size8e9.py). cap3+8e9 = no-op (29/32 byte-identical). Dials: np_dn -0.096% (held25 -0.062), np_up +0.08, k4band +-; np_dn2 queued; k4lc running. Built hg_r0p03_s15_cap4s8e9 (hg seal x1.5 + cap4 + 8e9): VM32+fresh launched 2026-10-21T19:53:23Z; Mac32 queued after the hgfh chain's Mac32; direct cap4+8e9 vs cap3 wall pair requested.
    - 2026-10-21T20:26:52Z [fable-lead] TEAMMATE fire-evidence pair cap4+8e9 vs cap3 (concurrent, through the memory crunch): adj -0.118% (held25 -0.114), bill -0.160%, wall mean +6.2% (noisy; max-rank +134% one spike), slow10ctl 0.901 with 8/10 ranks at 0.95-1.03 -> cap4+8e9 wall = cap3 wall within noise on the slow nets (same margin class as #337565/#337617), 0.16% cheaper. DIAL: k4lc_up (_K4L_C x1.05) adj -0.144% (held25 -0.179, 17/32, bill 0): near lead candidate; k4lc_up2 (x1.10) running (ETA 20:38Z); np_dn2, k4band3 running; k4lc_dn re-queued.
    Sweep result detail: - 2026-10-21T18:45:10Z | f21_sm40_size8e9 | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) >= 3 and k[0] * k[1] * k[2] >= 8.0e9)} | ref f21_ctl_sm40_size8e9 | all32 adj -0.021% (se 0.047) | held25 adj +0.003% (se 0.052) | wins 11/32 | raw -0.088% | bill +0.067% | requests +0.08% | puts -0.17% | put_bytes_f4 -0.29% | wall -4.69% | backend -4.70% | residual -4.19% | max-rank wall +0.82% | slow10 wall -3.02% req +0.13% adj -0.044% | slow10ctl ratio 0.976 | sm40 calls 5.5 vs 6.125 | byte-identical n (12/32) | VM run /home/arcworker/arc-runs/f21_sm40_size8e9 | sha256 dfccfa1e826ed6c49f8b268f71f9f6a67e01b8db56c6467071226de04e8fcec1 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_size8e9.py
    - 2026-10-21T19:21:39Z | f21_sm40_cap4_ctl2 | class _Sm40CapPlans(dict): ;     def get(self, key, default=None): ;         if _ASTRA_SM40_STATS.get('calls', 0) >= _SM40_CAP: ;             return None ;      | ref first128_paid_nocensus_native32_v1 | all32 adj +0.197% (se 0.054) | held25 adj +0.225% (se 0.067) | wins 5/32 | raw +0.017% | bill +0.180% | requests +0.09% | puts -0.70% | put_bytes_f4 -0.77% | wall +43.06% | backend +41.89% | residual +34.42% | max-rank wall +69.48% | slow10 wall +48.43% req +0.16% adj +0.168% | slow10ctl ratio 1.484 | sm40 calls 4.0 vs 6.125 | byte-identical n (4/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap4_ctl2 | sha256 40d00aed4da79e1bfb3e28e7feca4560d69a4f808521319b63cbd42fada4588c | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap4.py
    - 2026-10-21T19:21:42Z | f21_sm40_cap4_size8e9 | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) >= 3 and k[0] * k[1] * k[2] >= 8.0e9)} ; class _Sm40CapPlans(dict): ;     def get | ref f21_sm40_cap4_ctl2 | all32 adj -0.059% (se 0.042) | held25 adj -0.052% (se 0.048) | wins 8/32 | raw -0.077% | bill +0.018% | requests +0.04% | puts -0.00% | put_bytes_f4 +0.05% | wall -5.67% | backend -5.49% | residual -5.23% | max-rank wall +40.40% | slow10 wall +0.44% req +0.07% adj -0.018% | slow10ctl ratio 0.948 | sm40 calls 3.9375 vs 4.0 | byte-identical n (20/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap4_size8e9 | sha256 7eb1d130dddd888bdcbcf45253006a567f1f8434e11c43c64d93c8187b40a4d5 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap4_size8e9.py
    - 2026-10-21T19:29:51Z | f21_sm40_cap3_ctl2 | class _Sm40CapPlans(dict): ;     def get(self, key, default=None): ;         if _ASTRA_SM40_STATS.get('calls', 0) >= _SM40_CAP: ;             return None ;      | ref first128_paid_nocensus_native32_v1 | all32 adj +0.257% (se 0.076) | held25 adj +0.287% (se 0.083) | wins 10/32 | raw -0.102% | bill +0.359% | requests +0.33% | puts -0.65% | put_bytes_f4 -0.28% | wall +5.70% | backend +5.47% | residual +10.32% | max-rank wall +14.07% | slow10 wall +2.75% req +0.45% adj +0.247% | slow10ctl ratio 1.028 | sm40 calls 3.0 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap3_ctl2 | sha256 d10bcbba67a63e75ba02295ee19fecd9334c9d478aeef2e66ed3dc939b4f3776 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap3.py
    - 2026-10-21T19:29:54Z | f21_sm40_cap3_size8e9 | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) >= 3 and k[0] * k[1] * k[2] >= 8.0e9)} ; class _Sm40CapPlans(dict): ;     def get | ref f21_sm40_cap3_ctl2 | all32 adj +0.005% (se 0.014) | held25 adj +0.006% (se 0.017) | wins 1/32 | raw +0.004% | bill +0.001% | requests -0.01% | puts -0.01% | put_bytes_f4 -0.05% | wall -0.11% | backend +0.03% | residual -2.86% | max-rank wall +8.98% | slow10 wall +7.75% req +0.01% adj -0.002% | slow10ctl ratio 1.023 | sm40 calls 3.0 vs 3.0 | byte-identical n (29/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap3_size8e9 | sha256 1b95275bb6a48f6ab8b3eaa94ea3b724cae92a1dc4e3f381dcfd00649c1b0100 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap3_size8e9.py
    - 2026-10-21T20:25:07Z | f21_sm40_cap3_b | class _Sm40CapPlans(dict): ;     def get(self, key, default=None): ;         if _ASTRA_SM40_STATS.get('calls', 0) >= _SM40_CAP: ;             return None ;      | ref first128_paid_nocensus_native32_v1 | all32 adj +0.257% (se 0.076) | held25 adj +0.287% (se 0.083) | wins 10/32 | raw -0.102% | bill +0.359% | requests +0.33% | puts -0.65% | put_bytes_f4 -0.28% | wall +42.89% | backend +41.67% | residual +38.59% | max-rank wall +78.47% | slow10 wall +44.98% req +0.45% adj +0.247% | slow10ctl ratio 1.450 | sm40 calls 3.0 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap3_b | sha256 d10bcbba67a63e75ba02295ee19fecd9334c9d478aeef2e66ed3dc939b4f3776 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap3.py
    - 2026-10-21T20:25:11Z | f21_sm40_cap4_size8e9_b | _ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) >= 3 and k[0] * k[1] * k[2] >= 8.0e9)} ; class _Sm40CapPlans(dict): ;     def get | ref f21_sm40_cap3_b | all32 adj -0.118% (se 0.059) | held25 adj -0.114% (se 0.072) | wins 17/32 | raw +0.042% | bill -0.160% | requests -0.20% | puts -0.06% | put_bytes_f4 -0.43% | wall +6.20% | backend +6.29% | residual +4.96% | max-rank wall +134.48% | slow10 wall +0.87% req -0.21% adj -0.095% | slow10ctl ratio 0.901 | sm40 calls 3.9375 vs 3.0 | byte-identical n (2/32) | VM run /home/arcworker/arc-runs/f21_sm40_cap4_size8e9_b | sha256 7eb1d130dddd888bdcbcf45253006a567f1f8434e11c43c64d93c8187b40a4d5 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_sm40_cap4_size8e9.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/tails/sm40_cap4_size8e9.py](code/tails/sm40_cap4_size8e9.py)
- [code/cand_sm40_size8e9.py](code/cand_sm40_size8e9.py)
- [code/cand_sm40_cap4.py](code/cand_sm40_cap4.py)
- [code/cand_sm40_cap4_size8e9.py](code/cand_sm40_cap4_size8e9.py)
- [code/cand_sm40_cap3.py](code/cand_sm40_cap3.py)
- [code/cand_sm40_cap3_size8e9.py](code/cand_sm40_cap3_size8e9.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7528](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7528)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7534](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7534)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7546](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7546)
- [VM_SWEEP_RESULTS_2026-10-21.md#L73](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L73)
- [VM_SWEEP_RESULTS_2026-10-21.md#L77](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L77)
- [VM_SWEEP_RESULTS_2026-10-21.md#L78](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L78)
- [VM_SWEEP_RESULTS_2026-10-21.md#L79](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L79)
- [VM_SWEEP_RESULTS_2026-10-21.md#L80](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L80)
- [VM_SWEEP_RESULTS_2026-10-21.md#L87](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L87)
- [VM_SWEEP_RESULTS_2026-10-21.md#L88](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L88)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

## Related work

- shipped in: [Submission #337619: 961-column on-policy readout, grouped-read cap 4 and 8e9 size threshold](../../../submissions/337619/README.md)
- shipped in: [Submission #337641: Final 961-column readout at strength 1.35 with residual-meter references removed](../../../submissions/337641/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-read-cap-4-plus-8e9-matrix-size-threshold-matches) · [Copy provenance](../../../PROVENANCE.json)

