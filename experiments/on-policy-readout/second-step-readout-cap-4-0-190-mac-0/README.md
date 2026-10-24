# Second-step readout with cap 4: -0.190% Mac, -0.220% Linux and -0.018% fresh adjusted

2026-10-21T15:37:21Z · experiment · rejected · measured

The first-128 second-step readout with four permitted grouped reads improved fixed native panels but narrowly improved the fresh set. A two-read cap was too expensive, and the three-read version was nearly score-neutral before richer features were fitted.

Recorded measurement:

    - 2026-10-21T15:37:21Z [fable-lead] STACK s1p5_cap4 Mac32 vs 49bd: all32 -0.190% (SE 0.200, 16/32), held25 -0.308% (13/25), raw -0.370%, bill +0.180%; Mac projection 1.4034e-9 -> expected hosted ~1.4068e-9 with ~-5% wall. Additive with refit (-0.43 raw) and cap4 (+0.18 bill). Release gates running on its release file (no fire).
    - 2026-10-21T15:38:12Z [fable-lead] Release gates CLEAR for stack s1p5_cap4 (tag fable_s1p5_cap4_ce80c06f_v1, archive 578405b89c3a; setup max 3.36 s; prefire only gc) -> HELD, not fired (expected ~1.4068e-9, margin ~-5% wall: below Matt's 1.404 bar). Built stack s1p5_cap2 (tails/sm40_cap2.py); Mac32 running 15:38Z. VM memory tight (11 GB free) with stack VM32 + 2 fresh + teammate cap pairs + 2 fits.
    - 2026-10-21T15:40:36Z [fable-lead] fresh12 s1p5_cap4 vs 49bd: adjusted -0.018% (SE 0.253, 7/12), raw -0.13%, bill +0.113% (cap bill on fresh nets 0.08-0.13%). Fresh outliers (+1.4/+1.75 on seeds 7109005/7109009) carry over from the s1.5 refit.
    - 2026-10-21T15:45:52Z [fable-lead] STACK s1p5_cap2 Mac32 vs 49bd: all32 +0.162% (SE 0.199, 13/32), held25 +0.077%, raw -0.352%, bill +0.516% -> cap2 too expensive (reads 3-4 save the most FLOPs). Cap price curve: cap4 +0.18% bill (~-5% wall), cap2 +0.52%. Safe stack remains s1p5_cap4 unless the extended fits add >= -0.2%.
    - 2026-10-21T15:48:17Z [fable-lead] STACK s1p5_cap4 VM32 vs global 49bd: all32 -0.220% (SE 0.159, 20/32), held25 -0.282%, raw -0.400%, bill +0.180%; wall vs concurrent cap3 control -5.6% (backend -6.0%, slow10 -5.3%), residual -1.0% mean (-1.6% median), puts -0.70%. Pooled Mac/VM/fresh ~ -0.18% -> expected hosted ~1.4068e-9; projected max public wall ~108.6 s.
    - 2026-10-21T16:13:52Z [fable-lead] STACK s1p5_cap3 (sci 3da1cf06) Mac32 vs 49bd: all32 -0.032% (SE 0.200, 16/32), held25 -0.102%, raw -0.390%, bill +0.359% -> neutral score with -11% wall; needs the extended gain. VM32+fresh12 running.
    - 2026-10-21T16:26:35Z [fable-lead] STACK s1p5_cap3 VM32 vs 49bd: all32 -0.154% (SE 0.171, 17/32), held25 -0.253%, raw -0.511%, bill +0.359%; fresh12 +0.010% (SE 0.224, 7/12). With Mac32 -0.032%: pooled ~-0.08% -> ~1.4082e-9 at -11% wall (neutral score, safe profile).
    Sweep result detail: - 2026-10-21T18:06:43Z [fable-lead] Mac32 extgate_s15_small: all32 -0.033% (raw -0.471, bill +0.440), held25 -0.195%; VM32 all32 -0.134% held25 -0.262%. Status to Matt 11:07 PDT: no sub-1.3986 candidate yet; ridge-0.03 stacks pending.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/stage_candidate_v1.py](code/stage_candidate_v1.py)
- [code/score_fresh_candidate_v1.sh](code/score_fresh_candidate_v1.sh)
- [code/tails/sm40_cap2.py](code/tails/sm40_cap2.py)
- [code/build_stack_v1.py](code/build_stack_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7457](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7457)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7458](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7458)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7459](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7459)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7460](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7460)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7461](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7461)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7470](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7470)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7473](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7473)
- [VM_SWEEP_RESULTS_2026-10-21.md#L63](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L63)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [validate.json](validate.json) · result
- [stage_a.json](stage_a.json) · result
- [package.json](package.json) · result
- [validate_package.json](validate_package.json) · result
- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-second-step-readout-cap-4-0-190-mac-0) · [Copy provenance](../../../PROVENANCE.json)

