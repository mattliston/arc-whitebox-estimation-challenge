# 705-column gate readout with cap 4: -0.378% pooled adjusted, held for limited runtime margin

2026-10-21T16:49:04Z · experiment · rejected · measured

The four-read cap retained more FLOP savings and improved all three measured panels, but its projected slowest hosted wall was closer to the limit. The gated source was held rather than submitted.

Recorded measurement:

    - 2026-10-21T16:49:04Z [fable-lead] extgate_s15_cap4 (HELD) all three sets: Mac32 -0.341 (held25 -0.588), VM32 -0.395 (held25 -0.540, raw -0.581, bill +0.188), fresh12 -0.436 (8/12); pooled -0.378% -> ~1.4039e-9; wall ~-4.8% (slowest ~109 s); gates clear archive e096a47b1d4f (tag fable_extgate_s15_cap4_b5a2d638_v1). Ridge/damping gate fits launched 16:48Z (r0.03/d0.5, r0.3/d0.5, r0.03/d1.0). Next: iteration-2 captures on the fired trajectory.
    - 2026-10-21T16:46:53Z [fable-lead] extgate_s15_cap4 (HELD per Matt) Mac32 vs 49bd: all32: adjusted -0.3407% (SE 0.2815, lower 18/32); raw -0.5254%, bill +0.1858%; legacy -0.3163 pp | heldout25: adjusted -0.5880% (SE 0.3330, lower 17/25); raw -0.7817%, bill +0.1950%; legacy; gates: 2026-10-21T16:46:37Z GATES CLEAR: archive e096a47b1d4f ready for fire_once
    - Bank bar after #337565 (if clean ~1.4067): <= ~1.4017e-9 with engineered margin; cap4-gate alone (1.4039) does not clear it.
    - Bank bar = 1.3986e-9 (0.005e-9 under BEST) = Matt's "under 1.40 with #337565's margin (cap3)". Reserve 3 shots for ~21:30-22:30Z; cap4-gate (~1.401 by held25 projection) prepped (FIRE_cap4_gate.sh) but no longer a bank. Hosted predictor: held25 pooled (#337565 local pooled -0.18 / held25 -0.36 -> hosted -0.40).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/FIRE_cap4_gate.sh](code/FIRE_cap4_gate.sh)
- [code/stacks/extgate_s15_cap4/candidate_extgate_s15_cap4_science_v1.py](code/stacks/extgate_s15_cap4/candidate_extgate_s15_cap4_science_v1.py)
- [code/tails/sm40_cap4.py](code/tails/sm40_cap4.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7489](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7489)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7488](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7488)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7496](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7496)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7514](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7514)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [validate.json](validate.json) · result
- [stage_a.json](stage_a.json) · result
- [package.json](package.json) · result
- [validate_package.json](validate_package.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-705-column-gate-readout-cap-4-0-378-pooled) · [Copy provenance](../../../PROVENANCE.json)

