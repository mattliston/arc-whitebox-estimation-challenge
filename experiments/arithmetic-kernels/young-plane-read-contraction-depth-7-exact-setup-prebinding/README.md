# Young plane read at contraction depth 7 with exact setup prebinding: two-network -0.48%, science-24 -0.098%, adopted

2026-10-14T23:20:49Z · experiment · adopted · measured

The dominant young-read product was moved to a depth-7 contraction (YR7). The first version recovered only part of the modeled saving because the new shapes bound cold at first use; an exact setup-prebinding repair recovered the rest at unchanged predictions. It passed the 24-network science and was fired as #336125.

Recorded measurement:

    v1 fixed 2: adjusted -0.14975%, bill -0.06216%, max prediction displacement 0.03125 of parent RMS error (cold R7 binding explained ~0.28% missing); v2 fixed 2: adjusted -0.4804907%, bill -0.3931867%, MSE -0.0876486%, prebind recoveries 797M/933M match oracle; pilot 8 -0.1721%; science 24: fresh 16 log net -0.0009525920, OOS 8 -0.0010378156, pooled -0.0009809999 (~0.09805%); seal 8 -0.08297%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_young_read7_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_ON_HELD_v2.py](code/candidate_ON_HELD_v2.py)
- [code/build_prebind_source_record_v2.py](code/build_prebind_source_record_v2.py)
- [code/candidate_OFF_HELD_v1.py](code/candidate_OFF_HELD_v1.py)
- [code/candidate_OFF_HELD_v2.py](code/candidate_OFF_HELD_v2.py)
- [code/candidate_ON_HELD_v1.py](code/candidate_ON_HELD_v1.py)
- [code/fragment_prebind_HELD_v2.py](code/fragment_prebind_HELD_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3150](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3150)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3152](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3152)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3159](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3159)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3163](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3163)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3171](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3171)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3176](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3176)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3192](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3192)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3207](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3207)

<details>
<summary>All result files and records (2)</summary>

- [DECISION_initial8_v1.json](DECISION_initial8_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #336125: young plane read at contraction depth 7 with exact setup prebinding on #336101, 1.50435e-9](../../../submissions/336125/README.md)
- ← shipped in: [Submission #336125: young plane read at contraction depth 7 with exact setup prebinding on #336101, 1.50435e-9](../../../submissions/336125/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-young-plane-read-contraction-depth-7-exact-setup-prebinding) · [Copy provenance](../../../PROVENANCE.json)

