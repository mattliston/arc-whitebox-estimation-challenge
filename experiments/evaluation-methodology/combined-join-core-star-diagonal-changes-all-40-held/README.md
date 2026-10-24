# Combined join-core and star-diagonal changes: all 40 held-out networks improve adjusted score

2026-09-26 · experiment · adopted · measured

The final combined source passed matched native validation and client/server checks. Its gain was primarily lower compute, with no statistically clear raw-accuracy gain.

Recorded measurement:

    VALID40 versus #332361: adjusted -2.62273%, paired bootstrap CI [-2.81987%, -2.42397%], 40/40 wins; raw -0.16104%, CI [-0.34893%, +0.02528%]; utilization -2.46845%; CPU +14.33189%; setup 2.341 seconds; 489 intended keys bound; maximum group RSS 6,044,893,184 bytes; zero swap growth.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v1/candidate_bundle_diagreuse_v1.py](code/codex_resume_20260926_v1/candidate_bundle_diagreuse_v1.py)
- [code/fable_fire_v1/fire_candS28/candS28_S27_refit_ks_cap_sched_g5_mem.py](code/fable_fire_v1/fire_candS28/candS28_S27_refit_ks_cap_sched_g5_mem.py)
- [code/fable_fire_v1/fire_candS30/candS30_S29_core_star_reuse_mem.py](code/fable_fire_v1/fire_candS30/candS30_S29_core_star_reuse_mem.py)

## Evidence

- [RESULTS.md#L40](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md#L40)

<details>
<summary>All result files and records (3)</summary>

- [RESULTS.md](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md) · record
- [final_vs_s28_valid40_v1.json](final_vs_s28_valid40_v1.json) · result
- [FIRE_CHECKS_v1.json](FIRE_CHECKS_v1.json) · result

</details>

## Related work

- shipped in: [Submission #332492: approximate join core, shared star diagonal and refitted learned correction](../../../submissions/332492/README.md)

[Research graph](../../../graph/index.html#node=2026-09-26-combined-join-core-star-diagonal-changes-all-40-held) · [Copy provenance](../../../PROVENANCE.json)

