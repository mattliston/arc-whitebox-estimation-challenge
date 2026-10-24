# Setup closure-cycle removal and heap freezing cut maximum full-collection pause by 58.90%

2026-10-17T07:27:52Z · experiment · adopted · measured

Breaking setup-owned closure cycles and freezing the live setup graph shortened full-generation scans while retaining automatic young collection. Ten natural-lifecycle predictions preserved bytes and FLOPs, but the original remote spike cause did not reproduce.

Recorded measurement:

    Maximum full-GC time 26.267→10.797 ms (−58.90%); mean residual instructions −.282%; concurrent starts 4.920–4.940 s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_crunch_runtime_v1/failure336657_v1/build_review_v22b_v1.py](code/lead/runs/astra_crunch_runtime_v1/failure336657_v1/build_review_v22b_v1.py)
- [code/lead/runs/astra_crunch_runtime_v1/failure336657_v1/candidate_setup_freeze_v22b.py](code/lead/runs/astra_crunch_runtime_v1/failure336657_v1/candidate_setup_freeze_v22b.py)
- [code/release/runs/a_astra_runtime_v22b_1fdc8d9b/payload/estimator.py](code/release/runs/a_astra_runtime_v22b_1fdc8d9b/payload/estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4837](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4837)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4839](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4839)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

## Related work

- shipped in: [Submission #336678: freeze the setup object graph: one public residual failure](../../../submissions/336678/README.md)

[Research graph](../../../graph/index.html#node=2026-10-17-setup-closure-cycle-removal-heap-freezing-cut-maximum-full) · [Copy provenance](../../../PROVENANCE.json)

