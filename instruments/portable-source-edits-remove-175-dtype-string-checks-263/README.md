# Portable source edits remove 175 dtype-string checks and 263 write-only diagnostic statements

2026-10-17T09:50:53Z · tool · adopted · measured

Two source-only helpers were prepared for the repair lineage: apply_dtype_v2 removes the str wrappers from 175 dtype equality checks (2,800 native and client tiny cases pass; about 4.09 ms per 3,054 checks), later carried as the dtype trim in v30; and a write-only scalar-diagnostics diet replaces 263 statements in 64 families by padded passes with every public arithmetic and view call preserved (no measured runtime gain claimed).

Recorded measurement:

    dtype v25d: 175 checks, native micro ~4.09 ms / 3,054 checks; v28d tiny equivalence 2,800 cases, saves 4.2 ms per 3,054 checks (~1.1% of a platform residual). Stats diet: 263 statements / 64 families, candidate 00df9545.

## Programs

- [code/astra_crunch_runtime_v1/failure336687_v1/candidate_current_meter_v25a.py](code/astra_crunch_runtime_v1/failure336687_v1/candidate_current_meter_v25a.py)
- [code/astra_crunch_physics_v1/stats_diet_v1/candidate_stats_scalar_v1.py](code/astra_crunch_physics_v1/stats_diet_v1/candidate_stats_scalar_v1.py)
- [code/astra_crunch_physics_v1/stats_diet_v1/apply_v1.py](code/astra_crunch_physics_v1/stats_diet_v1/apply_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4869](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4869)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4871](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4871)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4881](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4881)

<details>
<summary>All result files and records (4)</summary>

- [SOURCE_AUDIT_v1.json](SOURCE_AUDIT_v1.json) · record
- [HANDOFF_v1.json](HANDOFF_v1.json) · record
- [PAYLOAD_AUDIT_v1.json](PAYLOAD_AUDIT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-17-portable-source-edits-remove-175-dtype-string-checks-263) · [Copy provenance](../../PROVENANCE.json)

