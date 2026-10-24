# Nine essential setup fusions: forced-skip requests fall by 870 with limited residual improvement

2026-09-30 · experiment · inconclusive · measured

Nine depth-three entries were fused early under the same setup deadline; forced omission of the deferred phase preserved outputs and bill but barely changed residual time.

Recorded measurement:

    Normalsetup3.335/3.357/3.376s;649 fused/9 essential;forced requests-870/network;residual.1715/.1612 ->.1682/.1617;all16 and FLOPs exact.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/build.py](code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/build.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/probe_client.py](code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/probe_client.py)
- [code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/probe_driver.py](code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/probe_driver.py)
- [code/codex_resume_20260926_v2/hill_climb_scatter_forward_gather_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_scatter_forward_gather_v1/candidate.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

## Evidence

- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L150](../all-setup-keys-fused-eagerly-degraded-setup-drops-bindings/HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L150)

<details>
<summary>All result files and records (3)</summary>

- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z](../all-setup-keys-fused-eagerly-degraded-setup-drops-bindings/HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z) · record
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-nine-essential-setup-fusions-forced-skip-requests-fall-870) · [Copy provenance](../../../PROVENANCE.json)

