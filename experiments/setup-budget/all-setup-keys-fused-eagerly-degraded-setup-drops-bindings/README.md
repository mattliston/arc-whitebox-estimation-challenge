# All setup keys fused eagerly: degraded setup drops bindings and increases billed compute

2026-09-30 · experiment · rejected · measured

All649 keys were made eager under the same deadline. A three-second degraded-setup fixture retained only593 bindings and failed the strict FLOP-equality gate, so no release qualification followed.

Recorded measurement:

    Full-skip requests-4423/-4252;degraded593bound/593fused versus649bound/0fused;extra144160303F(+.05476397367%) and89862794F(+.03405111989%);all16 outputs exact.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_all_eager_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_all_eager_fusion_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_all_eager_fusion_v1/build.py](code/codex_resume_20260926_v2/hill_climb_all_eager_fusion_v1/build.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/probe_client.py](code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/probe_client.py)
- [code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/probe_driver.py](code/codex_resume_20260926_v2/hill_climb_essential_fusion_v1/probe_driver.py)
- [code/codex_resume_20260926_v2/hill_climb_scatter_forward_gather_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_scatter_forward_gather_v1/candidate.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

## Evidence

- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L155](HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L155)

<details>
<summary>All result files and records (3)</summary>

- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z](HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z) · record
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-all-setup-keys-fused-eagerly-degraded-setup-drops-bindings) · [Copy provenance](../../../PROVENANCE.json)

