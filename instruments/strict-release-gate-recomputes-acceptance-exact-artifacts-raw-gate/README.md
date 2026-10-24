# Strict release gate recomputes acceptance from exact artifacts and raw gate evidence

2026-10-10T23:22Z · tool · adopted · measured

Astra's audit of the graded candidate generation 46 (S46) pair verified exact archive and estimator SHAs and every raw stage A / C / 1.3x-slow gate, and found a latent hole in chain42: an INCOMPLETE worker verdict with spoiled pair notes could have submitted, and not all raw stage C memory and swap gates were checked. release_gate_v1.py (read-only) checks all raw gates, exact archive and source identities and the exact verdict (5 regression tests); release_gate_v2.py recomputes the designation judge from the exact pair / slow / worker paths with explicit primary mode, requires any old report to equal fresh output byte for byte and records source / evidence / report SHA provenance (8 tests). Both S46 files pass for release only (stage D not run).

Recorded measurement:

    S46 safe #335259 100/100, max setup 4.321 s, max residual 0.3429 s, max roundtrip 93.1 s; twin #335261 4.319 s / 0.3588 s / 87.0 s; stage C footprint about 3.0-3.1 GB; v1 5 tests PASS, v2 8 tests PASS

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/test_release_gate_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/test_release_gate_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/s46_worker_pair_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/s46_worker_pair_v1.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/_stale/designation_judge_before_first_priced_20261011T1810Z.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/_stale/designation_judge_before_first_priced_20261011T1810Z.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/loss_risk.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/loss_risk.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/native_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/native_probe.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/own_gate.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/own_gate.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_b_native40.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_b_native40.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_d_serving.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_d_serving.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L734](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L734)
- [TEAM_LOG_2026-10-01_to_10-22.md#L745](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L745)

<details>
<summary>All result files and records (8)</summary>

- [s46_eprimary_gate.json](s46_eprimary_gate.json) · result
- [s46_eprimary_postgrade_judge.txt](s46_eprimary_postgrade_judge.txt) · result
- [s46_safe_pz_gate.json](s46_safe_pz_gate.json) · result
- [s46_safe_pz_postgrade_judge.txt](s46_safe_pz_postgrade_judge.txt) · result
- [S47_RELEASE_v1.md](S47_RELEASE_v1.md) · result
- [s46_eprimary_gate-8b3c1ae8.json](s46_eprimary_gate-8b3c1ae8.json) · result
- [s46_safe_pz_gate-f2745a79.json](s46_safe_pz_gate-f2745a79.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-10-strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate) · [Copy provenance](../../PROVENANCE.json)

