# Eleven runtime revisions reduce projected maximum residual from .430 s to .3989 s but nearly exhaust setup time

2026-10-17T02:31:21Z · finding · informational · measured

Eleven runtime versions of the frozen peel science were qualified on the 35/100 production boundary against the closest graded the estimator submitted as #336412 (S2tr) telemetry (#336412 including its .422 s failed maximum): v7 added 3.1% requests and projected a residual maximum of .4302 s; v11 (request diets, tight frames, P40) still needed 5.5% fewer residual instructions; v12 (owned-B pool, terminal pair-add) 3.8%; v13 (QL off, compute off, dm2, lazybind v2) 1.16%; v14 and v15 (setup relocation of main-read geometry) failed cold or warm; v16 passed warm and failed cold by 0.9 ms; v17 (P-only prebind, finish 160) passed both at .399850559 / .398886493 s with the setup at 4.988 s CPU-adjusted, and was fired as #336631. Every version kept predictions byte-identical to v13.

Recorded measurement:

    v7 vs S2tr: 35: 48,784 vs 47,336 requests (+3.059%), residual instructions +2.381%, bill -.16758%; 100: 46,742 vs 45,646 (+2.401%), +1.411%, bill -.47777%; projected mean resid .3615 / max .4302, worst wall 121.26 s. v11: cold 48,034 / 2.162852 G / 244.382389 GF; warm 45,910 / 2.058131 G / 243.406442 GF; needs -5.518% (116.5 M instructions); wall 111.585 / 112.325 s. v12: 47,932 / 45,289 requests, ratio 1.00673585 / 0.98512499, warm tail .415836264 s. v13: 46,564 / 44,151, ratio 0.97664516 / 0.95868643, warm tail .404676149 s, slow setup 4.802063 s (16,346 requests), bill 244,112,205,138 / 243,349,132,716. v14 warm .40536 s; v15 cold .4067885 / warm .4015622 s; v16 cold .400895617 / warm .399351432 s; v17 cold .399850559 / warm .398886493 s, hardware-high wall 113.733 s, setup 4.974484 + .013602 = 4.988086 s.

## Programs

- [candidate_s2tr_peel_release_v17.py](candidate_s2tr_peel_release_v17.py)
- [boundary_run_v11.py](boundary_run_v11.py)
- [../2026-10-17-depth-peeling-exact-cubic-messages-four-coefficients-sixteen-network/candidate_s2tr_peel_release_v11.py](../2026-10-17-depth-peeling-exact-cubic-messages-four-coefficients-sixteen-network/candidate_s2tr_peel_release_v11.py)
- [candidate_s2tr_peel_release_v12.py](candidate_s2tr_peel_release_v12.py)
- [boundary_run_v12.py](boundary_run_v12.py)
- [../../../experiments/setup-budget/two-exact-setup-offloads-terminal-smirnov-scale-batch-8/code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v13.py](../../../experiments/setup-budget/two-exact-setup-offloads-terminal-smirnov-scale-batch-8/code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v13.py)
- [boundary_run_v13.py](boundary_run_v13.py)
- [candidate_s2tr_peel_release_v14.py](candidate_s2tr_peel_release_v14.py)
- [boundary_run_v14.py](boundary_run_v14.py)
- [boundary_run_v17.py](boundary_run_v17.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [boundary_client_v11.py](boundary_client_v11.py)
- [boundary_client_v12.py](boundary_client_v12.py)
- [boundary_client_v13.py](boundary_client_v13.py)
- [boundary_client_v14.py](boundary_client_v14.py)
- [boundary_client_v17.py](boundary_client_v17.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4613](../../TEAM_LOG_2026-10-01_to_10-22.md#L4613)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4625](../../TEAM_LOG_2026-10-01_to_10-22.md#L4625)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4628](../../TEAM_LOG_2026-10-01_to_10-22.md#L4628)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4629](../../TEAM_LOG_2026-10-01_to_10-22.md#L4629)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4630](../../TEAM_LOG_2026-10-01_to_10-22.md#L4630)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4632](../../TEAM_LOG_2026-10-01_to_10-22.md#L4632)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4634](../../TEAM_LOG_2026-10-01_to_10-22.md#L4634)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4659](../../TEAM_LOG_2026-10-01_to_10-22.md#L4659)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4661](../../TEAM_LOG_2026-10-01_to_10-22.md#L4661)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4663](../../TEAM_LOG_2026-10-01_to_10-22.md#L4663)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4669](../../TEAM_LOG_2026-10-01_to_10-22.md#L4669)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4677](../../TEAM_LOG_2026-10-01_to_10-22.md#L4677)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4687](../../TEAM_LOG_2026-10-01_to_10-22.md#L4687)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4693](../../TEAM_LOG_2026-10-01_to_10-22.md#L4693)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4710](../../TEAM_LOG_2026-10-01_to_10-22.md#L4710)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4727](../../TEAM_LOG_2026-10-01_to_10-22.md#L4727)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4740](../../TEAM_LOG_2026-10-01_to_10-22.md#L4740)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4744](../../TEAM_LOG_2026-10-01_to_10-22.md#L4744)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4748](../../TEAM_LOG_2026-10-01_to_10-22.md#L4748)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4750](../../TEAM_LOG_2026-10-01_to_10-22.md#L4750)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4752](../../TEAM_LOG_2026-10-01_to_10-22.md#L4752)

<details>
<summary>All result files and records (12)</summary>

- [BOUNDARY_v7_RESULT.json](BOUNDARY_v7_RESULT.json) · result
- [PROJECTION_v7_S2TR.json](PROJECTION_v7_S2TR.json) · result
- [BOUNDARY_v11_RESULT.json](BOUNDARY_v11_RESULT.json) · result
- [PROJECTION_v11_S2TR.json](PROJECTION_v11_S2TR.json) · result
- [PROJECTION_v12_S2TR.json](PROJECTION_v12_S2TR.json) · result
- [PROJECTION_v13_S2TR.json](PROJECTION_v13_S2TR.json) · result
- [PROJECTION_v14_S2TR.json](PROJECTION_v14_S2TR.json) · result
- [HANDOFF_RUNTIME_v15.md](HANDOFF_RUNTIME_v15.md) · record
- [RELEASE_v17.json](RELEASE_v17.json) · record
- [stage_a.json](stage_a.json) · result
- [PROJECTION_v17_S2TR.json](PROJECTION_v17_S2TR.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Submission #336631: depth peeling, rank-4 exact kernel and read-plan setup: startup timeout](../../../submissions/336631/README.md)

[Research graph](../../../graph/index.html#node=2026-10-17-eleven-runtime-revisions-reduce-projected-maximum-residual-430-s) · [Copy provenance](../../../PROVENANCE.json)

