# Seven-level strip-push variants rejected: gains -0.034% and -0.010% are too small

2026-10-11T20:45:46Z · experiment · rejected · measured

Three strip-lever swaps were priced at platform pace in the candidate generation 48 (S48) configuration against the submitted seven-level full-packet strip push (HZ7R) (seven-level Winograd strip pushes on full 2048-column packets with a 3% gain floor): seven-level strip push (HZ7) (every packet including the partial last one) +310 / +7 requests, metered FLOPs +0.055% / -0.040% on networks 35 / 100; HZ7g0.0 (every packet, any gain) +1,053 / +703, +0.069% / -0.027%; HZ7Rg0.0 (full packets, any gain) +731 / +243, -0.033% / -0.008%. A steady-state session on networks 35 / 100 / 37 / 39 put HZ7 at about -0.025% of metered FLOPs on warm networks for +281 first-network requests (about -0.02% overall); strips were judged not worth a dial in either file and the gain-floor dial was dropped from the twin.

Recorded measurement:

    HZ7 +310 / +7 requests, bill +0.055% / -0.040%; HZ7g0.0 +1,053 / +703, +0.069% / -0.027%; HZ7Rg0.0 +731 / +243, -0.033% / -0.008%; averaged over 6 first + 94 warm: HZ7 -0.034%, HZ7Rg0.0 -0.010%, HZ7g0.0 -0.021%; steady state S48b vs S48b + HZ7: requests +281 / +7 / +10 / +0, bill +0.027% / -0.040% / -0.039% / 0.000%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/engineer/a62_modules/run_boundary_counts_v2.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/run_boundary_counts_v2.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s48hz7_sd_rc/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s48hz7_sd_rc/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s48hz7g0_sd_rc/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s48hz7g0_sd_rc/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s48hz7rg0_sd_rc/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s48hz7rg0_sd_rc/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/chain_hz7_counts_v1.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/chain_hz7_counts_v1.sh)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_readcol_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_readcol_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_dir_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_dir_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s48/s48b/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s48/s48b/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s48bhz7/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s48bhz7/candidate.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/opus_reexam_20261001_v1/engineer/stack/rl/s48hz7_build/candidate.py](code/opus_reexam_20261001_v1/engineer/stack/rl/s48hz7_build/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/stack/rl/s48hz7g0_build/candidate.py](code/opus_reexam_20261001_v1/engineer/stack/rl/s48hz7g0_build/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/stack/rl/s48hz7rg0_build/candidate.py](code/opus_reexam_20261001_v1/engineer/stack/rl/s48hz7rg0_build/candidate.py)
- [code/opus_reexam_20261001_v1/release/boundary_client.py](code/opus_reexam_20261001_v1/release/boundary_client.py)
- [code/opus_reexam_20261001_v1/release/boundary_run.py](code/opus_reexam_20261001_v1/release/boundary_run.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/server_light.py](code/opus_reexam_20261001_v1/release/server_light.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1198](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1198)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1200](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1200)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1201](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1201)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1202](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1202)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1203](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1203)

<details>
<summary>All result files and records (28)</summary>

- [guard.json](guard.json) · result
- [launcher.json](launcher.json) · result
- [manifest.json](manifest.json) · result
- [rows.jsonl](rows.jsonl) · result
- [run.log](run.log) · log
- [server.json](server.json) · result
- [server.log](server.log) · log
- [server_marks.jsonl](server_marks.jsonl) · result
- [guard-a620f389.json](guard-a620f389.json) · result
- [launcher-c4862b1c.json](launcher-c4862b1c.json) · result
- [manifest-52c85b0e.json](manifest-52c85b0e.json) · result
- [rows-cbc13472.jsonl](rows-cbc13472.jsonl) · result
- [run-25a13a8d.log](run-25a13a8d.log) · log
- [server-06b783ed.json](server-06b783ed.json) · result
- [server-29829a04.log](server-29829a04.log) · log
- [server_marks-40a20720.jsonl](server_marks-40a20720.jsonl) · result
- [guard-55cba05c.json](guard-55cba05c.json) · result
- [launcher-786318fc.json](launcher-786318fc.json) · result
- [manifest-2e177456.json](manifest-2e177456.json) · result
- [rows-bc6a968e.jsonl](rows-bc6a968e.jsonl) · result
- [run-6fc9d2f2.log](run-6fc9d2f2.log) · log
- [server-b35b98d7.json](server-b35b98d7.json) · result
- [server-aaedddf6.log](server-aaedddf6.log) · log
- [server_marks-080a8157.jsonl](server_marks-080a8157.jsonl) · result
- [build.json](build.json) · result
- [build-e166e931.json](build-e166e931.json) · result
- [build-0f53a3f7.json](build-0f53a3f7.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-seven-level-strip-push-variants-rejected-gains-0-034) · [Copy provenance](../../../PROVENANCE.json)

