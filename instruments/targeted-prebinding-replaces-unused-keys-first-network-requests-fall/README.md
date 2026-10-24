# Targeted prebinding replaces unused keys: first-network requests fall from 44,222 to 43,661

2026-10-11T20:30:50Z · tool · adopted · measured

The submitted 649-key setup prebind list had 140 keys that none of 20 candidate generation 48 (S48) census networks asks for; swapping them for the read column cut's lazily bound shapes (fragment_prebind_s48b_v2.py, fit on 20 census networks, networks 35 / 100 out of sample) cut in-sample lazy binds from 0.23 / 0.22% to 0.10 / 0.12% of the metered FLOPs and first-network lazy keys from 46 to 14-16, putting the first network back under the 43,800 line at no setup cost. S48b (= S48 + this list; predictions byte-identical) was never submitted itself but became the base of the candidate generation 49 (S49) / candidate generation 50 (S50) files and the recipe (prebind_targeted_v2.py on a file's own key census) for every later twin.

Recorded measurement:

    first network 35: 43,661 requests (S48 44,222: -561; S47e 43,596), warm network 100: 41,702 (S48 41,971: -269); bill 251.4407 / 251.4019 G = -0.051% / -0.006% vs S48; setup 18,397 requests, 592 of 649 keys bound, last phase 4.123 s (S48 4.122 s); lazy binds 0.23 / 0.22% -> 0.10 / 0.12% of the bill; first-network lazy keys 46 -> 14-16; tree elements of the first 593 keys 26.44 -> 25.19 G

## Programs

- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_prebind_s48b_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_prebind_s48b_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s48/s48b/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s48/s48b/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/release/boundary_client.py](code/opus_reexam_20261001_v1/release/boundary_client.py)
- [code/opus_reexam_20261001_v1/release/boundary_run.py](code/opus_reexam_20261001_v1/release/boundary_run.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/server_light.py](code/opus_reexam_20261001_v1/release/server_light.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1192](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1192)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1198](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1198)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1253](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1253)

<details>
<summary>All result files and records (9)</summary>

- [guard.json](guard.json) · result
- [launcher.json](launcher.json) · result
- [manifest.json](manifest.json) · result
- [rows.jsonl](rows.jsonl) · result
- [run.log](run.log) · log
- [server.json](server.json) · result
- [server.log](server.log) · log
- [server_marks.jsonl](server_marks.jsonl) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-11-targeted-prebinding-replaces-unused-keys-first-network-requests-fall) · [Copy provenance](../../PROVENANCE.json)

