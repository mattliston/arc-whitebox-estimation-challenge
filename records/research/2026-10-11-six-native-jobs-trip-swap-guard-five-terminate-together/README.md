# Six native jobs trip the swap guard: five terminate together and are requeued at concurrency two

2026-10-11T17:43:58Z · incident · failed · measured

Running six native processes on a box at load 50 grew system swap past the lifecycle guards' '256 MiB additional system swap limit', which stopped five runs at once at 17:39:40Z: the sealed S47e reference arm (2 of 32 networks, reference only), the carrier-strength refit-after-drop round (MS) certificate, the column stack's count session and two MS parity runs. The one-native-process rule had been broken; the work was re-queued as chain_s48_v6.sh with at most two processes (only the sealed pair runs two).

Recorded measurement:

    five runs stopped at 17:39:40Z under the '256 MiB additional system swap limit' at load 50; re-queued with <= 2 processes

## Programs

- [chain_s48_v6.sh](chain_s48_v6.sh)
- [../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [gate_rdz_v1.py](gate_rdz_v1.py)
- [candidate.py](candidate.py)
- [candidate-86af325b.py](candidate-86af325b.py)
- [candidate-5fd5a52e.py](candidate-5fd5a52e.py)
- [../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py](../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py)
- [../2026-10-11-combined-candidates-receive-one-sealed-comparison-against-submitted-estimator/candidate.py](../2026-10-11-combined-candidates-receive-one-sealed-comparison-against-submitted-estimator/candidate.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1137](../../TEAM_LOG_2026-10-01_to_10-22.md#L1137)

<details>
<summary>All result files and records (19)</summary>

- [chain_s48_v5.out](chain_s48_v5.out) · result
- [gate_rdz_s48_m_off_cert.log](gate_rdz_s48_m_off_cert.log) · log
- [gate_rdz_s48_m_off_cert_guard.json](gate_rdz_s48_m_off_cert_guard.json) · result
- [gate_rdz_sealed_s47e.log](gate_rdz_sealed_s47e.log) · log
- [gate_rdz_sealed_s47e_guard.json](gate_rdz_sealed_s47e_guard.json) · result
- [guard.json](guard.json) · result
- [manifest.json](manifest.json) · result
- [rows.jsonl](rows.jsonl) · result
- [run.log](run.log) · log
- [server.json](server.json) · result
- [server.log](server.log) · log
- [server_marks.jsonl](server_marks.jsonl) · result
- [quick_s47e_fired_651_378.log](quick_s47e_fired_651_378.log) · log
- [quick_s47e_fired_651_378_guard.json](quick_s47e_fired_651_378_guard.json) · result
- [quick_s48m_651_378.log](quick_s48m_651_378.log) · log
- [quick_s48m_651_378_guard.json](quick_s48m_651_378_guard.json) · result
- [run_ms_parity.out](run_ms_parity.out) · result
- [run_sealed_s47e.out](run_sealed_s47e.out) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together) · [Copy provenance](../../../PROVENANCE.json)

