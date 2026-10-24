# Submission #333528 has approximately 2% estimated private-run network-loss risk

2026-10-02T12:17Z · finding · informational · measured

By the analyst's residual model re-evaluated on 1,090 public runs, S39 (mean residual 0.3018 s, worst 0.347 s) loses a network in a fresh run with probability about 2%, the same as S37 and S38: the worst network must sit 1.19 times above the team's usual worst-to-mean ratio (13 of 727 leader-sized runs), 2 of 83 runs with a mean residual of 0.29-0.32 s lost a network, and in 81 same-file pairs the worst residual never rose more than 13.3% (S39 needs 15.4%). Every loss in this range is a slow-worker event that no file since S35 would survive; what requests change is how small an event suffices. The analyst's first-network model reads 1.0% for S39 and 1.1% for S39 + 200 requests.

Recorded measurement:

    readings 1.8% (13 of 727), 2.4% (2 of 83 at 0.29-0.32 s; 2 of 94 at 0.30-0.33 s), 0 of 81 pairs (max rise 13.3%); with requests: +1,000 2.2%, +2,000 2.9%, +3,000 4.8%, +4,000 8.1%, +5,000 11.6%; -1,000 1.65%, -2,000 1.2%; A19: S39 1.0% (0.4-2.4%), with 680 requests off each first network 0.6%, S40 = S39 + 200 1.1% (0.7% with the lever); slow workers 8-10 of 729 runs at 1.17-1.31x; wall: 0 of 591 runs under 85 s lost a network

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L295](../../TEAM_LOG_2026-10-01_to_10-22.md#L295)
- [TEAM_LOG_2026-10-01_to_10-22.md#L307](../../TEAM_LOG_2026-10-01_to_10-22.md#L307)
- [RESULT.md#L1110](../2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md#L1110)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](../2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md) · result
- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-submission-333528-has-approximately-2-estimated-private-run-network) · [Copy provenance](../../../PROVENANCE.json)

