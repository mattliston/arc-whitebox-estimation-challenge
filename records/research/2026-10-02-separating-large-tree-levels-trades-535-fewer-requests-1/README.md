# Separating large tree levels trades 535 fewer requests for 1.8 seconds of wall time

2026-10-02T12:15Z · finding · informational · measured

The only request lever that touches no physics is how many tree kernels are un-fused (since S36 every tree above 32M floats). Measured on the S39 file with byte-identical predictions and equal bill: un-fusing only trees above 64M floats removes 535 requests per network (-0.004 s of residual) for +2.3% kernel work (about +1.8 s of mean wall, 79.1 s); no un-fusing removes 1,170 (-0.008 s) for +6.9% kernel work (about +5.7 s, worst network about 94 s) and brings the binding phase to 18,068 requests against the designation rule's 18,100. Room for a carrier's requests costs about 300 requests per second of wall for the first 535 and 160 per second after.

Recorded measurement:

    lvfs64: -535 requests, kernel +2.3%, +1.8 s (79.1 s), binding phase 17,596; fused: -1,170 requests, kernel +6.9%, +5.7 s (83.0 s, worst about 94 s), binding phase 18,068 (limit 18,100)

## Programs

- [candidate.py](candidate.py)
- [candidate-32344c5d.py](candidate-32344c5d.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L279](../../TEAM_LOG_2026-10-01_to_10-22.md#L279)
- [TEAM_LOG_2026-10-01_to_10-22.md#L294](../../TEAM_LOG_2026-10-01_to_10-22.md#L294)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-separating-large-tree-levels-trades-535-fewer-requests-1) · [Copy provenance](../../../PROVENANCE.json)

