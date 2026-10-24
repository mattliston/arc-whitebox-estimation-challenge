# Padded covariance cut at gate floor 1e-3 adopted: -0.370% FLOPs on ten fresh networks

2026-10-10T03:27Z · decision · adopted · derived

At gate floor 1e-3 the cut's C_pre error (0.6-1.5e-4) sits at the float32 product's own level; run on the specialised padded kernel it is request-neutral and error-free, and it went into candidate generation 43 (S43) and every later file.

Recorded measurement:

    ten fresh networks vs the skeleton: bill -0.370% mean (-0.211..-0.485%), final error +0.06% pooled (se 0.07, 7 of 10 lower), charged ops -58; generic form -0.250% / -1,397 requests

## Programs

- [fragment_dcv_v2.py](fragment_dcv_v2.py)
- [probe_deadcov_v3.py](probe_deadcov_v3.py)
- [compare_sessions.py](compare_sessions.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L518](../../TEAM_LOG_2026-10-01_to_10-22.md#L518)
- [TEAM_LOG_2026-10-01_to_10-22.md#L532](../../TEAM_LOG_2026-10-01_to_10-22.md#L532)
- [TEAM_LOG_2026-10-01_to_10-22.md#L535](../../TEAM_LOG_2026-10-01_to_10-22.md#L535)
- [TEAM_LOG_2026-10-01_to_10-22.md#L536](../../TEAM_LOG_2026-10-01_to_10-22.md#L536)
- [TEAM_LOG_2026-10-01_to_10-22.md#L537](../../TEAM_LOG_2026-10-01_to_10-22.md#L537)
- [TEAM_LOG_2026-10-01_to_10-22.md#L541](../../TEAM_LOG_2026-10-01_to_10-22.md#L541)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Covariance sandwich at gate floor 0.01: final-layer error +7-16% for less than 1% fewer FLOPs](../../../experiments/covariance-structure/covariance-sandwich-gate-floor-0-01-final-layer-error/README.md)
- shipped in: [Submission #335147: joint correction refit, layer-15 law diagonal and padded covariance cut](../../../submissions/335147/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-padded-covariance-cut-gate-floor-1e-3-adopted-0) · [Copy provenance](../../../PROVENANCE.json)

