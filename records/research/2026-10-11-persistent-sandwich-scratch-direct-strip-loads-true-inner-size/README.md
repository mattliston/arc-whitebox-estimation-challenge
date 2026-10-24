# Persistent sandwich scratch, direct strip loads and true-inner-size products adopted: -0.200/-0.284% FLOPs

2026-10-11T08:33Z · decision · adopted · measured

Persistent scratch for the covariance sandwich (-494 / -532 requests, byte-identical), direct strip loads into the tree's level-1 views (-0.088% metered FLOPs, byte-identical) and the dead-cut sandwich at its true inner size m (-0.14%, rounding level) stacked onto candidate generation 47 (S47)'s physics and submitted as S47e.

Recorded measurement:

    XRQS 42,868 / 41,686 requests (-494 / -532), bill -0.003%; HZD -0.088 / -0.088%; XSDM -0.102 / -0.175%, ops +314 / +196; whole stack vs S47s: 43,596 / 41,898 requests, bill -0.200 / -0.284%

## Programs

- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/engineer/a56_req/fragment_rqs_v1.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/engineer/a56_req/fragment_rqs_v1.py)
- [fragment_hzd_v1.py](fragment_hzd_v1.py)
- [fragment_hzd_v3.py](fragment_hzd_v3.py)
- [probe_packs.py](probe_packs.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/engineer/a60_psdm/fragment_psdm_v1.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/engineer/a60_psdm/fragment_psdm_v1.py)
- [s48_build.sh](s48_build.sh)
- [dly_from_pt_v2.py](dly_from_pt_v2.py)
- [../2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/read_pt.py](../2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/read_pt.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1034](../../TEAM_LOG_2026-10-01_to_10-22.md#L1034)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1046](../../TEAM_LOG_2026-10-01_to_10-22.md#L1046)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1054](../../TEAM_LOG_2026-10-01_to_10-22.md#L1054)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1055](../../TEAM_LOG_2026-10-01_to_10-22.md#L1055)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1056](../../TEAM_LOG_2026-10-01_to_10-22.md#L1056)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1057](../../TEAM_LOG_2026-10-01_to_10-22.md#L1057)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #335349: covariance scratch reuse and cheaper strip and sandwich products](../../../submissions/335349/README.md)
- ← enables: [Request caches and seven-level strip pushes adopted: byte-identical request savings and about -0.18% FLOPs](../2026-10-11-request-caches-seven-level-strip-pushes-adopted-byte-identical/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-persistent-sandwich-scratch-direct-strip-loads-true-inner-size) · [Copy provenance](../../../PROVENANCE.json)

