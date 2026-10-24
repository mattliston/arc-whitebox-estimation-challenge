# Designation caps set to 43,800 first-network requests, 42,700 warm requests and 82 s mean wall

2026-10-02T16:19Z · decision · adopted · measured

Because the worst network of a run is a first network after a setup, the designation rule's mean-residual line (0.305 s) was replaced by caps on the two counts of the boundary census: at most 43,800 requests on a worker's first network and 42,700 on a warm one (first networks at 0.322 s, others at 0.307 s, usual worst 0.341 s, the 1% risk of candidate generation 39 (S39)), and the wall line moved from 80 to 82 s (wall has 31% of room, 2 s buy 535 requests). The candidate generation 41 (S41) stack without diet sat over both caps; the variant with the strips' leaf views by one index expression, un-fusing only above 64M floats and the first-load views of the 128 most-used keys built at bind (first 43,466 / warm 42,062, byte-identical predictions) met all six judge lines and was fired as S41.

Recorded measurement:

    S39 43,565 / 41,682; S40 44,578 / 42,678; S41 stack 44,881 / 42,982 (over by 1,081 / 282); + HV 44,311 / 42,602; + Rl3_bind128 44,025 / 42,668; + Rlvfs_limit64 43,466 / 42,062; prices 7.19 microseconds per warm request, 12.7 per extra first-network request; loss probability readings: S39 1.1% / 1.1%, S40 1.4% / 3.0%, S41 + HV 1.1% / 2.2%, + bind128 0.8% / 2.2%, + both 0.55% / 1.4%; slow setups: 1.124x pace 596 keys bound, first 43,875 / warm 42,429; 1.3x pace 497 keys, 44,452 / 42,668 (+1.86%, limit +2%); binding phase 18,028 (limit 18,100)

## Programs

- [variant_table.py](variant_table.py)
- [loss_risk.py](loss_risk.py)
- [first_network_cost.py](first_network_cost.py)
- [../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py](../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L373](../../TEAM_LOG_2026-10-01_to_10-22.md#L373)
- [TEAM_LOG_2026-10-01_to_10-22.md#L374](../../TEAM_LOG_2026-10-01_to_10-22.md#L374)
- [TEAM_LOG_2026-10-01_to_10-22.md#L375](../../TEAM_LOG_2026-10-01_to_10-22.md#L375)
- [TEAM_LOG_2026-10-01_to_10-22.md#L376](../../TEAM_LOG_2026-10-01_to_10-22.md#L376)
- [TEAM_LOG_2026-10-01_to_10-22.md#L383](../../TEAM_LOG_2026-10-01_to_10-22.md#L383)
- [RESULT.md#L1037](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md#L1037)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333587: young-pair fourth-chaos plane, norm-coupling dial solve and fewer kernel requests](../../../submissions/333587/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-designation-caps-set-43-800-first-network-requests-42) · [Copy provenance](../../../PROVENANCE.json)

