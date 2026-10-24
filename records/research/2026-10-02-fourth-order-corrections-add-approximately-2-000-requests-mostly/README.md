# Fourth-order corrections add approximately 2,000 requests, mostly short-vector operations

2026-10-02T03:50Z · finding · informational · derived

A call-by-call count of the two physics fragments located where their 2,000 requests go (the column shift step about 760, the kurtosis term's birth 465, local 450 and final parts 325) and listed request levers with their bill and wall cost: merging small-vector algebra (about -400, free), un-fusing less (-550 or -1,100 for +3 or +7 s of wall, built), thin-star products on stacked operands (-700), one view less per operand load (-600) and shallower covariance rectangles (-462 for +0.09% bill). Three-request operand loads and the edge cut followed from it.

Recorded measurement:

    S37 42,864 requests a network (first network 43,461), target under 41,500 = -1,364; F4 + K4L +2,000 (multiply +732, getitem +281, add +197, astype +139, divide +118); _f4_step 760, _k4l_birth 465, _k4l_local 450, _k4l_dk 325; variants lvfs64 (sha 41d525fc) -550 / +3 s, fused (sha 026087ea) -1,100 / +7 s

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L107](../../TEAM_LOG_2026-10-01_to_10-22.md#L107)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-fourth-order-corrections-add-approximately-2-000-requests-mostly) · [Copy provenance](../../../PROVENANCE.json)

