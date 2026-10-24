# Four-second setup budget binds approximately 19,000 requests against 18,555 required

2026-10-01T23:00Z · finding · informational · derived

The flops a platform worker books during setup identify exactly how many tree keys it bound before the estimator's 4.0 s guard fired. A normal worker gets through about 19,000 requests (range 18,600-20,100); S35's binding phase needs 18,555 requests (97.7% of that), so one worker in five of #333111 did not finish it; the slow re-set-up worker of #333099 bound only 580 of 649 keys and then ran 1.30x slow in residual. With the un-fuse fragment the binding phase needs 17,198 requests (90.5%). A light setup-margin tool reports this for any candidate.

Recorded measurement:

    #333111 worker w-2: 34,378,719,666 setup flops = 623 of 649 keys = 17,954 requests; #333099 slow worker: 33,113,512,758 = 580 keys = 14,589 requests, residual 1.30x; complete S35 binding phase 34,922,268,623 flops at 4.27-4.33 s; normal worker about 19,000 requests (18,600-20,100); S35 / RRPI 18,555 requests (97.7%), with lvfuse_sized32 17,198 (90.5%)

## Programs

- [setup_margin.py](setup_margin.py)
- [setup_clock.py](setup_clock.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L38](../../TEAM_LOG_2026-10-01_to_10-22.md#L38)
- [TEAM_LOG_2026-10-01_to_10-22.md#L52](../../TEAM_LOG_2026-10-01_to_10-22.md#L52)

<details>
<summary>All result files and records (2)</summary>

- [margin_table_v1.log](margin_table_v1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests) · [Copy provenance](../../../PROVENANCE.json)

