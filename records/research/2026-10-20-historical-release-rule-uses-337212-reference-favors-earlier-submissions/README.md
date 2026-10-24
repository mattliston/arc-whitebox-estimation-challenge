# Historical release rule uses #337212 as the reference and favors earlier submissions during quiet platform periods

2026-10-20T00:35:00Z · decision · adopted · measured

After #337222 and #337226 lost public networks the owner refused to accept platform randomness: the base for every future submission is candidate_rel_mstack1.py (#337212), the second dial round was rebuilt on that physics, and the lead wrote the rule to fire in the first hours of the UTC day (platform quiet) and never in the last two, never adding wall; for the private designation a profile with 10% or more wall margin on the slow networks was to be preferred if the owner agreed.

Recorded measurement:

    lineage walls across fires: median 98-100 s, p90 105-107, max 111-118; kernel median 89-91, overhead 8.6-8.9, residual median .34-.35, max .367-.397; failed fires kernel medians 91.4 and 94.0

## Programs

- [../../../submissions/337212/estimator.py](../../../submissions/337212/estimator.py)
- [../2026-10-20-second-finite-difference-round-rebuilds-33-vector-dial-partitions/candidate_ms1base_sci_v1.py](../2026-10-20-second-finite-difference-round-rebuilds-33-vector-dial-partitions/candidate_ms1base_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6081](../../TEAM_LOG_2026-10-01_to_10-22.md#L6081)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6083](../../TEAM_LOG_2026-10-01_to_10-22.md#L6083)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6087](../../TEAM_LOG_2026-10-01_to_10-22.md#L6087)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337212: combined zero-cost dial optima on #337142, graded 1.4296259e-9](../../../submissions/337212/README.md)

[Research graph](../../../graph/index.html#node=2026-10-20-historical-release-rule-uses-337212-reference-favors-earlier-submissions) · [Copy provenance](../../../PROVENANCE.json)

