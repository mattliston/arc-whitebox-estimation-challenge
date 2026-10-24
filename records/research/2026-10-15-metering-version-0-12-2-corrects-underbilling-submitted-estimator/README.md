# Metering version 0.12.2 corrects underbilling; the submitted estimator retains identical billing

2026-10-15T14:00Z · finding · informational · measured

The automatic watch reported new flopscope, flopscope-client and flopscope-server 0.12.2 releases whose changelog corrects underbilling and invalid symmetry discounts (unary ufuncs priced by the selected loop; wider out= destinations charged independently). The lead ordered a billing-change audit of the fired source before any further release while keeping the pinned local environment; the later check found the fired file's bill identical under 0.12.2.

Recorded measurement:

    PyPI upload 2026-10-15T13:59 (0.12.1 -> 0.12.2); later check: 2b108 on 7103248 bills 237,983,971,680 FLOPs under both 0.12.0 and 0.12.2

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3381](../../TEAM_LOG_2026-10-01_to_10-22.md#L3381)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3387](../../TEAM_LOG_2026-10-01_to_10-22.md#L3387)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3636](../../TEAM_LOG_2026-10-01_to_10-22.md#L3636)

<details>
<summary>All result files and records (3)</summary>

- [BILLING_SOURCE_AUDIT_v1.json](BILLING_SOURCE_AUDIT_v1.json) · result
- [official_changelog_v0122.md](official_changelog_v0122.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-metering-version-0-12-2-corrects-underbilling-submitted-estimator) · [Copy provenance](../../../PROVENANCE.json)

