# Tail-completion file fails the request gate at 24.58% loss risk; a typed-stack reduction of 52 requests reaches 19.31%

2026-10-15T05:00:15Z · incident · adopted · measured

The tail-completion file's first production run failed the primary request gate. Turning off the forced depth-7 route with basis reuse removed about 500 cold and 250 warm requests but still failed by a hair; a further exact norm/typed-stack reduction of 52 requests (all 13 saved and 6 tiny Q arrays exact, float64 reduction order disclosed) passed and was the fired source.

Recorded measurement:

    original PRIMARY risk 24.58% (limit 20%); D7-off + basis reuse (329b): -498 cold / -251 warm requests, risk 20.32275% FAIL; e406: -52 requests, risk 19.31% PASS

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3330](../../TEAM_LOG_2026-10-01_to_10-22.md#L3330)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3336](../../TEAM_LOG_2026-10-01_to_10-22.md#L3336)

<details>
<summary>All result files and records (4)</summary>

- [FINAL_RELEASE_GATE_v1.json](FINAL_RELEASE_GATE_v1.json) · result
- [NATIVE_TEN_REPAIR_EXPORT_v1.json](NATIVE_TEN_REPAIR_EXPORT_v1.json) · result
- [CURRENT24_NO_FORCED_D7_TARIFF_v1.json](CURRENT24_NO_FORCED_D7_TARIFF_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Submission #336178: young-read rank-two tail completion with request repair on #336125, bill -1.34%, 1.49701e-9](../../../submissions/336178/README.md)
- shipped in: [Submission #336178: young-read rank-two tail completion with request repair on #336125, bill -1.34%, 1.49701e-9](../../../submissions/336178/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-tail-completion-file-fails-request-gate-24-58-loss) · [Copy provenance](../../../PROVENANCE.json)

