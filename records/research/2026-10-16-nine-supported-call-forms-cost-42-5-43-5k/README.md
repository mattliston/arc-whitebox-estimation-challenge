# Nine supported call forms cost 42.5-43.5K residual instructions per request

2026-10-16T03:31:10Z · finding · rejected · measured

Synthetic getitem, reshape, matmul, reduction, cast and output-form tests found no material call-form residual lever.

Recorded measurement:

    all42.5-43.5Kinstructions/request;spread<=2.3%

## Programs

- [synth_form_v1.py](synth_form_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3785](../../TEAM_LOG_2026-10-01_to_10-22.md#L3785)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [CALLCOST_RESULT_v1.json](CALLCOST_RESULT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-nine-supported-call-forms-cost-42-5-43-5k) · [Copy provenance](../../../PROVENANCE.json)

