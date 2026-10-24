# Signed-message training stops after 134 complete graphs when retired RSS reaches 2,075,721,728 bytes

2026-09-21 · incident · failed · measured

The 135th backward graph exceeded the fixed post-retirement 2GB memory cap despite staying below the outer peak cap. Both dependent evaluation stages correctly stopped, and no partial payload became a candidate.

Recorded measurement:

    134 complete graph receipts;33 optimizer updates;post-retirement RSS2,075,721,728bytes;outer peak6,526,107,648bytes;zero added swap.

## Evidence

- [FAILURE.md#L3](FAILURE.md#L3)

<details>
<summary>All result files and records (2)</summary>

- [FAILURE.md](FAILURE.md) · record
- [resource_failure.json](resource_failure.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-signed-message-training-stops-after-134-complete-graphs-when) · [Copy provenance](../../../PROVENANCE.json)

