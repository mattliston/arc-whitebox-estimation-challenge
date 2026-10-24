# Cubic response and coordinate-star masks differ: exact tensor restoration does not preserve masked reads

2026-09-07T08:49Z · finding · informational · derived

Independent review of the retained response-row candidate found that its intermediate CP off-diagonal read covers response_indices x live_indices while the compact star read covers the full live x live plane, so moving coordinate-star content between the CP bank and the explicit star can change the rows omitted from the CP read and a later linear push can expose the decomposition dependence. Any score-bearing test of a CP compression therefore has to run the exact current readers, all heads and the complete rollout.

## Evidence

- [GLOBAL_CP_KERNEL_RESPONSE_READER_CAVEAT_V1.md#L6](GLOBAL_CP_KERNEL_RESPONSE_READER_CAVEAT_V1.md#L6)
- [GLOBAL_CP_KERNEL_RESPONSE_READER_CAVEAT_V1.md#L19](GLOBAL_CP_KERNEL_RESPONSE_READER_CAVEAT_V1.md#L19)

<details>
<summary>All result files and records (1)</summary>

- [GLOBAL_CP_KERNEL_RESPONSE_READER_CAVEAT_V1.md](GLOBAL_CP_KERNEL_RESPONSE_READER_CAVEAT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-cubic-response-coordinate-star-masks-differ-exact-tensor-restoration) · [Copy provenance](../../../PROVENANCE.json)

