# Native 40-network refit test stopped at 7.5 GB: parent call 13, trained call 31 at 7,533,494,272 B

2026-09-21T13:45Z · incident · failed · measured

The original native endpoint for the refit candidate stopped at its 7.5 GB guard after 12 completed parent calls; a lifecycle study that saved each bill and cleared the closed-context diagnostic history then completed the three null calls and 41 parent calls with exact historical controls, but its trained sequence stopped during rank 181 (call 31) at 7,533,494,272 bytes. No VALID target was decoded and no candidate was rejected on science; the runs were superseded by the eight-call allocator comparison and the retirement-native study.

Recorded measurement:

    parent endpoint failed on call 13 (12 completed); trained lifecycle sequence completed 30 calls, stopped at 7,533,494,272 B under the 7.5 GB guard; zero added swap in both

## Evidence

- [native_retirement_checkpoint99.md#L7](native_retirement_checkpoint99.md#L7)
- [REEXAMINATION_INDEX_20260919.md#L464](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L464)
- [REEXAMINATION_INDEX_20260919.md#L474](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L474)

<details>
<summary>All result files and records (4)</summary>

- [native_retirement_checkpoint99.md](native_retirement_checkpoint99.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [controller_result.json](controller_result.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- enables: [Native allocator comparison: peak memory fell from 6,827,851,776 to 2,507,538,432 bytes with exact outputs](../2026-09-21-native-allocator-comparison-peak-memory-fell-6-827-851/README.md)

[Research graph](../../../graph/index.html#node=2026-09-21-native-40-network-refit-test-stopped-7-5-gb) · [Copy provenance](../../../PROVENANCE.json)

