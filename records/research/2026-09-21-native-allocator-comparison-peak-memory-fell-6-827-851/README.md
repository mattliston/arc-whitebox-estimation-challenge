# Native allocator comparison: peak memory fell from 6,827,851,776 to 2,507,538,432 bytes with exact outputs

2026-09-21T13:59Z · tool · adopted · measured

A baseline arm and a relief arm each ran native calls 37/39/42/37; all 16 outputs, complete FLOP bills, variance domains and selectors matched exactly, 18 tracked call-local objects were already dead before intervention, and post-call unused-mapping relief lowered the retired RSS. The qualified per-call retirement (18-object release plus relief after the saved closed bill) became the standard for every later native call in this lineage.

Recorded measurement:

    maximum retired RSS 6,827,851,776 B to 2,507,538,432 B; largest single drop 4,252,450,816 B; baseline/relief guard peaks 6,962,020,352 / 6,677,397,504 B; zero added swap

## Evidence

- [native_retirement_checkpoint99.md#L14](../2026-09-21-native-40-network-refit-test-stopped-7-5-gb/native_retirement_checkpoint99.md#L14)
- [REEXAMINATION_INDEX_20260919.md#L465](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L465)

<details>
<summary>All result files and records (4)</summary>

- [qualification.json](qualification.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [native_retirement_checkpoint99.md](../2026-09-21-native-40-network-refit-test-stopped-7-5-gb/native_retirement_checkpoint99.md) · record
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- ← enables: [Native 40-network refit test stopped at 7.5 GB: parent call 13, trained call 31 at 7,533,494,272 B](../2026-09-21-native-40-network-refit-test-stopped-7-5-gb/README.md)
- enables: [Refit mean encoder on the 40-network validation split: adjusted ratio 0.9848676992236052, 32/40 wins at equal compute](../../../experiments/evaluation-methodology/refit-mean-encoder-40-network-validation-split-adjusted-ratio/README.md)

[Research graph](../../../graph/index.html#node=2026-09-21-native-allocator-comparison-peak-memory-fell-6-827-851) · [Copy provenance](../../../PROVENANCE.json)

