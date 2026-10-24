# Fixed 4x4-bit AND/popcount products cost 2.34% and 2.05% more than four-level Winograd

2026-09-08T09:41Z · finding · rejected · derived

A quantized product represented as sixteen binary bitplane pairs packed into uint64 words (broadcast AND, popcount, word sum) was priced under the installed flopscope tariffs before any quantization test; because uint64 AND and popcount are each rate 2, the sixteen-pair core alone is already more expensive than the existing four-level Winograd product on both representative CP push shapes, so the circuit was rejected at its price prerequisite without any precision or fewer-bit rescue.

Recorded measurement:

    core 16*m*n*(5*q-1), q = ceil(k/64): 5,301,600,256 F vs D4 5,180,428,288 F on (1024,1024,4096) and 8,589,934,592 F vs 8,417,538,304 F on (1024,832,8192), i.e. about 2.34% / 2.05% more expensive before quantization, packing and corrections

## Evidence

- [BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L40](BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L40)
- [BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L49](BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L49)
- [BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L70](BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L70)

<details>
<summary>All result files and records (1)</summary>

- [BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md](BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-fixed-4x4-bit-popcount-products-cost-2-34-2) · [Copy provenance](../../../PROVENANCE.json)

