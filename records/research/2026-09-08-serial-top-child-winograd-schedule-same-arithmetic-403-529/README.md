# Serial top-child Winograd schedule: same arithmetic, 403/529/655 calls at depth 4/5/6 vs 73/91/109 batched

2026-09-08T10:58Z · finding · informational · derived

Executing the seven top operand pairs of a Winograd product serially with a breadth-batched depth-(d-1) child kernel sharing one arena keeps the real-arithmetic DAG and bill F_d = 7^d (m/2^d)(n/2^d)(2k/2^d - 1) + (7/4)(A+B+C) sum q^j but multiplies kernel calls (18 + 7(18(d-1)+1) instead of 18d+1) while shrinking the scratch owner from the breadth layout; non-GEMM logical traffic is unchanged and no timing gain was established. This serial-top circuit is the depth-6 push kernel later used by the shared-384 card and replaced in its wall repair.

Recorded measurement:

    warm kernel calls batched/serial-top: D4 73/403, D5 91/529, D6 109/655; archived (1024,832,8192) push: D5 kernel arithmetic 7,599,277,504 F vs D4 8,417,538,304 F; scratch bytes at the 1024x8192x1024 read tile: breadth D4 1,168,900,096, breadth D5 2,045,575,168, serial-top D5 420,151,296, serial-top D6 639,320,064; logical non-GEMM traffic 3,509,975,040 B (D4) and 6,456,635,904 B (D5); D5 with bulk copies had one official 120 s timeout in #329884

## Evidence

- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L23](../2026-09-08-reusing-numerical-left-winograd-transform-across-same-layer-products/SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L23)
- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L40](../2026-09-08-reusing-numerical-left-winograd-transform-across-same-layer-products/SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L40)
- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L53](../2026-09-08-reusing-numerical-left-winograd-transform-across-same-layer-products/SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L53)
- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L72](../2026-09-08-reusing-numerical-left-winograd-transform-across-same-layer-products/SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L72)
- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L88](../2026-09-08-reusing-numerical-left-winograd-transform-across-same-layer-products/SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L88)

<details>
<summary>All result files and records (2)</summary>

- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md](../2026-09-08-reusing-numerical-left-winograd-transform-across-same-layer-products/SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md) · record
- [CONTINUATION_2026_09_08_0451_PDT.md](CONTINUATION_2026_09_08_0451_PDT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-serial-top-child-winograd-schedule-same-arithmetic-403-529) · [Copy provenance](../../../PROVENANCE.json)

