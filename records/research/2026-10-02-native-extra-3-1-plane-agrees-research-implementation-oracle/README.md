# Native extra (3,1) plane agrees with research implementation: oracle lowers MSE 15.31% versus 15.60%

2026-10-02T10:45Z · finding · informational · measured

The native pair map already accepts an extra (3,1) plane (standardised inside as plane_ij / (sigma_i^3 sigma_j), zero diagonal, added to T31). Fed with the oracle matrix on network 651 at strength 1 over layers 1-14, the native prediction change has cosine 0.94-0.995 with the torch change from layer 3 on, projection 0.94-1.005 and a leftover of 1.4-5.1e-6 rms (two rounding-level native builds differ by 4-5e-6): any carrier that hands the pair map a plane in the torch convention acts natively as in the card.

Recorded measurement:

    final-layer MSE -15.31% natively (15.240 -> 12.907e-9) vs -15.60% torch (15.622 -> 13.185e-9); cosine 0.94-0.995 (0.88 at layer 2), projection 0.94-1.005, leftover 1.4-5.1e-6 rms

## Programs

- [oracle_native.py](oracle_native.py)
- [parity.py](parity.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L259](../../TEAM_LOG_2026-10-01_to_10-22.md#L259)
- [RESULT.md#L1185](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L1185)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-native-extra-3-1-plane-agrees-research-implementation-oracle) · [Copy provenance](../../../PROVENANCE.json)

