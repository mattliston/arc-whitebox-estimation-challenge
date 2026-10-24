# Native carrier cost: 0.0109% of bill per matrix-vector product; exact covariance eigendecomposition 3.85%

2026-10-02T11:51Z · finding · informational · measured

Read from flopscope's own counters on real calls (float32, in percent of the candidate generation 39 (S39) bill of 2.513e11 flops per network over 13 layers): one product of W or C with a k-column block costs 0.0109% per vector, one pass over the young bank columns 0.056% per vector, a QR of an n x 56 block 0.065% a layer, a small eigen-decomposition under 0.01%, probes from weight columns are free, and eigh of the covariance bills 9 n^3 = 3.85% per call, so no exact eigenvectors natively. Warm-started power iteration needs 16 + 16 extra columns and 2 sweeps (1.40% of bill) to hold 0.98-0.999 of the top-16 subspace because the spectrum is flat at layers 1-2 and unseparated at 16-17. The vector carrier at K 16 costs 1.4-1.6% (3.2-3.4% with the bank pass) and the sketched carrier 2.0% (4.7% with conversion): a carrier pays only if the conversion over the bank is dropped.

Recorded measurement:

    n x n x k matmul: k 8 / 16 / 24 / 32 / 56 = 0.087 / 0.174 / 0.260 / 0.347 / 0.607%; bank pass 0.056% per vector (k 16: 0.89%, k 24: 1.34%); old tier through factors k 16: 0.64% a pass; plane from rank-k factors k 16 / 32: 0.195% / 0.369%; QR n x 56 0.065%; eigh 9.66 Gflop = 3.85% per call; power iteration K 16 no extras 2 sweeps: 0.81 / 0.86 / 0.88 / 0.89 of the subspace (0.70% bill), + 8 extras 0.91 / 0.98 / 0.95 / 0.995 (1.05%), + 16 extras 0.98 / 0.996 / 0.995 / 0.999 (1.40%); eigenvalues 1 / 2 / 8 / 16 / 17 / 32 at layer 12: 0.062 / 0.032 / 0.020 / 0.014 / 0.014 / 0.0084 of trace

## Programs

- [price_probe.py](price_probe.py)
- [dump_state.py](dump_state.py)
- [eig_offline.py](eig_offline.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L274](../../TEAM_LOG_2026-10-01_to_10-22.md#L274)
- [TEAM_LOG_2026-10-01_to_10-22.md#L281](../../TEAM_LOG_2026-10-01_to_10-22.md#L281)
- [TEAM_LOG_2026-10-01_to_10-22.md#L284](../../TEAM_LOG_2026-10-01_to_10-22.md#L284)
- [RESULT.md#L1253](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L1253)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-native-carrier-cost-0-0109-bill-per-matrix-vector) · [Copy provenance](../../../PROVENANCE.json)

