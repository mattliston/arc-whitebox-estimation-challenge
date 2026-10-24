# Grouped strided-read primary schedules: useful kernel saving, insufficient alone

2026-09-05T23:46Z · finding · informational · measured

Actual frozen-bank matmul metadata was newly collected on35/100, with both complete output hashes BYTE IDENTICAL to their prior native receipts. No truth or raw score was read. Crossing adjusted4e-9 at the bank's unchanged raw requires19.715% savings.

Recorded measurement:

    Actual frozen-bank matmul metadata was newly collected on35/100, with both complete output hashes BYTE IDENTICAL to their prior native receipts. No truth or raw score was read. Covered matrix calls account for466.6B FLOPs of each~490B native card. Diagnostic wall18.4s/net is not a new persistent runtime qualification. Shape-only envelope: one/two primary Smirnov levels or their exact matrix-transpose dual, then Winograd depth0/3/4/5, chosen per shape. All SLP arithmetic and repeated child padding counted. Ignoring copies, allocations and remote overhead, savings are11.961% /11.889% of the measured35/100 remote bank bills. Requiring even the necessary simultaneous full-batch leaf A/B/C storage to fit the current1,168,900,096-byte arena reduces the optimistic figures to7.964% /8.025%. That memory condition is NOT a full live-memory implementation certificate. Crossing adjusted4e-9 at the bank's unchanged raw requires19.715% savings. Thus these tested schedules do not provide a standalone threshold bank. No production kernel is built on this receipt. This is scoped: cyclic duals, arbitrary mixed recursion, new memory scheduling and other algorithms have not been ruled out. Keep the exact identity/source for a future independently strong raw improvement, rather than claiming a universal FLOP lower bound.

## Programs

- [audit_smirnov_cyclic_v1.py](audit_smirnov_cyclic_v1.py)
- [audit_smirnov_linear_circuit_v1.py](audit_smirnov_linear_circuit_v1.py)
- [audit_smirnov_slp_v1.py](audit_smirnov_slp_v1.py)
- [../../../submissions/329926/estimator.py](../../../submissions/329926/estimator.py)
- [census_bank_matmuls_v1.py](census_bank_matmuls_v1.py)
- [price_mixed_fmm_envelope_v1.py](price_mixed_fmm_envelope_v1.py)
- [price_smirnov_bank_v1.py](price_smirnov_bank_v1.py)

## Evidence

- [SMIRNOV_BANK_AUDIT_V1.md#L1](SMIRNOV_BANK_AUDIT_V1.md#L1)
- [SMIRNOV_BANK_AUDIT_V1.md#L22](SMIRNOV_BANK_AUDIT_V1.md#L22)

<details>
<summary>All result files and records (4)</summary>

- [SMIRNOV_BANK_AUDIT_V1.md](SMIRNOV_BANK_AUDIT_V1.md) · record
- [smirnov_slp_audit_v1.json](smirnov_slp_audit_v1.json) · result
- [bank_matmul_census_v1.json](bank_matmul_census_v1.json) · result
- [smirnov_bank_cost_v1.json](smirnov_bank_cost_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-grouped-strided-read-primary-schedules-useful-kernel-saving-insufficient) · [Copy provenance](../../../PROVENANCE.json)

