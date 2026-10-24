# Eager key binding spends over 130 billion setup operations to save 12 billion later operations

2026-10-01T22:18Z · finding · informational · measured

Over a 12-network worker session lazy tree binds (reshape billed at the tree's element count) cost 11.9 Gflop, 0.374% of the bill, across about 500 distinct network-specific keys with only 2 eviction rebinds. Binding the bare tree view of every possible key in setup would remove this byte-identically but cost 130+ Gflop of setup per worker; not recommended. Row cuts rounded to 32 (about -0.1%) and strips under the B-tree cache (the bill rule does not take them) were dropped at the same time.

Recorded measurement:

    11.9 Gflop = 0.374% of the 12-network bill; per network 0.49 0.57 0.27 0.32 0.17 0.09 0.12 0.11 0.87 0.74 0.38 0.36%; 2 eviction rebinds; about 500 distinct keys; prebind all: 130+ Gflop setup to save about 12

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L44](../../TEAM_LOG_2026-10-01_to_10-22.md#L44)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-eager-key-binding-spends-over-130-billion-setup-operations) · [Copy provenance](../../../PROVENANCE.json)

