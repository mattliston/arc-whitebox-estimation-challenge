# Three-request operand loads and paired output writes remove 1,373 calls while preserving bytes

2026-10-02T06:41Z · experiment · adopted · measured

Every operand load of the tree kernel was six requests (reshape, two splits, two copies, a sum); binding the key's four level-1 data slots as one view makes it three (L3), and writing the upper two output quadrants by one pair addition saves two calls per product (L3o). Both are byte-identical natively; together they take about 1,373 logged calls per network off S37 and lower the setup's binding phase by 638 requests, for a bill of about +0.065% from the one-time view reshapes. Shipped in S38.

Recorded measurement:

    unit check: every output byte-identical on 11 cases x 2 calls; L3 on S37 about -1,260 requests (421 loads) for +0.049% bill estimated, measured +0.127% / +0.075% on a process's first / second network (steady +0.05%), about +0.065% over 20 networks; L3o -254 / -364 calls (about -430 steady), bill identical; S38 logged calls 35,833 vs S37 37,206 (-1,373); binding phase 17,364 requests (S37 18,002)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a21_load3`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_load3.py](code/fragment_load3.py)
- [code/fragment_load3o.py](code/fragment_load3o.py)
- [code/unit_load3.py](code/unit_load3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L119](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L119)
- [TEAM_LOG_2026-10-01_to_10-22.md#L121](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L121)
- [TEAM_LOG_2026-10-01_to_10-22.md#L132](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L132)
- [TEAM_LOG_2026-10-01_to_10-22.md#L159](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L159)
- [TEAM_LOG_2026-10-01_to_10-22.md#L161](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L161)
- [TEAM_LOG_2026-10-01_to_10-22.md#L171](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L171)
- [TEAM_LOG_2026-10-01_to_10-22.md#L205](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L205)
- [RESULT.md#L736](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L736)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333481: fitted dials, masked covariance correction and three-request operand loads](../../../submissions/333481/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-three-request-operand-loads-paired-output-writes-remove-1) · [Copy provenance](../../../PROVENANCE.json)

