# Weighted paired scorer corrects pre-weight FLOP percentages in finite-difference reports

2026-10-20T04:13:02Z · tool · informational · measured

The finite-difference tool reported unweighted census cost; the paired scorer supplied the platform-like weighted cost and adjusted verdict. The same arm read bill -0.85% in the former and +0.084% in the latter.

Recorded measurement:

    - [fable-lead] 2026-10-20T04:13:02Z TOOL NOTE: fd_pair_asym_v1.py's 'bill' column is the PRE-weight census cost (every m15b arm reads bill -0.85% vs science32_stk1_r79, e.g. npca_m 'adjusted -0.874%'); compare32_v1.py's bill (+0.084% for the same pair) is the platform-like weighted one. Adjusted verdicts come from compare32 only; npca_m is +0.06% adjusted (nothing).

## Programs

- [code/fable_oct19_kc_v1/fd_pair_asym_v1.py](code/fable_oct19_kc_v1/fd_pair_asym_v1.py)
- [code/fable_oct19_half_v1/compare32_v1.py](code/fable_oct19_half_v1/compare32_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6130](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6130)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-20-weighted-paired-scorer-corrects-pre-weight-flop-percentages-finite) · [Copy provenance](../../PROVENANCE.json)

