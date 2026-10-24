# Half-zero strip kernel for odd-32 dimensions lowers charged work by 1.30%

2026-10-01T21:22Z · experiment · adopted · measured

Products with a dimension that is an odd multiple of 32 ran at 5 recursion levels or zero-padded to 6; the half-zero strip interleaves the odd 32 rows with 32 zero rows on the deepest index bit so the strip needs 4 of the 7 deepest sub-products and is billed at the 5-level rate (certified against float64 on 17 layouts, rounding level). Full strips cut the bill 1.30% on the 12 screen and check networks with a +0.11% MSE reading that was later shown to be the parent's selected rounding draw; the full-strip stack entered S36.

Recorded measurement:

    screen 35,100,36,45: bill -1.310%, MSE +0.015%; check 8 nets: bill -1.294%, MSE +0.148% (7/8 up); 12 nets pooled bill -1.30% (-1.40% vs S35), MSE +0.11%; kernel calls +1046 (+2.5%), projected platform kernel +4.3%; pad-only variant bill -0.701%, MSE +0.223%, kernel calls +232

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hzs/probe1.py](code/hzs/probe1.py)
- [code/hzs/certify_forced.py](code/hzs/certify_forced.py)
- [code/a07_hzs_v6/candidate_k250000_g750000_d51.py](code/a07_hzs_v6/candidate_k250000_g750000_d51.py)
- [code/hzs/lab.py](code/hzs/lab.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L36](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L36)
- [TEAM_LOG_2026-10-01_to_10-22.md#L49](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L49)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333429: float32 arithmetic, half-zero strips, refreshed setup keys and separate tree levels](../../../submissions/333429/README.md)

[Research graph](../../../graph/index.html#node=2026-10-01-half-zero-strip-kernel-odd-32-dimensions-lowers-charged) · [Copy provenance](../../../PROVENANCE.json)

