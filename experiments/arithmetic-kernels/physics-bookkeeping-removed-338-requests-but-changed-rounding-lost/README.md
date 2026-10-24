# Physics bookkeeping removed 338 requests but changed rounding and lost 0.050% adjusted

2026-10-18T05:29:15Z · experiment · rejected · measured

A bookkeeping rewrite reduced public requests but altered rounding enough to offset its small bill saving. A conservative version restored the original dense float32 matrix and float64 mean diagonal before subsequent combinations.

Recorded measurement:

    First 2: requests -338 on both, raw +.079% (SE .042), bill -.029%, adjusted +.050% → approximately 1.45223e-9; conservative ~224 calls saved.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_bill_v1/build_bookkeeping_v1.py](code/astra_oct18_bill_v1/build_bookkeeping_v1.py)
- [code/astra_oct18_bill_v1/build_bookkeeping_v2.py](code/astra_oct18_bill_v1/build_bookkeeping_v2.py)
- [code/astra_oct18_bill_v1/candidate_v61j_bookkeeping_algebra_HELD_v1.py](code/astra_oct18_bill_v1/candidate_v61j_bookkeeping_algebra_HELD_v1.py)
- [code/astra_oct18_bill_v1/candidate_v61j_bookkeeping_algebra_HELD_v2.py](code/astra_oct18_bill_v1/candidate_v61j_bookkeeping_algebra_HELD_v2.py)
- [code/astra_oct18_bill_v1/fragment_bookkeeping_algebra_v1.py](code/astra_oct18_bill_v1/fragment_bookkeeping_algebra_v1.py)
- [code/astra_oct18_bill_v1/fragment_bookkeeping_algebra_v2.py](code/astra_oct18_bill_v1/fragment_bookkeeping_algebra_v2.py)
- [code/astra_oct18_bill_v1/tiny_bookkeeping_v1.py](code/astra_oct18_bill_v1/tiny_bookkeeping_v1.py)
- [code/opus_fix336687_v1/candidate_v61j.py](code/opus_fix336687_v1/candidate_v61j.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5239](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5239)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-physics-bookkeeping-removed-338-requests-but-changed-rounding-lost) · [Copy provenance](../../../PROVENANCE.json)

