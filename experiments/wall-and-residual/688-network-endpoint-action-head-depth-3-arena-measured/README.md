# 688-network endpoint-action head in a depth-3 arena: measured and rejected

2026-08-27 · experiment · rejected · measured

The mechanical depth-3 third-cumulant arena (D3) fallback passes all target-free [35,279,35] gates at exact bills [1,387,818,002,786; 1,425,446,354,419; 1,386,800,025,067], maximum residual 0.373081 s, maximum resident-set size (RSS) 4,726,603,776 bytes, and exact repeat. Archive b860f042 was validated. It is held: full688 out-of-fold (OOF) mean squared error (MSE) is 3.1113% worse than native on the same 688 networks, and projected adjusted 1.92636e-8 is worse than actual compute-only depth-4 third-cumulant arena (D4) #328622 at 1.783621479e-8.

Recorded measurement:

    The mechanical D3 fallback passes all target-free [35,279,35] gates at exact bills [1,387,818,002,786; 1,425,446,354,419; 1,386,800,025,067], maximum residual 0.373081 s, maximum RSS 4,726,603,776 bytes, and exact repeat. Archive b860f042 was validated. It is held: full688 OOF MSE is 3.1113% worse than native on the same 688 networks, and projected adjusted 1.92636e-8 is worse than actual compute-only D4 #328622 at 1.783621479e-8.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py](code/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py)
- [code/public_q24_d3_alpha075_full688_qhead_v1/estimator.py](code/public_q24_d3_alpha075_full688_qhead_v1/estimator.py)

## Evidence

- [RESULT.json#L4914](RESULT.json#L4914)
- [D3_CANDIDATE_RESULT.json#L1](D3_CANDIDATE_RESULT.json#L1)
- [FULL_GATE_RESULT.json#L1](FULL_GATE_RESULT.json#L1)
- [SUBMIT_WATCH.log#L1](SUBMIT_WATCH.log#L1)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.json](RESULT.json) · result
- [D3_CANDIDATE_RESULT.json](D3_CANDIDATE_RESULT.json) · result
- [FULL_GATE_RESULT.json](FULL_GATE_RESULT.json) · result
- [SUBMIT_WATCH.log](SUBMIT_WATCH.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-688-network-endpoint-action-head-depth-3-arena-measured) · [Copy provenance](../../../PROVENANCE.json)

