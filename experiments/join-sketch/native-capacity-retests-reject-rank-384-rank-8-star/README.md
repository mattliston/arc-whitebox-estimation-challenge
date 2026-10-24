# Native capacity retests reject rank 384 and rank-8 star: adjusted +0.378% and +0.454%

2026-10-12T00:14Z · experiment · rejected · measured

Three levers closed on PyTorch error versus native metered FLOPs were re-measured natively on 8 fresh networks: their native MSE gains match PyTorch (1.27 vs 1.26-1.31%), so the native amplification seen elsewhere applies to error-adding changes only; the moment-closure estimator is at its 1:1 frontier natively too.

Recorded measurement:

    rank 384: MSE -1.272% (se 0.415), bill +1.651%, score +0.378%; ss8: MSE -0.114%, bill +0.568%, score +0.454%; RR at L10: MSE -0.663% (se 0.540), bill +0.639%, score -0.024%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/make_rr_v1.py](code/make_rr_v1.py)
- [code/judge_vs_s48b_v1.py](code/judge_vs_s48b_v1.py)
- [code/s49/r10_384/candidate.py](code/s49/r10_384/candidate.py)
- [code/s49/ss8b/candidate.py](code/s49/ss8b/candidate.py)
- [code/s49/rr10/candidate.py](code/s49/rr10/candidate.py)
- [code/s48/s48b/candidate.py](code/s48/s48b/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1235](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1235)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1237](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1237)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1241](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1241)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1243](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1243)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1244](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1244)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1245](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1245)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Exact old tier gives -2.151% MSE; dropped pair differences are intrinsically high rank](../../../records/research/2026-10-11-exact-old-tier-gives-2-151-mse-dropped-pair/README.md)
- ← enables: [One-direction covariance star adopted: -0.26% MSE for +0.035% FLOPs](../../../records/research/2026-10-11-one-direction-covariance-star-adopted-0-26-mse-0/README.md)
- enables: [Rank redistribution rejected: restoring layer-10 rank 384 gives +0.148% net out-of-selection change](../rank-redistribution-rejected-restoring-layer-10-rank-384-gives/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-native-capacity-retests-reject-rank-384-rank-8-star) · [Copy provenance](../../../PROVENANCE.json)

