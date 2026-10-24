# Row-sum representation of the third-order remainder saves 2.37% work for 0.50% additional error

2026-10-02T10:45Z · experiment · adopted · measured

What a layer takes from the star plane A is (W o W)' A' W, which splits into a part needing only the row sums of A (holding 99.4-99.8% of its power at layers 4-14) and a remainder; the rank-32 factors missed 1-2% of those row sums, 15-45% of the power of their whole error. Carrying the row-sum part exactly with no factorization, no QR and no thin product (rank 0) costs +0.50% of MSE for -2.37% of bill on ten fresh networks (4.7 times, near the lead's 5x rule); rank 8 and a one-direction form do not earn their bill back. Shipped in S39, where the ten-rank native estimate of the error change was -0.04% and the board gave -0.93%.

Recorded measurement:

    row-sum part 99.4-99.8% of s2 power (layers 4-14); rank 32 + exact row sums 651 / 378: MSE -0.24% / -0.52%, bill +0.036%; ten fresh vs S38: rank 0 MSE +0.50% [-0.06,+0.97] (2 of 10 lower), bill -2.372%, adjusted -1.88% [-2.44,-1.43], 150 calls fewer; rank 8 + row sums MSE +0.31%, bill -1.765%, adjusted -1.47%; one direction MSE +0.89% [+0.36,+1.47], bill -2.353%; offline error of the transported plane 4.0-7.6% (rank 0), 2.6-4.1% (rank 8), 1.9-2.8% (rank 32), fired form 2.2-3.5%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/a31_star/probe_star3.py](code/a31_star/probe_star3.py)
- [code/a33_starsum/unit_ss.py](code/a33_starsum/unit_ss.py)
- [code/a33_starsum/forms_offline.py](code/a33_starsum/forms_offline.py)
- [code/a33_starsum/star_ss_torch.py](code/a33_starsum/star_ss_torch.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L240](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L240)
- [TEAM_LOG_2026-10-01_to_10-22.md#L252](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L252)
- [TEAM_LOG_2026-10-01_to_10-22.md#L259](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L259)
- [TEAM_LOG_2026-10-01_to_10-22.md#L271](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L271)
- [TEAM_LOG_2026-10-01_to_10-22.md#L322](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L322)
- [RESULT.md#L1114](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L1114)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333528: joint dial solve and third-order remainder carried by row sums cut bill 2.2%](../../../submissions/333528/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-row-sum-representation-third-order-remainder-saves-2-37) · [Copy provenance](../../../PROVENANCE.json)

