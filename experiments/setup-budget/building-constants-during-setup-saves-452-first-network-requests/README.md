# Building constants during setup saves 452 first-network requests with byte-identical predictions

2026-10-02T16:38Z · experiment · adopted · measured

482 requests of every first network built constants and buffers that exist once per process (the learned-correction layers' parameter views, the decoded literals, the pair map's persistent buffers, birth coefficients, feature columns, the tables of the pair law, the exact-term module and the kurtosis term). A wrapper of the binding phase now builds them at its head with the prediction's own calls (a failure inside clears the caches and the binding goes on); setup gains 0.031 s and 269 logged operations, the first network loses 452 requests by the count and 0.057% of bill, warm networks are unchanged and predictions byte-identical. Entered candidate generation 42 (S42) with the byte-identical diet.

Recorded measurement:

    setup +0.031 s, +269 logged operations, +153 Mflop; first network 43,221 vs 43,673 requests (-452: logged -264, index calls -188), bill -0.057%; second network identical; 482 first-network requests by site (parameter views 135 + 21, literals 56, pair map buffers 43 + 26 + 17 + 10, birth coefficients 42, feature columns 31, tables 19 + 13, others 69)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a40_setup`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_cs_v1.py](code/fragment_cs_v1.py)
- [code/reset_check.py](code/reset_check.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L374](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L374)
- [TEAM_LOG_2026-10-01_to_10-22.md#L379](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L379)
- [TEAM_LOG_2026-10-01_to_10-22.md#L383](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L383)
- [RESULT.md#L1611](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L1611)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333607: learned norm-coupling diagonal input with two fitted steps and fewer requests](../../../submissions/333607/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-building-constants-during-setup-saves-452-first-network-requests) · [Copy provenance](../../../PROVENANCE.json)

