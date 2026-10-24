# Request-cut prices: optional path off is cheapest; two-strip covariance saves about 4,600 requests per 1% bill

2026-10-16T14:43:20Z · finding · informational · measured

Every switch that removes client requests from the estimator submitted as #336412 (S2tr) was priced in exact flops and billed ops on the first (7103248) and warm (7103256) judge networks against the dm2 + lazybind v2 base. Flags with identical predictions (QL, compute, cluster, ubown) are the cheapest requests per percent of bill; the two-strip covariance sandwich and the five-full limit are the next tier; switching the Smirnov read plans off saves 16% of server instructions but little bill. The cheapest set for the count gates on the slow boundary was QL + compute + two-strip, with a prune for the wall.

Recorded measurement:

    QL off +0.024/+0.016% flops, billed ops -834/-702, predictions identical; compute off +0.054/+0.054%, -330/-330; cluster off +0.066/+0.062%, -60; ubown off +0.040/+0.038%, -40; SM40 off +0.657/+0.787%, -375/-446, process instructions -15.9/-16.4%, MSE +0.09%; D7+D7SQ off +1.178/+0.888%, -993/-1,020; two-strip +0.647/+0.645%, -3,220/-3,124; SAFE (two-strip + five-full) +1.126/+1.095%, -3,645/-3,549. Requests per % bill: QL ~30,000, compute ~5,800, two-strip ~4,600, five-full ~1,100, ubown/cluster ~1,000, D7 ~900, SM40 ~520. S2tr slow boundary 47,690 / 45,776 -> need -3,890 / -3,076.

## Programs

- [build_lowres_v1.py](build_lowres_v1.py)
- [grade_wj_v1.py](grade_wj_v1.py)
- [run_wj_v1.sh](run_wj_v1.sh)
- [cand_lowres_twostrip_v1.py](cand_lowres_twostrip_v1.py)
- [cand_lowres_sm40_v1.py](cand_lowres_sm40_v1.py)
- [warm_judge_v1.py](warm_judge_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../2026-10-16-safe-covariance-geometry-requires-restoring-missing-prebind-key-keeping/fragment_covariance_two_strip_HELD_v1.py](../2026-10-16-safe-covariance-geometry-requires-restoring-missing-prebind-key-keeping/fragment_covariance_two_strip_HELD_v1.py)
- [../2026-10-16-safe-covariance-geometry-requires-restoring-missing-prebind-key-keeping/fragment_safe_five_full_HELD_v2.py](../2026-10-16-safe-covariance-geometry-requires-restoring-missing-prebind-key-keeping/fragment_safe_five_full_HELD_v2.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_exact_v1/dm2/build_dm2_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_exact_v1/dm2/build_dm2_v1.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_S2tr_v1.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_S2tr_v1.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4246](../../TEAM_LOG_2026-10-01_to_10-22.md#L4246)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4299](../../TEAM_LOG_2026-10-01_to_10-22.md#L4299)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-request-cut-prices-optional-path-off-cheapest-two-strip) · [Copy provenance](../../../PROVENANCE.json)

