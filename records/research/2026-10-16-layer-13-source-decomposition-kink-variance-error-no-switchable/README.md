# Layer-13 source decomposition of the kink variance error: no switchable source over 2.4%; true L13 means -78%

2026-10-16T17:34:30Z · finding · informational · measured

Each covariance term added at layer 13 was switched off or made exact on the estimator submitted as #336412 (S2tr) and the change of the kink-unit layer-14 variance was regressed on its true error over 8 store ranks: the third-order pair-map cross terms (K21 off-diagonal) move the variance by 5x the error but with share 0.5% (large and not mis-scaled), the fourth-order radial cross plane 0.1-0.2%, the learned covariance channel and edge 0%, exact old tier (no joins) 2.4% in the right direction, exact star 0.2%, all old rows 0.3%: total attributable about 3.5%. Substituting the true layer-13 per-unit means removes 77.95% of the final MSE (the inherited mean error) but leaves the kink variance error unchanged; the true layer-13 variances remove 3.91%. So about 96% of the -9.3% kink prize is structural cross-covariance accumulated over earlier layers and carried through an accurate Gaussian pair kernel, which the representation does not carry.

Recorded measurement:

    Shares (8 ranks, ~1,900 kink units, true error sd 0.108%): K21 off-diag D sd 0.52%, corr -0.04, share 0.5%; K22 off-diag D 0.053%, corr +0.04, 0.1%; GN dcov@L13 0.008%, 0.0%; edge@L13 0.009%, 0.0%; no-joins corr +0.14, beta +1.02, share 2.4%; exact star 0.2%; OTT all rows 0.3%; L13 per-unit mu/v ~0%. L13 input oracles: true means mu13 -> final -77.95% (SE 4.6), kink v error +0.1%; true variances v13 -> final -3.91% (SE 1.5); both -93.8%.

## Programs

- [build_decomp_v1.py](build_decomp_v1.py)
- [analyze_decomp_v1.py](analyze_decomp_v1.py)
- [cand_decomp_k21off13_v1.py](cand_decomp_k21off13_v1.py)
- [cand_decomp_nojoin_v1.py](cand_decomp_nojoin_v1.py)
- [cand_s2tr_kinkoracle_v2.py](cand_s2tr_kinkoracle_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4363](../../TEAM_LOG_2026-10-01_to_10-22.md#L4363)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4368](../../TEAM_LOG_2026-10-01_to_10-22.md#L4368)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4369](../../TEAM_LOG_2026-10-01_to_10-22.md#L4369)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4388](../../TEAM_LOG_2026-10-01_to_10-22.md#L4388)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4393](../../TEAM_LOG_2026-10-01_to_10-22.md#L4393)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4397](../../TEAM_LOG_2026-10-01_to_10-22.md#L4397)

<details>
<summary>All result files and records (2)</summary>

- [DECOMP_TABLE_v1.txt](DECOMP_TABLE_v1.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [True layer-14 pre-activation variance on kink units removes 9.29% of final MSE; no cheap per-unit predictor](../2026-10-16-true-layer-14-pre-activation-variance-kink-units-removes/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-layer-13-source-decomposition-kink-variance-error-no-switchable) · [Copy provenance](../../../PROVENANCE.json)

