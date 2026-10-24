# Three native approximations explain 91% of the native-versus-research error gap

2026-10-10T23:10Z · finding · informational · derived

Every native-only approximation of candidate generation 45 (S45) saves more metered FLOPs than its error costs and the thresholds sit at their optimum; the three cuts make 91% of the +0.6-1.2% native-vs-PyTorch error gap, which motivated refitting on a cut-simulated moment-closure estimator.

Recorded measurement:

    all three exact: native-torch rms 16.4 -> 4.8e-6; old-tier tails +2.92% bill for +0.98% error; young dead tail +1.65% for -0.17%; pair-mean truncation +1.48% for -0.71%; gap steps S42 +0.63, S43 +0.92, S44 +0.86, S45 +0.84, S46 +1.20%

## Programs

- [make_variant.py](make_variant.py)
- [audit_table.py](audit_table.py)
- [gap_decomp.py](gap_decomp.py)
- [../../../experiments/third-order-planes/native-approximation-audit-rejects-aggressive-positive-tail-cuts-pair/code/level_table.py](../../../experiments/third-order-planes/native-approximation-audit-rejects-aggressive-positive-tail-cuts-pair/code/level_table.py)
- [../../../experiments/third-order-planes/native-approximation-audit-rejects-aggressive-positive-tail-cuts-pair/code/probe_uvp.py](../../../experiments/third-order-planes/native-approximation-audit-rejects-aggressive-positive-tail-cuts-pair/code/probe_uvp.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L699](../../TEAM_LOG_2026-10-01_to_10-22.md#L699)
- [TEAM_LOG_2026-10-01_to_10-22.md#L704](../../TEAM_LOG_2026-10-01_to_10-22.md#L704)
- [TEAM_LOG_2026-10-01_to_10-22.md#L710](../../TEAM_LOG_2026-10-01_to_10-22.md#L710)
- [TEAM_LOG_2026-10-01_to_10-22.md#L715](../../TEAM_LOG_2026-10-01_to_10-22.md#L715)
- [TEAM_LOG_2026-10-01_to_10-22.md#L727](../../TEAM_LOG_2026-10-01_to_10-22.md#L727)
- [TEAM_LOG_2026-10-01_to_10-22.md#L729](../../TEAM_LOG_2026-10-01_to_10-22.md#L729)
- [TEAM_LOG_2026-10-01_to_10-22.md#L730](../../TEAM_LOG_2026-10-01_to_10-22.md#L730)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Joint-refit evaluation: -0.16% native MSE on ten fresh networks versus -0.60% research MSE on 50 ranks](../2026-10-10-joint-refit-evaluation-0-16-native-mse-ten-fresh/README.md)
- enables: [Cut-simulated research estimator removes 95% of the native prediction discrepancy](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/README.md)
- enables: [Pair-mean rank 320 rejected natively: -0.92% MSE, +1.26% FLOPs and +0.33% score](../../../experiments/third-order-planes/pair-mean-rank-320-rejected-natively-0-92-mse/README.md)
- enables: [Tail-repair alternatives rejected: exact tails gain about 1.06% MSE for 2.9% + 1.65% additional FLOPs](../../../experiments/third-order-planes/tail-repair-alternatives-rejected-exact-tails-gain-about-1/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-three-native-approximations-explain-91-native-versus-research-error) · [Copy provenance](../../../PROVENANCE.json)

