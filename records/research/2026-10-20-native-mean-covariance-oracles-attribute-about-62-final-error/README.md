# Native mean and covariance oracles attribute about 62% of final error to inherited errors

2026-10-20T03:47:32Z · finding · informational · measured

Exact post means and covariance out of layer 14 removed -62.2 / -61.6% of final MSE on networks 35 / 100. Exact post-variance diagonals at layers 10-14 changed MSE by +0.2 / +1.6%, distinguishing off-diagonal covariance errors from diagonal variance errors.

Recorded measurement:

    - [fable-lead] 2026-10-20T03:47:32Z NATIVE ORACLES, MEAN SIDE (same runners, nets 35 / 100, final-layer MSE): exact post MEAN out of L14: -52.8 / -52.1; exact post mean+cov out of L14: -62.2 / -61.6 (= out of L10-14: -62.6); exact mu+C INTO L14: -54.0 / -54.2 (so L14's own pair map adds ~1% given exact inputs); exact per-unit k3 / k4 into L14: -2.6 / -3.8 and -3.3 / +0.1; exact k3 at L10-14: +0.2 / -6.8, k4 at L10-14: -2.6 / (pending), k3+k4 at L10-14: -5.5 / -14.3; exact mu+C+k3+k4 into L10-14: -58.3 / -60.9; exact post VARIANCES (diag) at L10-14: +0.2 / +1.6 (nothing); exact post COVARIANCE at L10-14: -23.8 / -33.2. READ: the final error = ~38% generated in the TERMINAL layer (l=15: nonlinear_scalar(mu, v, k3, k4) + learned head, given exact mu15/C15) + ~62% inherited, of which the mean path (mu14 exact alone -41%) is itself made layer by layer from off-diagonal covariance errors (v) and per-unit cumulants; diagonal post variances are already right. Terminal split (exact mu / v / k3 / k4 / head off) running (P7).

## Programs

- [../2026-10-20-native-covariance-oracles-reduce-late-layer-error-less-than/native_cov_oracle_v1.py](../2026-10-20-native-covariance-oracles-reduce-late-layer-error-less-than/native_cov_oracle_v1.py)
- [../2026-10-20-native-covariance-oracles-reduce-late-layer-error-less-than/native_k31_oracle_v1.py](../2026-10-20-native-covariance-oracles-reduce-late-layer-error-less-than/native_k31_oracle_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6127](../../TEAM_LOG_2026-10-01_to_10-22.md#L6127)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-native-mean-covariance-oracles-attribute-about-62-final-error) · [Copy provenance](../../../PROVENANCE.json)

