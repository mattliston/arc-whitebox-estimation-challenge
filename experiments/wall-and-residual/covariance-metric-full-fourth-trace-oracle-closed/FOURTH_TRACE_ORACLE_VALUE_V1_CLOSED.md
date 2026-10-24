# Covariance-metric full fourth-trace oracle — CLOSED

2026-09-06 03:40:50 PDT, Phase 2. Both fixed existing independent8M
streams passed new-statistic qualification. Post12/pre13 shifted means
reproduced exactly. R-diagonal independent reduction differences were
0.0482%/0.0481%; pre13 K22/K31 marginal-diagonal checks about0.192%.
Acquisition72.881s/76.679s, one MPS job each, peak hostRSS1.263/1.283GB.
Shared float32 numerical bias remains possible; replicas remove only
independent sampling noise from cross-energy estimates.

The actual bank inputs were frozen first, prefix13 comparisons byte-exact,
independent pair baseline max6.94e-18. Oracle core reconstruction max
4.49e-15 relative; finite response vs direct pair max4.47e-15 absolute.

This reconstruction matches true post12 Euclidean partial trace with an
increment Sym(A,C_bank), then pushes both factors through W13. It is a
trace-matching covariance-metric reconstruction, NOT an orthogonal
projection in Frobenius tensor norm. K3 remains frozen. Targets isolate
the exact fourth-only pair response, not total covariance or final error.

| Consumer | Residual/baseline replica cross-energy | Cross A->B | Cross B->A | Signal/noise |
|---|---:|---:|---:|---:|
| Covariance response | 0.6135335245 | 0.6257960528 | 0.6231072761 | 42.73 |
| Next mean response | 0.6112197142 | 0.6287411502 | 0.6175581810 | 39.37 |

Required cross-energy ratios <=0.50 and both ordinary cross ratios<=0.65.
Both fail the first requirement despite qualifying signal and the ordinary
ratio requirements. The ~39% isolated response-error reduction is not an
adjusted or raw score improvement. No fit, larger sample/net/layer study,
carrier, port, panel, or submission under this design.

The canonical Euclidean harmonic reconstruction Sym(A,I), explicitly
specified in the earlier full-carrier proposal, was NOT tested here.
Do not silently extend this negative result to that different subspace.
Any separately justified information audit can reuse these statistics;
no new sampling follows from this failure itself.

Receipt fourth_trace_oracle_value_v1.json/.npz. The two new statistic
files are fourth_trace_oracle_r35_A_v1.npz and B, with hashes recorded in
the receipt. Bank#329926 is unchanged at4.982231460183304e-9 adjusted.
