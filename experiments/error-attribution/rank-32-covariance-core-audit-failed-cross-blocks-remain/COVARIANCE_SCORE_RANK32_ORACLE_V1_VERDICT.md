# Rank32 covariance core audit: failed; cross blocks remain distinct

September6,2026,05:33:29 PDT. Total actual post13 covariance error on
mini57 gregory-wilson, qualified100M public targets; not a network score.
All six input boundary byte checks pass. Stored raw diagonal error .499718
f32ULP, post13->pre14 covariance consistency2.73797e-6 relative, conservative
f32 storage bound .027476% of total covariance error RMS. MC uncertainty
is not bounded by those storage checks and no noise correction was applied.

Input-only top32 risk-tilted bank covariance basis, oracle core coefficients:
risk residual .861707, actual next-mean response .823676, ordinary covariance
.892811. Oracle top32 tilted-error spectral basis: .328450/.369462/.137720.
Both fail the fixed .25 risk AND actual-response gate. Target covariance
RMS1.22541108e-4; target next-mean response RMS5.02260501e-5. Fixed-subspace
normal stationarity <=7.3e-15. The error spectral basis is a Frobenius
truncation comparator, not globally optimal under the trace-augmented risk.

No rank or frame sweep, new predictor, target download or carrier write.
Receipt covariance_score_rank32_oracle_v1.json; output SHA256
d57d9511acbb65d25ead1581e9f3d05a020842654ee3b51858219809007078ed.

Prior-record check CURRENT_COVARIANCE_GEOMETRY_V1.md is important: on old
rank35/pre15, top8 retained-core error captured only8.06% ordinary energy,
while including both high/complement cross blocks captured81.23%. The
present failed retained-core is likewise NOT a test of covariance subspace
rotation/tangent correction. A separate fixed-frame cross-block audit is
justified, using the already certified score metric and same32 directions;
it must not be described as reopening the failed one-high THIRD tensor
carriers, changing the frame, or making a correction learnable.
