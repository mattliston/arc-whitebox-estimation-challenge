#!/usr/bin/env python3
"""TAG fable-covhunter (CH-28b): POSITIVE CONTROL for final_fit.py. A known signal is injected into e (a linear one in standardized
alpha and skew, and a nonlinear one: tanh of skew times the kink indicator), scaled so the true R2_0 of the signal is ~0.08 (the bar).
The same ridge / GBM pipeline must recover it on the held-out VALID nets, or the null of final_fit.py is vacuous."""
import numpy as np
exec(open(__file__.replace('final_fit_posctl.py', 'final_fit.py')).read().split("sc = StandardScaler().fit(X[tr])")[0])
rng = np.random.default_rng(1)
ia, isk, ik = names.index('alpha'), names.index('skew'), names.index('kink')
for kind in ('linear', 'nonlinear'):
    z = (X[:, ia] - X[:, ia].mean()) / X[:, ia].std() + (X[:, isk] - X[:, isk].mean()) / X[:, isk].std() if kind == 'linear' \
        else np.tanh(3 * (X[:, isk] - X[:, isk].mean()) / X[:, isk].std()) * X[:, ik]
    sig = z * S; sig *= np.sqrt(0.087 * np.sum(E ** 2) / np.sum(sig ** 2))   # signal energy ~8.7% of the error energy
    E2 = E + sig; Y2 = E2 / S
    r2_true = 1 - np.sum((E2 - sig) ** 2) / np.sum(E2 ** 2)
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    rid = Ridge(alpha=1e4).fit(Xs[tr], Y2[tr]); eh = rid.predict(Xs[va]) * S[va]
    r2r = 1 - np.sum((E2[va] - eh) ** 2) / np.sum(E2[va] ** 2)
    gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(X[tr], Y2[tr])
    r2g = 1 - np.sum((E2[va] - gbm.predict(X[va]) * S[va]) ** 2) / np.sum(E2[va] ** 2)
    print(f'{kind:9s} injected signal: oracle R2_0 {r2_true:.3f} | recovered on VALID: ridge {r2r:+.3f}  GBM {r2g:+.3f}')
