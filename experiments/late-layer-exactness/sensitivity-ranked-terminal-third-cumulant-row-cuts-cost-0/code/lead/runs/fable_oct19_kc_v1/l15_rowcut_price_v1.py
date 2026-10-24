"""TAG fable-lead 2026-10-20: price list of the terminal k3 rows from the native probe (l15probe_rank<r>.npz): the k3 term of the
scored mean is c_i = k3_i e_i / 6 (e = -a phi(a) / v); dropping row i's k3 changes the net's MSE by d_i = (c_i^2 - 2 c_i r_i) / n with
r = y15 - truth. Bill per row from the fired band (z 3.5 -> 3.0 = -0.295% of the bill for the rows with 3.0 < z <= 3.5).
Rankings: ORACLE (by d_i), PROXY-T (diag transport k3hat = sum_a W_ia^3 post_k3_a(L14), loss (k3hat e/6)^2), LOWZ (drop z < z_low),
SENS (|e| ascending). Prints raw cost (% of MSE) at bill savings 0.1 / 0.2 / 0.3 / 0.5 / 0.75 / 1.0 %."""
import sys, numpy as np
from scipy.stats import norm
ranks = [int(x) for x in sys.argv[1:]] or [35, 100, 37, 39]
BAND_BILL = 0.295
for rank in ranks:
    d = np.load('l15probe_rank%d.npz' % rank)
    mu, v, k3, y, t = d['mu15'], d['v15'], d['k3_15'], d['y15'], d['truth15']
    s = np.sqrt(np.maximum(v, 1e-10)); a = mu / s; z = np.abs(a)
    e = -a * norm.pdf(a) / v; c = k3 * e / 6; r = y - t; n = len(mu)
    mse = np.mean(r ** 2)
    kept = z <= 3.0; band = (z > 3.0) & (z <= 3.5)
    bill_per_row = BAND_BILL / max(1, band.sum())
    dmse = (c ** 2 - 2 * c * r) / n            # exact change of the net's MSE if row i's k3 is dropped (first-order in the head)
    W = d['W15']
    post_k3 = d['l14_post_k3']
    k3hat = (W ** 3).T @ post_k3 if W.shape == (1024, 1024) else None
    # orientation: the card uses mu = W^T m, so unit i of L15 is column i of W15
    k3hat_T = (W ** 3).T @ post_k3; k3hat_N = (W ** 3) @ post_k3
    ck = max((np.corrcoef(k3hat_T[kept], k3[kept])[0, 1], 'T', k3hat_T), (np.corrcoef(k3hat_N[kept], k3[kept])[0, 1], 'N', k3hat_N), key=lambda x: abs(x[0]))
    k3hat = ck[2]
    print('rank %d: MSE %.4f e-9, kept rows %d (z<=3), band rows %d -> bill/row %.4f%%; k3 term rms %.2e vs error rms %.2e; k3 share of MSE if ALL kept rows dropped: %+.1f%%; corr(k3hat diag-transport, k3) %.3f (%s), corr(|k3hat|,|k3|) %.3f'
          % (rank, mse * 1e9, kept.sum(), band.sum(), bill_per_row, np.sqrt((c[kept] ** 2).mean()), r.std(), 100 * dmse[kept].sum() / mse, ck[0], ck[1], np.corrcoef(np.abs(k3hat[kept]), np.abs(k3[kept]))[0, 1]))
    idx = np.where(kept)[0]
    rankings = {
        'ORACLE  ': idx[np.argsort(dmse[idx])],
        'PROXY-T ': idx[np.argsort((k3hat[idx] * e[idx] / 6) ** 2)],
        'LOWZ    ': idx[np.argsort(z[idx])],
        'SENS    ': idx[np.argsort(np.abs(e[idx]))],
        'PROXY-S ': idx[np.argsort((np.abs(k3hat[idx]) * np.abs(e[idx])))],
    }
    for name, order in rankings.items():
        cum = np.cumsum(dmse[order]) / mse * 100
        out = []
        for b in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            k = int(round(b / bill_per_row))
            if k < len(order): out.append('%.2f%%bill: raw %+.3f%% (%d rows, zmax %.2f)' % (b, cum[k - 1], k, z[order[:k]].max()))
        print('   %s %s' % (name, ' | '.join(out)))
