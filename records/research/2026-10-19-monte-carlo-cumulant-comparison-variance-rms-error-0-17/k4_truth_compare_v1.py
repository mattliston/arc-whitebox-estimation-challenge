"""TAG fable-lead 2026-10-19: per-unit comparison of the card's k4 internals (probe_k4dump_v1) with Monte Carlo truth (mc_cumulants_v1)
for one fresh seed. Per layer: alignment check (corr of mu), variance error, k4 model vs truth (isotropic part and total), the
truth-optimal scalar scale of the model k4, and what per-unit features the k4 error correlates with.
usage: k4_truth_compare_v1.py <dump npz> <mc npz>"""
import sys

import numpy as np

D = np.load(sys.argv[1]); M = np.load(sys.argv[2])
mu_t, var_t, k3_t, k4_t = M['mu'], M['var'], M['k3'], M['k4']
layers = sorted({int(k.split('_')[0][1:]) for k in D.files if k.startswith('L')})
# alignment: card layer l vs MC layer l, l-1, l+1
for l in (4, 8):
    mu_c = D['L%d_pre_mu' % l]
    print('align card L%d: corr with MC L%d %.4f, L%d %.4f, L%d %.4f' % (l, l - 1, np.corrcoef(mu_c, mu_t[l - 1])[0, 1], l, np.corrcoef(mu_c, mu_t[l])[0, 1], l + 1, np.corrcoef(mu_c, mu_t[l + 1])[0, 1]))
print('layer | var rel err (mean, rms) | ex-kurt truth mean (unit sd) | model iso mean | model total mean | corr(total,truth) | s* iso | s* total | corr(err, z^2) corr(err, vv) corr(err, k3n^2)')
for l in layers:
    mu_c, vv_c = D['L%d_pre_mu' % l], D['L%d_pre_vv' % l]
    k4_iso, k4_tot = D['L%d_band_k4' % l], D['L%d_total_k4' % l]
    vt, kt = var_t[l], k4_t[l]
    ex_t = kt / vt ** 2; ex_iso = k4_iso / vv_c ** 2; ex_tot = k4_tot / vv_c ** 2
    verr = (vv_c - vt) / vt
    z = mu_c / np.sqrt(vv_c); k3n = k3_t[l] / vt ** 1.5
    err = k4_tot - kt
    s_iso = float(np.dot(k4_iso, kt) / np.dot(k4_iso, k4_iso)); s_tot = float(np.dot(k4_tot, kt) / np.dot(k4_tot, k4_tot))
    print('L%2d | %+.4f %.4f | %+.4f (%.4f) | %+.4f | %+.4f | %.3f | %.3f | %.3f | %+.2f %+.2f %+.2f'
          % (l, verr.mean(), np.sqrt((verr ** 2).mean()), ex_t.mean(), ex_t.std(), ex_iso.mean(), ex_tot.mean(), np.corrcoef(k4_tot, kt)[0, 1], s_iso, s_tot,
             np.corrcoef(err, z * z)[0, 1], np.corrcoef(err, vv_c)[0, 1], np.corrcoef(err, k3n * k3n)[0, 1]))
