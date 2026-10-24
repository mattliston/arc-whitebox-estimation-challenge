#!/usr/bin/env python3
"""TAG analyst A13 certificate, part 2: the truth code (mc_truth.py cert <rank>) on a PUBLIC network against the organizers' truth.
d = my 64M-sample post-activation means - the organizers' means (P2/fable_refit_v1/truth688, about 1e9 samples), per layer.
If the generator and the forward pass are right, d is pure sampling noise with covariance C_l (1/N + 1/N_org), C_l = the covariance
of the layer's post-activations (taken from the independent 32M-sample CPU-generator run fable_covhunter_v1/mcall_rank*_P32M.npz).
Statistics per layer:
  plain     mean_a d_a^2 / mean_a var_a (1/N + 1/N_org). Units are correlated, so by chance alone this scatters with relative sd
            sqrt(2 |C|_F^2) / tr C: 5.5% at layer 0 up to 19.6% at layer 15 on network 35. Reported, NOT part of the rule: a
            10% band on it would reject a correct generator most of the time at the deep layers.
  whitened  the same ratio along the principal axes of C_l (mean over axes of (u_i . d)^2 / (lambda_i (1/N + 1/N_org))): the axes
            are independent, so the chance scatter is sqrt(2 / 1024) = 4.4% at every layer.
  z_mean    d along the mean vector, in standard errors: a wrong input scale moves every unit along the mean (it would be
            invisible in the whitened ratio, where that axis is one of 1024).
  halves    the two 32M halves against each other, plain and whitened (no organizer truth involved).
  var_scale mean over units of (my per-unit variance / the 32M run's) - 1, per layer: twice the relative error of the input scale.
DECISION RULE (written before the certificate run was looked at):
  PASS      every layer's whitened ratio within [0.90, 1.10] (the lead's 10%, on the statistic where 10% is 2.3 standard errors),
            every |z_mean| < 3, and the generator's variance within 3 standard errors of 1 (rng_check*.json).
  MARGINAL  at most two layers with a whitened ratio between 10% and 13.3% off (3 standard errors), the 16-layer mean within 5%,
            the other conditions as for PASS: production starts and the lead is told which layers.
  FAIL      anything else: no production.
usage: cert_eval.py <rank> -> cert_rank{rank}.json + text"""
import sys, json, glob
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[2]; rank = int(sys.argv[1]); NORG = 1e9
c = np.load(HERE / f'cert_rank{rank:04d}.npz'); N = float(c['N']); Nh = float(c['N_half'])
org = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
mc = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
rows = []; f = 1 / N + 1 / NORG
for l in range(16):
    C = mc[f'cov_post_{l}'].astype(np.float64); lam, U = np.linalg.eigh(C); keep = lam > 1e-8 * lam.max()
    d = c['mc_post'][l] - org[l]; dab = c['mc_post_A'][l] - c['mc_post_B'][l]; var = c['mc_post_var'][l]; m = org[l]
    plain = float((d * d).mean() / (var.mean() * f)); lit = float((d * d).mean() / (var.mean() / N))
    null_plain = float(np.sqrt(2 * (C * C).sum()) / np.trace(C))
    pd = U.T @ d; white = float(((pd[keep] ** 2) / (lam[keep] * f)).mean()); null_white = float(np.sqrt(2 / keep.sum()))
    zm = float((m @ d) / np.sqrt(m @ C @ m * f))
    ab_plain = float((dab * dab).mean() / (var.mean() * 2 / Nh)); pab = U.T @ dab; ab_white = float(((pab[keep] ** 2) / (lam[keep] * 2 / Nh)).mean())
    ok = np.diag(C) > 1e-6; vr = var[ok] / np.diag(C)[ok] - 1          # dead units (variance ~ 0) left out of this ratio
    rows.append(dict(layer=l, msd=float((d * d).mean()), predicted=float(var.mean() * f), plain=plain, literal_var_over_N=lit, null_sd_plain=null_plain, whitened=white, null_sd_whitened=null_white,
                     axes=int(keep.sum()), z_mean=zm, halves_plain=ab_plain, halves_whitened=ab_white, var_scale=float(vr.mean()), var_scale_rms=float(np.sqrt((vr ** 2).mean())),
                     noise_of_pooled_mean=float(var.mean() / N)))
rng = [json.load(open(p)) for p in sorted(glob.glob(str(HERE / 'rng_check*.json')))]
gain = {Path(p).stem: json.load(open(p)) for p in sorted(glob.glob(str(HERE / 'rng_gain_test_seed*.json')))}
rng_ok = all(abs(r['var_z']) < 3 for r in rng if r['generator'] == 'mc_truth.draw')
w = np.array([r['whitened'] for r in rows]); z = np.array([r['z_mean'] for r in rows]); off = np.abs(w - 1)
if (off <= 0.10).all() and (np.abs(z) < 3).all() and rng_ok: verdict = 'PASS'
elif (off <= 0.1333).all() and (off > 0.10).sum() <= 2 and abs(w.mean() - 1) <= 0.05 and (np.abs(z) < 3).all() and rng_ok: verdict = 'MARGINAL'
else: verdict = 'FAIL'
out = dict(TAG='analyst A13 certificate', rank=rank, N=N, N_org=NORG, verdict=verdict, whitened_mean=float(w.mean()), layers_off_10pct=[int(i) for i in np.where(off > 0.10)[0]],
           plain_within_10pct=int((np.abs(np.array([r['plain'] for r in rows]) - 1) <= 0.10).sum()), rng=rng, gain_direction_tests=gain, rows=rows)
json.dump(out, open(HERE / f'cert_rank{rank:04d}.json', 'w'), indent=1)
print(f'certificate on public network {rank}: N = {N:.0f} samples of mine against the organizers\' ~1e9. VERDICT: {verdict}')
print('| layer | mean-squared difference (x1e-9) | predicted sampling noise (x1e-9) | plain ratio (chance sd) | ratio to variance / N alone | whitened ratio (chance sd 4.4%) | difference along the mean, standard errors | halves A vs B: plain / whitened | my variance vs the 32M run: mean ratio - 1 |')
print('|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print(f"| {r['layer']} | {r['msd'] * 1e9:.3f} | {r['predicted'] * 1e9:.3f} | {r['plain']:.3f} ({r['null_sd_plain'] * 100:.1f}%) | {r['literal_var_over_N']:.3f} | {r['whitened']:.3f} | {r['z_mean']:+.2f} | "
          f"{r['halves_plain']:.3f} / {r['halves_whitened']:.3f} | {r['var_scale']:+.1e} |")
print(f"whitened ratio: mean over layers {w.mean():.3f}, range {w.min():.3f}-{w.max():.3f}; layers more than 10% off: {out['layers_off_10pct'] or 'none'}; plain ratio within 10% in {out['plain_within_10pct']} of 16 layers")
print('generator checks: ' + '; '.join(f"{r['generator']}, {r['N']:.2g} values: variance - 1 = {r['var_minus_1']:+.1e} ({r['var_z']:+.1f} se), excess kurtosis {r['excess_kurtosis']:+.1e}, largest |x| {r['max_abs']:.2f}" for r in rng))
for name, g in gain.items():
    print(f"gain-direction test {name} ({g['batches']} batches of {g['batch']}): " + '; '.join(
        f"layer {l}: batch-mean variance x batch / lambda1 = {r['dir1']['batch_mean_var_x_B_over_lambda']:.3f}, lag-1 {r['dir1']['lag1']:+.3f}, mean vs organizers {r['dir1']['z_mean_vs_org']:+.2f} se" for l, r in g['layers'].items()))
