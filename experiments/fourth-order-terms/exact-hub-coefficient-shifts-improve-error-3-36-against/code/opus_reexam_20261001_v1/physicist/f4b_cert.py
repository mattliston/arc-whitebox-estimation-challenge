#!/usr/bin/env python3
"""TAG opus-physicist F4b certificate (module f4b.py on the current torch card).
 1 strength 0 reproduces the unchanged card exactly
 2 mode 'ruleA' (the pair-implied S_m = A sum+- lam+- t+-^m pushed through the same projection, shift and anchor code) reproduces
   f4.py at the same strength: the difference of the two predictions must be at rounding level, far below F4's own effect
 3 birth geometry assumed by the formulas: the two newborn columns of a pair differ only on the hub's row, by t+ - t-, and each
   column's hub entry is t+- (so p+- = s + t+- e_h with s_h = 0)
 4 exact mode at strength 1: final-layer MSE, size of its effect, cosine with F4's effect (information, not a pass criterion)
usage: f4b_cert.py <rank> <threads>"""
import sys, json
import numpy as np, torch
import pcard as pc
import f4, f4b
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2]); tp = pc.truth(rank); W = pc.weights(rank)
out = dict(TAG='opus-physicist F4b certificate', rank=rank)
base = pc.run(pc.build(W)); out['base_final_x1e9'] = round(float(pc.mse(base, tp)[15]) * 1e9, 4)
z = pc.run(f4b.install(pc.build(W), torch, 0.0)); out['strength0_max_abs_diff'] = float(np.abs(z - base).max())
a = pc.run(f4.install(pc.build(W), torch, 0.4)); out['f4_final_x1e9'] = round(float(pc.mse(a, tp)[15]) * 1e9, 4)
# birth geometry, captured on the ruleA run
geo = dict(off_hub=0.0, gap=0.0, hub_entry=0.0); tmp = {}
cd = pc.build(W)
bd0 = cd.env['_bd_update']
def bd(bp, Cpre, K21, aux, d, alive, births, a_, t_, s_, coefficients):
    tmp['tpm'] = (t_.detach(), s_.detach()); return bd0(bp, Cpre, K21, aux, d, alive, births, a_, t_, s_, coefficients)
cd.env['_bd_update'] = bd
sel0 = cd.select
def select(score):
    r = sel0(score); tmp['sel'] = r[0]; return r
cd.select = select
f4b.install(cd, torch, 0.4, mode='ruleA')
def check(l, o, ctx):
    if l <= 14:
        p, stops = o[2], o[12]; k0 = int(stops[l - 1]) if l else 0; K = (int(stops[l]) - k0) // 2; sel = tmp['sel']; assert K == sel.numel()
        plus, minus = p[:, k0:k0 + K].to(torch.float64), p[:, k0 + K:k0 + 2 * K].to(torch.float64); tpl, tmi = tmp['tpm'][0].to(torch.float64)[sel], tmp['tpm'][1].to(torch.float64)[sel]
        diff = plus - minus; idx = torch.arange(K); hub = diff[sel, idx]; rest = diff.clone(); rest[sel, idx] = 0
        geo['off_hub'] = max(geo['off_hub'], float(rest.abs().max())); geo['gap'] = max(geo['gap'], float((hub - (tpl - tmi)).abs().max() / (tpl - tmi).abs().max()))
        geo['hub_entry'] = max(geo['hub_entry'], float((plus[sel, idx] - tpl).abs().max() / tpl.abs().max()), float((minus[sel, idx] - tmi).abs().max() / tmi.abs().max()))
    return o
pc.wrap(cd, 'state_out', check)
b = pc.run(cd); out['ruleA_final_x1e9'] = round(float(pc.mse(b, tp)[15]) * 1e9, 4)
out['ruleA_vs_f4_max_abs_diff'] = float(np.abs(b - a).max()); out['ruleA_vs_f4_rms_diff_final'] = float(np.sqrt(((b - a)[15] ** 2).mean()))
out['f4_effect_rms_final'] = float(np.sqrt(((a - base)[15] ** 2).mean())); out['f4_effect_max_abs'] = float(np.abs(a - base).max())
out['birth_geometry_max_dev'] = geo
e = pc.run(f4b.install(pc.build(W), torch, 1.0)); out['exact_c1_final_x1e9'] = round(float(pc.mse(e, tp)[15]) * 1e9, 4)
out['exact_c1_effect_rms_final'] = float(np.sqrt(((e - base)[15] ** 2).mean()))
out['exact_c1_vs_f4_effect_cos_final'] = float(((e - base)[15] * (a - base)[15]).sum() / np.linalg.norm((e - base)[15]) / np.linalg.norm((a - base)[15]))
print(json.dumps(out, indent=1))
