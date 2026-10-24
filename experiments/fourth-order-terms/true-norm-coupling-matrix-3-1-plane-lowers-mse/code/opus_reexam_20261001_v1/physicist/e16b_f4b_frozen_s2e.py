#!/usr/bin/env python3
"""TAG opus-physicist copy of the lead's O/e16b_k4_closed_loop.py (K4L; version of 2026-10-02 01:39Z) with two additions: arm options
f4b=<c>, f4banchor=0|1, f4bage=<k>, f4blo=<l>, f4bhi=<l>, f4bterms=<digits>, f4bbl=<L>, f4bmix=1 with f4bc1..4 and f4bt2..4 (module physicist/f4b.py, installed first), and the predictions of every arm
saved to k4l_pred_rank{rank}_{tag}.npz. Added 2026-10-02 08:10Z (lead's order of 07:05Z):
  nt4=<x>, nt31=<x>, ntf=<x>, ntr=<r>   ORACLE second-chaos matrix (physicist/ntor.py; t19_oracle_rank*.npz): the true network-specific part of Ntilde
              injected as per-unit k4 / K31 remainder / feed plane into the next layer's K21, at rank r (0 = full)
  f4bth1=<x>, f4bth2=<x>   F4 rule strength per age (four-entry form, with f4b=1,f4bmix=1);  X1=<value>, bscale=<value>  physics dials set to a value
  ds=<x>      the learned mean correction times x at layers 1-15;  e34=<x> the third-order cross term -k3 k4 He5(alpha) pdf / (144 sigma^6) of the mean (same slot as e2)
  e2=<x>      the lead's arm: second-order skew term of the MEAN through the learned-correction slot, layers 1-15 (the covariance is recentred for it)
  so2m=<x>    physicist/so2.py: the same mean term INSIDE the pair map at layers so2lo..min(so2hi, 14) with the consistent per-unit variance
              (E relu^2 term at strength so2r, default = so2m); so2f=<x> the mean term at layer 15 after the learned correction
  so2c=<x>    the pair map's second-order covariance list (products of two third cumulants), so2k = highest power of rho (default 1),
              so2p = digits 1-6 choosing the parts a b q c d e (default 123456);  so2t=<x> the second-order part of the third-cumulant target
  kn=<x>, kiso=<x>, k31=<x>   the constants of the covariance knob hook (gn_linear_hook: k21n, iso, k31) set to x
  bsk=<x>     the birth coefficient with the skew-corrected density at the kink: d (1 + radial/3) (1 - x (t3/6) He3(alpha))
  gsk=<x>, gk4=<x>  the gate that transports the carried third-order content (columns, old tier, inherited plane, b) with the first-order
              corrections of the mean gate: g + gsk (t3/6) He2(alpha) phi - gk4 (t4/24) He3(alpha) phi, t4 = the card's per-unit k4 / sigma^4 (versions before s2e: radial)
  bta=<x>     the variance response beta of the births: 2 (m_G - m g) -> 2 m (1 - mean gate with those corrections)
  k4all=1     K4L's age-1 part from ALL hubs of the previous layer (Wt = the weight row, Y = (C_post W - W diag C_post) / gate: the covariance
              push's intermediate product) instead of the selected hubs' bank columns; age 2 and older as before Original header: TAG opus-reexam E16b: closed-loop screen of a NON-ISOTROPIC per-unit fourth cumulant built from the exact hub coefficients
(current torch card, learned parts ON, no refit; final-layer MSE against the organizers' truth).
The card's per-unit fourth cumulant is radial * v^2 (one number per layer). Every column pair of the bank gives, elementwise,
   k4_local(i) = sum over pairs  4 S1 Y_i^3 Wt_i + 6 S2 Y_i^2 Wt_i^2 + 4 S3 Y_i Wt_i^3 + S4 Wt_i^4       (e15_local_feed.py, e16_local_k4.py)
and the arm replaces  k4 -> k4 + c * (k4_local - beta v^2),  beta = the projection of k4_local on v^2 weighted by the size of the
kurtosis term of the mean formula ((alpha^2 - 1) pdf)^2, so the trained isotropic level stays where it is. Applied at the pair map
of layers 1-14 (also ctx['k4']) and at the final layer's mean.
arms: name:c=<float>,age=<max column age, 99 = all>[,final=0/1][,pair=0/1][,f4=<theta of the physicist's F4, installed first>][,on=0 learned mean correction off]
      [,bsrc=<x>: the Sym(C x b) source radial d v/6 gets + x d dk/(6 v)] [,k22=1: the isotropic (2,2) plane scaled by sqrt(ratio_i ratio_j), ratio = new k4 / old k4]
      [,bs=<x>: the isotropic Sym(C x b) source radial d v/6 times x (the trainer's dial; 0.85 suggested)]
      [,covk4=0: the covariance output (layers 2-14, recomputed by the knob hook from ctx['k4']) keeps the isotropic k4]
      [,meank4=0: the first pair call (mean, plane target, radial) keeps the isotropic k4]
      [,lmin=<l>,lmax=<l>: layers where the correction is applied (default 1..15)]
      [,mean=0: the correction goes to the pair map's fourth-order rows only, the mean formula keeps the isotropic value (pair call split is not available at hook level: implemented as k4 for ctx only)]
usage: e16b_k4_closed_loop.py <rank> <threads> <tag> arm [arm ...]"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, torch
import pcard as pc
import f4 as F4
import f4b_frozen_fr24k as F4B
import so2_frozen_s2b as SO2
import ntor_frozen_s2e as NTOR
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2]); tag = sys.argv[3]; ARMS = sys.argv[4:]
# rank >= 7,000,000 = a generated fresh network of the analyst's production (O/analyst/fresh/loader.py: weights and 64M-sample truth by seed)
if rank >= 7000000:
    sys.path.insert(0, str(HERE / 'analyst' / 'fresh')); import loader as FL
    WEIGHTS = FL.weights(rank); tp = FL.truth(rank)['mc_post'].astype(np.float64)
else:
    WEIGHTS = rank; tp = pc.truth(rank)
t0 = time.time(); ORACLE = np.load(HERE / f't19_oracle_rank{rank:04d}.npz') if (HERE / f't19_oracle_rank{rank:04d}.npz').exists() else None


def run(spec):
    name, _, opts = spec.partition(':'); o = dict(c=0.0, age=99, final=1, pair=1, f4=0.0, on=1, bsrc=0.0, k22=0.0, lmin=1, lmax=15, covk4=1, meank4=1, bs=1.0, f4b=0.0, f4banchor=0, f4bage=2, f4blo=1, f4bhi=14, f4bterms=1234, f4bbl=-1, f4bc1=0.0, f4bc2=0.0, f4bc3=0.0, f4bc4=0.0, f4bt2=0.0, f4bt3=0.0, f4bt4=0.0, f4bmix=0, f4be2=1.0, f4bi2=0.0, f4bown=0, f4bbm=0, f4bcoll=0.0, f4bcollage=99,
             e2=0.0, so2m=0.0, so2r=-1.0, so2c=0.0, so2t=0.0, so2k=1, so2lo=1, so2hi=15, so2f=0.0, so2p=123456, k4all=0, kn=-9.0, kiso=-9.0, k31=-9.0, bsk=0.0, gsk=0.0, gk4=0.0, ds=1.0, e34=0.0, bta=0.0,
             nt4=0.0, nt31=0.0, ntf=0.0, ntr=0, f4bth1=-1.0, f4bth2=-1.0, X1=-9.0, bscale=-9.0)
    for kv in filter(None, opts.split(',')):
        k, v = kv.split('='); o[k] = float(v)
    dial_over = {k: o[k] for k in ('X1', 'bscale') if o[k] > -9}                     # physics dials set to a VALUE (candidate-2 dials: X1 0.59158, bscale 1.03414)
    cd = pc.build(WEIGHTS, learned=bool(o['on']), dials=dial_over or None); st = dict(births={}, tmp={}, calls=[], stat={}, hub={})
    if o['bsk'] or o['gsk'] or o['gk4'] or o['bta']:
        # the step's source re-executed with single lines changed (the lead's mechanism for his arm br):
        #  bsk: the birth coefficient takes the skew-corrected density at the kink, d (1 + radial/3) (1 - bsk (t3/6) He3(alpha))
        #  gsk, gk4: the gate that transports the carried third-order content (columns, old tier, planes, b) takes the first-order
        #       corrections of the mean gate, g + gsk (t3/6) He2(alpha) phi - gk4 (t4/24) He3(alpha) phi, t4 = radial
        import inspect, textwrap, types
        owner = next(k for k in type(cd).__mro__ if 'step' in k.__dict__); src = textwrap.dedent(inspect.getsource(owner.step))
        def edit(old, new):
            nonlocal src
            assert src.count(old) == 1, (src.count(old), old); src = src.replace(old, new)
        T3 = "(aux['pre_k3'] / aux['sigma'] ** 3)"; AL = "aux['alpha']"
        if o['bsk']: edit("birth = d * (1 + radial / 3)", f"birth = d * (1 + radial / 3) * (1 - {o['bsk']!r} * {T3} * ({AL} ** 3 - 3 * {AL}) / 6)")
        if o['bta']:
            # beta = d Var(relu) / d mu = 2 m (1 - mean gate): the card writes 2 (m_Gaussian - m g); bta takes it to 2 m (1 - gate with skew and kurtosis corrections)
            edit("beta = 2 * (aux['gaussian_mean'] - host * g)",
                 f"beta = 2 * (aux['gaussian_mean'] - host * g) + {o['bta']!r} * 2 * ((host - aux['gaussian_mean']) - host * ({T3} * ({AL} ** 2 - 1) - (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 3 - 3 * {AL}) / 4) * aux['pdf'] / 6)")
        if o['gsk'] or o['gk4']:
            edit("    g = aux['gate']\n    d = aux['pdf'] / aux['sigma']\n",          # the dedented source: method body at four spaces
                 "    g = aux['gate']\n    d = aux['pdf'] / aux['sigma']\n"
                 f"    gq = g + {o['gsk']!r} * {T3} * ({AL} ** 2 - 1) * aux['pdf'] / 6 - {o['gk4']!r} * (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 3 - 3 * {AL}) * aux['pdf'] / 24\n")
            edit("b = g * bpre + radial * d * vv / 6", "b = gq * bpre + radial * d * vv / 6")
            edit("inherited = g[:, None] ** 2 * g[None, :] * cp", "inherited = gq[:, None] ** 2 * gq[None, :] * cp")
            edit("p = t.cat((alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * pp),", "p = t.cat((alive.to(t.float32)[:, None] * (gq.to(t.float32)[:, None] * pp),")
            edit("Q = alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * U)", "Q = alive.to(t.float32)[:, None] * (gq.to(t.float32)[:, None] * U)")
        gl_ = dict(sys.modules[owner.__module__].__dict__); exec(src, gl_); cd.step = types.MethodType(gl_['step'], cd)
    if o['kn'] > -9 or o['kiso'] > -9 or o['k31'] > -9:
        # the three constants of the covariance knob hook (iso -0.094, k21n -0.028, k31 -0.212 in the research parent), re-installed with other values
        th = dict(pc.LC.gl.V3); th.update({k: o[n] for k, n in (('k21n', 'kn'), ('iso', 'kiso'), ('k31', 'k31')) if o[n] > -9}); pc.LC.gl.install(cd, torch, th)
    if o['f4']: F4.install(cd, torch, float(o['f4']), age=2, anchor='cheap')
    if o['f4b']: F4B.install(cd, torch, float(o['f4b']), age=int(o['f4bage']), anchor=int(o['f4bbl']) if o['f4bbl'] >= 0 else ('b' if o['f4banchor'] else 'star'), layers=(int(o['f4blo']), int(o['f4bhi'])),
                              terms=tuple(int(ch) for ch in str(int(o['f4bterms']))), mode='mix' if o['f4bmix'] else 'exact',
                              cs=tuple(o[k] for k in ('f4bc1', 'f4bc2', 'f4bc3', 'f4bc4')), ths=({1: (o['f4bth1'],) * 4, 2: (o['f4bth2'],) * 4} if o['f4bth1'] >= 0 else tuple(o[k] for k in ('f4bt2', 'f4bt3', 'f4bt4'))), even2=o['f4be2'], iso2=o['f4bi2'], form='own' if o['f4bown'] else 'ride', metric='birth' if o['f4bbm'] else 'gram', coll=o['f4bcoll'], coll_age=int(o['f4bcollage']))
    if o['so2m'] or o['so2c'] or o['so2t'] or o['so2f'] or o['so2r'] > 0:
        SO2.install(cd, torch, mean=o['so2m'], raw2=None if o['so2r'] < 0 else o['so2r'], cov=o['so2c'], tgt=o['so2t'], kmax=int(o['so2k']), layers=(int(o['so2lo']), int(o['so2hi'])),
                    parts=''.join('abqcde'[int(ch) - 1] for ch in str(int(o['so2p']))), final=o['so2f'])
    push0 = cd.shared['push']
    def push(t, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
        r = push0(t, layer, w, p, lam, previous, Q, F, core, stops, rank=rank); st['pp'] = r[0]; st['stops'] = (0,) + tuple(int(x) for x in stops); return r
    cd.shared['push'] = push
    bd0 = cd.env['_bd_update']
    def bd(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
        st['tmp'] = dict(plus=t.detach(), minus=s.detach()); return bd0(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients)
    cd.env['_bd_update'] = bd
    sel0 = cd.select
    def select(score):
        r = sel0(score); st['calls'].append(r[0].detach()); return r
    cd.select = select
    def local(l, vv, mu):
        pp = st['pp'].to(torch.float64); stops = st['stops']; loc = torch.zeros(1024, dtype=torch.float64)
        if o['k4all'] and (l - 1) in st['hub']:
            # age 1 from ALL hubs e of layer l-1: Wt_e(i) = W[e, i], Y_e(i) = sum_{a != e} C_post[e, a] W[a, i] / gate_e (the card forms C_post W for the covariance push)
            H = st['hub'][l - 1]; w64 = cd.ws[l]; M = (H['C'].to(torch.float32) @ cd.w[l]).to(torch.float64)
            Wt = (w64 * H['alive'][:, None]).T; Y = ((M - w64 * H['cdiag'][:, None]) * (H['alive'] / H['g'].clamp_min(1e-3))[:, None]).T; S1, S2, S3, S4 = H['S']
            loc = loc + (4 * S1[None, :] * Y ** 3 * Wt + 6 * S2[None, :] * Y ** 2 * Wt ** 2 + 4 * S3[None, :] * Y * Wt ** 3 + S4[None, :] * Wt ** 4).sum(1)
        for j in range(l):
            if l - j > o['age']: continue
            if o['k4all'] and j == l - 1 and (l - 1) in st['hub']: continue
            B = st['births'][j]; k0 = stops[j]; K = (stops[j + 1] - k0) // 2; assert K == len(B['tp']), (l, j, K, len(B['tp']))
            gap = (B['tp'] - B['tm'])[None, :]; pplus = pp[:, k0:k0 + K]; pminus = pp[:, k0 + K:k0 + 2 * K]
            Wt = (pplus - pminus) / gap; Y = (B['tp'][None, :] * pminus - B['tm'][None, :] * pplus) / gap; S1, S2, S3, S4 = B['S']
            loc = loc + (4 * S1[None, :] * Y ** 3 * Wt + 6 * S2[None, :] * Y ** 2 * Wt ** 2 + 4 * S3[None, :] * Y * Wt ** 3 + S4[None, :] * Wt ** 4).sum(1)
        al = mu / vv.clamp_min(1e-12).sqrt(); w4 = ((al * al - 1) * torch.exp(-al * al / 2)) ** 2 + 1e-12; iso = vv * vv
        beta = (w4 * loc * iso).sum() / (w4 * iso * iso).sum(); dk = loc - beta * iso
        st['stat'][l] = (float(beta), float((dk * dk * w4).sum().sqrt() / (w4 * iso * iso).sum().sqrt()))
        return dk
    def pin_hook(l, pin, ctx):
        if l >= 1 and o['c'] != 0.0 and o['pair'] and o['lmin'] <= l <= o['lmax']:
            dk = o['c'] * local(l, torch.diagonal(ctx['Cpre']), pin[0]); st['dk'] = dk
            k22 = pin[4]
            if o['k22']:
                ratio = ((pin[3] + dk) / pin[3].clamp_min(1e-30)).clamp_min(0.0).sqrt(); k22 = k22 * ratio[:, None] * ratio[None, :]
            # the card computes the pair map twice at layers 2-14: the first call (pin) gives the mean, the plane target and the radial;
            # the covariance-knob hook (gn3) recomputes the covariance output from ctx['k4'].
            if o['covk4']: ctx['k4'] = ctx['k4'] + dk
            if o['meank4']: pin = (pin[0], pin[1], pin[2], pin[3] + dk, k22, pin[5])
        return pin
    pc.wrap(cd, 'pair_in', pin_hook)
    def b_hook(l, b, ctx):
        if o['bsrc'] and l >= 1 and 'dk' in st: b = b + o['bsrc'] * ctx['d'] * st['dk'] / (6 * ctx['vv'].clamp_min(1e-12))
        if o['bs'] != 1.0: b = b + (o['bscale'] if o['bscale'] > -9 else pc.DIALS['bscale']) * (o['bs'] - 1) * ctx['radial'] * ctx['d'] * ctx['vv'] / 6     # scale of the isotropic Sym(C x b) source (the physicist's bsrc)
        return b
    pc.wrap(cd, 'b_post', b_hook)
    def k415(l, k4, ctx):
        if o['c'] != 0.0 and o['final'] and o['lmax'] >= 15: k4 = k4 + o['c'] * local(15, ctx['vv'], ctx['mu'])
        return k4
    pc.wrap(cd, 'k4_15', k415)
    def out_hook(l, out, ctx):
        if l <= 14:
            aux = ctx['aux']; sel = st['calls'][-1]; f = lambda x: x.detach().to(torch.float64)[sel]
            a, s, gt, ph, m = f(aux['alpha']), f(aux['sigma']), f(aux['gate']), f(aux['pdf']), f(ctx['host']); vv = s * s; dj = ph / s
            r2 = vv * ((1 + a * a) * gt + a * ph); r3 = vv * s * ((a ** 3 + 3 * a) * gt + (a * a + 2) * ph); r4 = vv * vv * ((a ** 4 + 6 * a * a + 3) * gt + (a ** 3 + 5 * a) * ph)
            S = (-a * ph / vv, 2 * (gt - gt * gt - m * dj), 3 * (1 - gt) * (r2 - 2 * m * m), r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2)
            st['births'][l] = dict(tp=f(st['tmp']['plus']), tm=f(st['tmp']['minus']), S=S)
            if o['k4all']:
                sel = slice(None); a, s, gt, ph, m = f(aux['alpha']), f(aux['sigma']), f(aux['gate']), f(aux['pdf']), f(ctx['host']); vv = s * s; dj = ph / s
                r2 = vv * ((1 + a * a) * gt + a * ph); r3 = vv * s * ((a ** 3 + 3 * a) * gt + (a * a + 2) * ph); r4 = vv * vv * ((a ** 4 + 6 * a * a + 3) * gt + (a ** 3 + 5 * a) * ph)
                S = (-a * ph / vv, 2 * (gt - gt * gt - m * dj), 3 * (1 - gt) * (r2 - 2 * m * m), r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2)
                Cp = out[1].detach().to(torch.float64); st['hub'] = {l: dict(S=S, g=gt, alive=out[6].detach().to(torch.float64), C=Cp, cdiag=torch.diagonal(Cp).clone())}
        st['calls'].clear()
        return out
    pc.wrap(cd, 'state_out', out_hook)
    if o['nt4'] or o['nt31'] or o['ntf']:
        NTOR.install(cd, torch, {l: ORACLE[f'R_L{l}'] for l in range(1, 16)}, k4=o['nt4'], k31=o['nt31'], feed=o['ntf'], rank=int(o['ntr']))
    if o['ds'] != 1.0: pc.wrap(cd, 'delta', lambda l, d, ctx: d if l == 0 else d * o['ds'])      # the learned mean correction times ds at layers 1-15 (before the terms below are added)
    if o['e2'] or o['e34']:
        # the lead's arm (O/e16b_k4_closed_loop.py): (k3^2 / 72) E f^(6) = e2 * k3^2 He4(alpha) pdf / (72 sigma^5) added to the mean through the learned-correction slot;
        # e34: the third-order cross term (k3 k4 / 144) E f^(7) = - e34 * k3 k4 He5(alpha) pdf / (144 sigma^6), same slot (T16: worth as much as e2 with true inputs)
        def e2_hook(l, d, ctx):
            if l == 0: return d
            aux = ctx['aux']; a = aux['alpha']; k3 = aux['pre_k3']; k4 = aux['pre_k4']
            return (d + o['e2'] * (k3 * k3 / 72) * (a ** 4 - 6 * a * a + 3) * aux['pdf'] / aux['sigma'] ** 5
                    - o['e34'] * (k3 * k4 / 144) * (a ** 5 - 10 * a ** 3 + 15 * a) * aux['pdf'] / aux['sigma'] ** 6)
        pc.wrap(cd, 'delta', e2_hook)
    pred = pc.run(cd); mse = pc.mse(pred, tp); PRED[name] = pred; INFO[name] = getattr(cd, 'so2_info', {}); NTI[name] = getattr(cd, 'ntor_info', {})
    return name, o, mse, st['stat']


base = None; PRED = {}; INFO = {}; NTI = {}
for spec in ARMS:
    name, o, mse, stat = run(spec)
    if base is None: base = mse
    print(json.dumps(dict(rank=rank, tag=tag, arm=name, opts=o, final_x1e9=round(float(mse[15]) * 1e9, 4), final_vs_first_arm_pct=round(100 * (float(mse[15]) / float(base[15]) - 1), 3),
                          all_layer_vs_first_arm_pct=round(100 * (float(mse.mean()) / float(base.mean()) - 1), 3),
                          per_layer_vs_first_arm_pct=[round(100 * (float(mse[i]) / float(base[i]) - 1), 2) for i in range(16)], per_layer_x1e9=[round(float(x) * 1e9, 2) for x in mse],
                          beta={k: round(v[0], 4) for k, v in stat.items() if k in (5, 9, 13, 15)}, dk_rel={k: round(v[1], 3) for k, v in stat.items() if k in (5, 9, 13, 15)},
                          so2={k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in INFO[name].items() if k in (2, 5, 9, 13, 14, 15)},
                          ntor={k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in NTI[name].items() if k in (2, 5, 9, 13, 14)},
                          wall=round(time.time() - t0, 1))), flush=True)
np.savez(HERE / f'k4l_pred_rank{rank:04d}_{tag}.npz', truth=tp, **PRED)
