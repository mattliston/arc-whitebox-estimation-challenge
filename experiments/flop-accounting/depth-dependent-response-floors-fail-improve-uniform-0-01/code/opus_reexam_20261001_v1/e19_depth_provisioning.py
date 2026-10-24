#!/usr/bin/env python3
"""TAG opus-reexam E19 (copy of E16b + depth-dependent selection floors; see the options rf*/bf* and the counts in the output).
Original header of E16b: closed-loop screen of a NON-ISOTROPIC per-unit fourth cumulant built from the exact hub coefficients
(current torch card, learned parts ON, no refit; final-layer MSE against the organizers' truth).
The card's per-unit fourth cumulant is radial * v^2 (one number per layer). Every column pair of the bank gives, elementwise,
   k4_local(i) = sum over pairs  4 S1 Y_i^3 Wt_i + 6 S2 Y_i^2 Wt_i^2 + 4 S3 Y_i Wt_i^3 + S4 Wt_i^4       (e15_local_feed.py, e16_local_k4.py)
and the arm replaces  k4 -> k4 + c * (k4_local - beta v^2),  beta = the projection of k4_local on v^2 weighted by the size of the
kurtosis term of the mean formula ((alpha^2 - 1) pdf)^2, so the trained isotropic level stays where it is. Applied at the pair map
of layers 1-14 (also ctx['k4']) and at the final layer's mean.
arms: name:c=<float>,age=<max column age, 99 = all>[,final=0/1][,pair=0/1][,f4=<theta of the physicist's F4, installed first>][,on=0 learned mean correction off]
      [,bsrc=<x>: the Sym(C x b) source radial d v/6 gets + x d dk/(6 v)] [,k22=1: the isotropic (2,2) plane scaled by sqrt(ratio_i ratio_j), ratio = new k4 / old k4]
      [,ds=<x>: learned mean correction times x at layers 1-15] [,rcx=<x>: recentring strength times x]
      [,bs=<x>: the isotropic Sym(C x b) source radial d v/6 times x (the trainer's dial; 0.85 suggested)]
      [,covk4=0: the covariance output (layers 2-14, recomputed by the knob hook from ctx['k4']) keeps the isotropic k4]
      [,meank4=0: the first pair call (mean, plane target, radial) keeps the isotropic k4]
      [,lmin=<l>,lmax=<l>: layers where the correction is applied (default 1..15)]
      [,mean=0: the correction goes to the pair map's fourth-order rows only, the mean formula keeps the isotropic value (pair call split is not available at hook level: implemented as k4 for ctx only)]
usage: e16b_k4_closed_loop.py <rank> <threads> <tag> arm [arm ...]"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
import f4 as F4
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2]); tag = sys.argv[3]; ARMS = sys.argv[4:]
# rank >= 7,000,000 = a generated fresh network of the analyst's production (O/analyst/fresh/loader.py: weights and 64M-sample truth by seed)
if rank >= 7000000:
    sys.path.insert(0, str(HERE / 'analyst' / 'fresh')); import loader as FL
    WEIGHTS = FL.weights(rank); tp = FL.truth(rank)['mc_post'].astype(np.float64)
else:
    WEIGHTS = rank; tp = pc.truth(rank)
t0 = time.time()


def run(spec):
    name, _, opts = spec.partition(':'); o = dict(c=0.0, age=99, final=1, pair=1, f4=0.0, on=1, bsrc=0.0, k22=0.0, lmin=1, lmax=15, covk4=1, meank4=1, bs=1.0, ds=1.0, rcx=1.0, rfe=0.01, rfm=0.01, rfl=0.01, bfe=0.01, bfm=0.01, bfl=0.01)
    for kv in filter(None, opts.split(',')):
        k, v = kv.split('='); o[k] = float(v)
    cd = pc.build(WEIGHTS, learned=bool(o['on'])); st = dict(births={}, tmp={}, calls=[], stat={})
    base_select = cd.select
    def floor_select(score):
        l = cd.call_layer
        bfloor = o['bfe'] if l <= 5 else (o['bfm'] if l <= 9 else o['bfl'])
        if l <= 14 and st.get('births_next') and bfloor != 0.01: return cd.select_floor(score, bfloor)
        return base_select(score)
    cd.select = floor_select
    if o['f4']: F4.install(cd, torch, float(o['f4']), age=2, anchor='cheap')
    push0 = cd.shared['push']
    def push(t, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
        r = push0(t, layer, w, p, lam, previous, Q, F, core, stops, rank=rank); st['pp'] = r[0]; st['stops'] = (0,) + tuple(int(x) for x in stops); return r
    cd.shared['push'] = push
    bd0 = cd.env['_bd_update']
    def bd(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
        st['tmp'] = dict(plus=t.detach(), minus=s.detach()); return bd0(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients)
    cd.env['_bd_update'] = bd
    sel0 = cd.select; st['counts'] = {}
    def select(score):
        # slot 0 = alive units, slot 1 = response rows (when the knob is at its default), slot 2 = birth hubs. E19: per-depth floors.
        l = cd.call_layer
        r = sel0(score)
        if st.get('births_next'): st['counts'].setdefault(l, {})['births'] = int(len(r[0]))
        st['births_next'] = False
        st['calls'].append(r[0].detach()); return r
    cd.select = select
    def mu_hook(l, v, ctx):
        rf = o['rfe'] if l <= 6 else (o['rfm'] if l <= 10 else o['rfl'])
        cd.knobs['resp_floor'] = rf if l >= 1 else .01
        return v
    pc.wrap(cd, 'mu', mu_hook)
    def bpost_mark(l, b, ctx):
        st['births_next'] = True          # the next select call of this step is the births selection (it follows the b_post hook)
        return b
    pc.wrap(cd, 'b_post', bpost_mark)
    def pin_counts(l, pin, ctx):
        st['counts'].setdefault(l, {}).update(resp=int(ctx['response_mask'].sum()), alive=int(ctx['alive'].sum()), cols=int(st['pp'].shape[1]) if 'pp' in st else 0, stops=list(st.get('stops', ())))
        return pin
    pc.wrap(cd, 'pair_in', pin_counts)
    def local(l, vv, mu):
        pp = st['pp'].to(torch.float64); stops = st['stops']; loc = torch.zeros(1024, dtype=torch.float64)
        for j in range(l):
            if l - j > o['age']: continue
            B = st['births'][j]; k0 = stops[j]; K = (stops[j + 1] - k0) // 2; assert K == len(B['tp']), (l, j, K, len(B['tp']))
            gap = (B['tp'] - B['tm'])[None, :]; pplus = pp[:, k0:k0 + K]; pminus = pp[:, k0 + K:k0 + 2 * K]
            Wt = (pplus - pminus) / gap; Y = (B['tp'][None, :] * pminus - B['tm'][None, :] * pplus) / gap; S1, S2, S3, S4 = B['S']
            loc = loc + (4 * S1[None, :] * Y ** 3 * Wt + 6 * S2[None, :] * Y ** 2 * Wt ** 2 + 4 * S3[None, :] * Y * Wt ** 3 + S4[None, :] * Wt ** 4).sum(1)
        al = mu / vv.clamp_min(1e-12).sqrt(); w4 = ((al * al - 1) * torch.exp(-al * al / 2)) ** 2 + 1e-12; iso = vv * vv
        beta = (w4 * loc * iso).sum() / (w4 * iso * iso).sum(); dk = loc - beta * iso
        st.setdefault('k4stat', {})[l] = (float(beta), float((dk * dk * w4).sum().sqrt() / (w4 * iso * iso).sum().sqrt()))
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
        if o['bs'] != 1.0: b = b + pc.DIALS['bscale'] * (o['bs'] - 1) * ctx['radial'] * ctx['d'] * ctx['vv'] / 6     # scale of the isotropic Sym(C x b) source (the physicist's bsrc)
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
        st['calls'].clear()
        return out
    pc.wrap(cd, 'state_out', out_hook)
    if o['ds'] != 1.0: pc.wrap(cd, 'delta', lambda l, d, ctx: d if l == 0 else d * o['ds'])      # shrink the learned mean correction (layers 1-15)
    if o['rcx'] != 1.0: cd.knobs['recenter'] = float(cd.knobs['recenter']) * o['rcx']               # recentring strength times x
    pred = pc.run(cd); mse = pc.mse(pred, tp)
    # cost proxies: read ~ response rows x alive columns x bank columns; push ~ 1024 x alive(previous) x young columns (+ 384 for the old tier)
    C = st['counts']; read = sum(C[l]['resp'] * C[l]['alive'] * C[l]['cols'] for l in C if 'resp' in C[l] and l >= 1)
    young = {}
    for l in C:
        if 'stops' not in C[l] or l < 1: continue
        stp = C[l]['stops']; joined = max([c_ for L_, cs in {7: [0, 1, 2], 10: [3, 4, 5], 13: [6, 7, 8]}.items() if L_ <= l for c_ in cs], default=-1)
        young[l] = stp[-1] - (stp[joined + 1] if joined >= 0 else 0)
    push = sum(1024 * C[l - 1].get('alive', 1024) * young[l] if l - 1 in C else 1024 * 1024 * young[l] for l in young)
    st['stat'] = dict(read=read, push=push, resp=[C[l].get('resp') for l in sorted(C) if 1 <= l <= 14], births=[C[l].get('births') for l in sorted(C) if l <= 14])
    return name, o, mse, st['stat']


base = None
for spec in ARMS:
    name, o, mse, stat = run(spec)
    if base is None: base = mse
    print(json.dumps(dict(rank=rank, tag=tag, arm=name, opts=o, final_x1e9=round(float(mse[15]) * 1e9, 4), final_vs_first_arm_pct=round(100 * (float(mse[15]) / float(base[15]) - 1), 3),
                          all_layer_vs_first_arm_pct=round(100 * (float(mse.mean()) / float(base.mean()) - 1), 3),
                          per_layer_vs_first_arm_pct=[round(100 * (float(mse[i]) / float(base[i]) - 1), 2) for i in range(16)], per_layer_x1e9=[round(float(x) * 1e9, 2) for x in mse],
                          read_proxy=stat['read'], push_proxy=stat['push'], resp_rows=stat['resp'], births=stat['births'],
                          wall=round(time.time() - t0, 1))), flush=True)
