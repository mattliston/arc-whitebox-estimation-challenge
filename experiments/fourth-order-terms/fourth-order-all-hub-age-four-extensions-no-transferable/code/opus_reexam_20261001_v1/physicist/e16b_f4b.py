#!/usr/bin/env python3
"""TAG opus-physicist copy of the lead's O/e16b_k4_closed_loop.py (K4L; version of 2026-10-02 01:39Z) with two additions: arm options
f4b=<c>, f4banchor=0|1, f4bage=<k>, f4blo=<l>, f4bhi=<l>, f4bterms=<digits>, f4bbl=<L>, f4bmix=1 with f4bc1..4 and f4bt2..4 (module physicist/f4b.py, installed first), and the predictions of every arm
saved to k4l_pred_rank{rank}_{tag}.npz. Added 2026-10-02 08:10Z (lead's order of 07:05Z):
  nt4=<x>, nt31=<x>, ntf=<x>, ntr=<r>   ORACLE second-chaos matrix (physicist/ntor.py; t19_oracle_rank*.npz): the true network-specific part of Ntilde
              injected as per-unit k4 / K31 remainder / feed plane into the next layer's K21, at rank r (0 = full);
              nto=<k>: the matrix of a recursion instead of the truth (t19i_recursion.py: 1 stale1T, 2 stale1, 3 stale2, 4 free, 5 freeT; t19j_forms.py: 6 x1, 7 x2, 8 x3; t19p_freerun.py (complete law): 9 law1, 10 lawF, 11 lawFk, 12 lawF0);
              ntx=1: the matrix REPLACES the card's isotropic K31 (for x1, x2, x3; use with k31=0)
  f4bth1=<x>, f4bth2=<x>   F4 rule strength per age (four-entry form, with f4b=1,f4bmix=1);  X1=<value>, bscale=<value>  physics dials set to a value
  nc=<strength>, ncr=<rank>, nck=0|1, ncskip=<bit mask: 1 gain, 2 rows, 4 vtilt, 8 own31, 16 k3c>, nclo, nchi,
              ncv=<bit mask of the conversion's parts: 1 bank, 2 star, 4 b mode>, nc31=1 (isotropic V-contraction of K31),
              ncage=<a>: conversion from the cohorts born at most a layers ago (-1: the young tier), ncsr=<r>: state cut to rank r, ncl0=<l>: law starts at layer l; ncpart=<m>: state in the partner form on the top m covariance
              eigenvectors; ncforms=1: the K31 plane removes only the multiple of C; ncb=<x>: the bank channel through ncar_bank.py (the module form with the lead's carrier interface; ncbconv 1 / 2 / 3 = all / young / old columns, ncbt = transport factor; ncbcol=1: the native form, a second weight per bank column; ncbtop=<m>: plane from the m heaviest columns); ncbas=<0..5>: basis of the ncos cut: 0 pushed columns + eigenvectors (TC), 1 TGRC, 2 TGR, 3 TGC, 4 GRC, 5 T (G / R = the previous layer's top-K eigenvectors gated / row-scaled and pushed); ncritz=1: Rayleigh-Ritz instead of one-sided; ncck=<K>: covariance inside the law = diagonal + top-K eigenpairs; nc6=<x>: the lead's pair-carried law (audit copy ncar6_audit.py) at strength x through remainder31 (0 = passive: carried, dumped, compared with the full-matrix carrier if ncpas=1 or nc is set, and with the oracle); tk21 / tk31 / tk22 = <x>: ORACLE, the Monte Carlo truth plane (8M samples; networks 35, 100; layers 2-14) in place of the card's off-diagonal K21 / whole K31 slot / K22 at weight x (tkrows=1: kink rows only; tkdraw=1: the second draw); set the matching covariance knobs to 0 (kn, kiso for K21; k31 for K31); budget=1: after the arm's run, the T8 budget on the same card: variance / skew / kurtosis of the per-unit mean formula made exact (32M truths, networks 35 and 100 only) at every layer; ncpas=1: my carrier passive; nc6g=1 / 2: the gain term carried on the pairs (kappa by the Frobenius projection / from the diagonals); nc6lev=1: fold inside the read; nc6df: dfit (2 the lead's table, 5 / 6 a constant regressor); ncbx=1: the bank channel is carried exactly and the state cuts act on the rest; nfo=<x>: the bank channel folded into K21 (ncar_fold.py; nfolev 0 = into the pair map's K21, 1 = inside the bank read, 2 = as remainder31; nforows 1 = response rows, 0 = all; nfoforms; nfocheck=1: first-order identity check at layers 5, 9, 13; nfod=3 / 4: form coefficients from a regression of the diagonal, nfods = factor on its tr C x C coefficient; nfovb=1: weights from the card's beta; nfodr=1: diagonal regression on the response rows; nfot / nfosrc: transport / source factors; nfok22 / nfok4: per-unit uses of the diagonal); ncown=<x>: the unit's own term inside the mean-tilt row form (0 = the certified law, default from frozen s2r on; 1 = what frozen s2k..s2q ran); ncos=<r>, ncosk=<K>: state cut to rank r inside the one-sided form on [pushed previous columns | top-K covariance eigenvectors];
              ncskip bits 32 / 64 / 128 / 256: off-diagonal K21 terms dropped / the two K21 x C product groups / the K21 x per-unit-vector products / the elementwise K3V term
              (lean law = 64 + 128 + 256 = 448); 512: the bank channel alone (transport + conversion); 1024: Gaussian births dropped; 2048 (with 512): no transport, one-layer conversion only
              CARRIER of the second-chaos matrix on the card's own state (physicist/ncarry.py): the complete one-layer law, used through the K31 plane
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
import f4b as F4B
import so2 as SO2
import ntor as NTOR
import ncarry as NCARRY
import pbase as PBASE
PBK = ('pbm', 'pbc', 'pbt', 'pbf', 'pbbsk', 'pbgsk', 'pbgk4', 'pbgbk', 'pbx1', 'pbx2')
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2]); tag = sys.argv[3]; ARMS = sys.argv[4:]
# rank >= 7,000,000 = a generated fresh network of the analyst's production (O/analyst/fresh/loader.py: weights and 64M-sample truth by seed)
if rank >= 7000000:
    sys.path.insert(0, str(HERE / 'analyst' / 'fresh')); import loader as FL
    WEIGHTS = FL.weights(rank); tp = FL.truth(rank)['mc_post'].astype(np.float64)
else:
    WEIGHTS = rank; tp = pc.truth(rank)
NTO_NAMES = ('', 'stale1T', 'stale1', 'stale2', 'free', 'freeT', 'x1', 'x2', 'x3', 'law1', 'lawF', 'lawFk', 'lawF0', 'cardA', 'cardB', 'cardC')      # nto=<k>: substitutes of the oracle matrix built by t19i_recursion.py
t0 = time.time(); ORACLE = np.load(HERE / f't19_oracle_rank{rank:04d}.npz') if (HERE / f't19_oracle_rank{rank:04d}.npz').exists() else None


def run(spec):
    name, _, opts = spec.partition(':'); o = dict(c=0.0, age=99, final=1, pair=1, f4=0.0, on=1, bsrc=0.0, k22=0.0, lmin=1, lmax=15, covk4=1, meank4=1, bs=1.0, f4b=0.0, f4banchor=0, f4bage=2, f4blo=1, f4bhi=14, f4bterms=1234, f4bbl=-1, f4bc1=0.0, f4bc2=0.0, f4bc3=0.0, f4bc4=0.0, f4bt2=0.0, f4bt3=0.0, f4bt4=0.0, f4bmix=0, f4be2=1.0, f4bi2=0.0, f4bown=0, f4bbm=0, f4bcoll=0.0, f4bcollage=99,
             e2=0.0, so2m=0.0, so2r=-1.0, so2c=0.0, so2t=0.0, so2k=1, so2lo=1, so2hi=15, so2f=0.0, so2p=123456, k4all=0, kn=-9.0, kiso=-9.0, k31=-9.0, bsk=0.0, gsk=0.0, gk4=0.0, ds=1.0, e34=0.0, bta=0.0,
             nt4=0.0, nt4m=0.0, nt4ml=0, k4mo=0, agem=0, cm=-1.0, mk4chk=0, mk4d=0, mk4lo=1, mk4hi=15, k4dump=0, edges=1.0, rg=0.0, rgm=1.0, rgs=1.0, rgt=0.0, rgq=0.0, rglo=3, rghi=14, rgw=0, nt31=0.0, ntf=0.0, ntr=0, nto=0, ntx=0, nt22=0.0, nc=0.0, ncr=0, nck=1, ncskip=0, nclo=1, nchi=14, ncv=0, nc31=0, ncage=0, ncsr=0, ncl0=0, ncpart=0, ncforms=4, ncos=0, ncosk=0, ncown=0.0, ncbx=0, ncbas=0, ncritz=0, ncck=0, ncpas=0, nc6=-1.0, nc6ar=1, nc6f=4, nc6g=0, nc6lev=0, nc6df=0, nc6ds=1.0, nc6k22=0.0, nc6pm=0, nc6pk=4, nc6k4=0.0, nc6k4l=0.0, nc6k4mo=0, tk21=0.0, tk31=0.0, tk22=0.0, tkrows=0, tkdraw=0, tkm=0, tkdump=0, tkl4=0, tkb=0.0, tkbl=0.0, tkbt=0, tks21=0.0, tks31=0.0, tkl0=2, tkl1=14, tkf=0, tkcol=0, tkm31=-1, tkm21=-1, tkk=0, bk3a=0.0, bk3b=0.0, bk3lo=1, bk3hi=14, bk22=0.0, bk4=0.0, bk3mu=0, gbk=0.0, bk3s=0, bkbb=0.0, pbm=0.0, pbc=0.0, pbt=0.0, pbf=0.0, pbbsk=0.0, pbgsk=0.0, pbgk4=0.0, pbgbk=0.0, pbx1=0.0, pbx2=0.0, gbku=15, gbklo=0, gbkhi=14, gbkk=0, k3s=1.0, k3sl=0.0, k3s15=1.0, k3l=0.0, k3la=2, k3lam=1, k3lf=0, k3lc=1, k3lr=0, k3lrow=0, k3ldump=0, budget=0, ncdumpR=0, ncb=0.0, ncbconv=1, ncbt=1.0, ncbcol=0, ncbtop=0, nfo=0.0, nfolev=0, nforows=1, nfoforms=1, nfocheck=0, nfod=0, nfods=1.0, nfovb=0, nfodr=0, nfot=1.0, nfosrc=1.0, nfok22=0.0, nfok4=0.0, f4bth1=-1.0, f4bth2=-1.0, X1=-9.0, bscale=-9.0)
    for kv in filter(None, opts.split(',')):
        k, v = kv.split('='); o[k] = float(v)
    dial_over = {k: o[k] for k in ('X1', 'bscale') if o[k] > -9}                     # physics dials set to a VALUE (candidate-2 dials: X1 0.59158, bscale 1.03414)
    cd = pc.build(WEIGHTS, learned=bool(o['on']), dials=dial_over or None); st = dict(births={}, tmp={}, calls=[], stat={}, hub={})
    if o['edges'] != 1.0: pc.wrap(cd, 'edge', lambda l, v, ctx: v * o['edges'])      # edges=<x>: the LEARNED covariance source of layers 8-14 (message edge) times x; with on=0 and edges=0 no learned part is left
    cd.mk4dev = []; cd.mk4 = {}      # MEAN-ONLY KURTOSIS (options nt4m, k4mo, agem, nc6k4mo): a per-unit change dk of the fourth cumulant that only the per-unit mean and the per-unit post variance see.
    # The pair map runs with the card's own k4 (its K31 row term, its per-unit covariance term e4 and the knob hook are untouched); its outputs get the first-order kurtosis terms of dk:
    # mean += sigma (dk / v^2) He2(alpha) pdf / 24, E relu^2 += dk (-alpha pdf / v) / 12, variance = E relu^2 - mean^2. The sources register dk in cd.mk4[layer] from their 'pair_in' hooks.
    if o['nt4m'] or o['k4mo'] or o['agem'] > o['age'] or o['nc6k4mo']:
        pair_mk0_ = cd.env['_nonlinear_pair_batch']
        def pair_mk4(mu, C, K21, k4, k22_override=None, metric='identity', **kw):
            m, covm, post21, radial, aux = pair_mk0_(mu, C, K21, k4, k22_override, metric, **kw); dk = cd.mk4.get(cd.call_layer)
            if dk is None or not torch.is_tensor(dk) or o['mk4d'] or not o['mk4lo'] <= cd.call_layer <= o['mk4hi']: return m, covm, post21, radial, aux      # mk4lo..mk4hi: layers where the mean-only kurtosis acts
            s_, a_, ph_ = (aux[k_].to(torch.float64) for k_ in ('sigma', 'alpha', 'pdf')); v_ = s_ * s_; dm_ = s_ * (dk / v_ ** 2) * (a_ * a_ - 1) * ph_ / 24; mn_ = m + dm_
            var_ = torch.diagonal(covm) + dk * (-a_ * ph_ / v_) / 12 - (mn_ * mn_ - m * m); aux = dict(aux); aux['post_v'] = var_
            if o['mk4chk']:      # certificate: the pair map itself, run with k4 + dk, must give the same per-unit mean and post variance
                m2_, c2_ = pair_mk0_(mu, C, K21, k4 + dk, k22_override, metric, **kw)[:2]
                cd.mk4dev.append((float((m2_ - mn_).abs().max() / dm_.abs().max()), float((torch.diagonal(c2_) - var_).abs().max() / (var_ - torch.diagonal(covm)).abs().max())))
            return mn_, torch.diagonal_scatter(covm, var_), post21, radial, aux
        cd.env['_nonlinear_pair_batch'] = pair_mk4
        if o['mk4d']:      # mk4d=1: the same first-order kurtosis term of the mean, but added through the CORRECTION SLOT (after the learned correction), so the card recentres the covariance
            # for it as it does for the learned correction (second moments held, strength 1 + rc); the per-unit post variance then follows from the recentring, not from the kurtosis formula
            def mk4_delta(l, d, ctx):
                dk = cd.mk4.get(l)
                if l < 1 or l > 14 or dk is None or not torch.is_tensor(dk) or not o['mk4lo'] <= l <= o['mk4hi']: return d
                aux = ctx['aux']; s_, a_, ph_ = (aux[k_].to(torch.float64) for k_ in ('sigma', 'alpha', 'pdf')); return d + (s_ * (dk / (s_ * s_) ** 2) * (a_ * a_ - 1) * ph_ / 24).to(d.dtype)
            pc.wrap(cd, 'delta', mk4_delta)
    if o['bsk'] or o['gsk'] or o['gk4'] or o['bta'] or o['gbk']:
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
        if o['gsk'] or o['gk4'] or o['gbk']:      # gbk: + gbk d bpre, the feed of the cross form Sym(b (x) kappa3) of the fourth cumulant into the transported third cumulant (theory 1.5)
            edit("    g = aux['gate']\n    d = aux['pdf'] / aux['sigma']\n",          # the dedented source: method body at four spaces
                 "    g = aux['gate']\n    d = aux['pdf'] / aux['sigma']\n"
                 f"    gq = g + {o['gsk']!r} * {T3} * ({AL} ** 2 - 1) * aux['pdf'] / 6 - {o['gk4']!r} * (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 3 - 3 * {AL}) * aux['pdf'] / 24 + {o['gbk']!r} * d * bpre * (1.0 if {int(o['gbklo'])} <= layer <= {int(o['gbkhi'])} else 0.0) * ((aux['alpha'].abs() < 1.5).to(d.dtype) if {int(o['gbkk'])} else 1.0)\n")
            U_ = int(o['gbku'])      # gbku: which uses of the gate take the corrected one: 1 the b vector, 2 the inherited post plane, 4 the young columns, 8 the old-tier basis (default all)
            if U_ & 1: edit("b = g * bpre + radial * d * vv / 6", "b = gq * bpre + radial * d * vv / 6")
            if U_ & 2: edit("inherited = g[:, None] ** 2 * g[None, :] * cp", "inherited = gq[:, None] ** 2 * gq[None, :] * cp")
            if U_ & 4: edit("p = t.cat((alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * pp),", "p = t.cat((alive.to(t.float32)[:, None] * (gq.to(t.float32)[:, None] * pp),")
            if U_ & 8: edit("Q = alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * U)", "Q = alive.to(t.float32)[:, None] * (gq.to(t.float32)[:, None] * U)")
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
        r = push0(t, layer, w, p, lam, previous, Q, F, core, stops, rank=rank); st['pp'] = r[0]; st['stops'] = (0,) + tuple(int(x) for x in stops); st['lam'] = lam; return r
    cd.shared['push'] = push
    bd0 = cd.env['_bd_update']
    def bd(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
        st['tmp'] = dict(plus=t.detach(), minus=s.detach()); return bd0(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients)
    cd.env['_bd_update'] = bd
    sel0 = cd.select
    def select(score):
        r = sel0(score); st['calls'].append(r[0].detach()); return r
    cd.select = select
    def local(l, vv, mu, amin=1, amax=None, raw=False):      # hub terms of the pairs of ages amin..amax (default 1..o['age']), isotropic part projected out
        amax = o['age'] if amax is None else amax
        pp = st['pp'].to(torch.float64); stops = st['stops']; loc = torch.zeros(1024, dtype=torch.float64)
        if o['k4all'] and (l - 1) in st['hub'] and amin <= 1:
            # age 1 from ALL hubs e of layer l-1: Wt_e(i) = W[e, i], Y_e(i) = sum_{a != e} C_post[e, a] W[a, i] / gate_e (the card forms C_post W for the covariance push)
            H = st['hub'][l - 1]; w64 = cd.ws[l]; M = (H['C'].to(torch.float32) @ cd.w[l]).to(torch.float64)
            Wt = (w64 * H['alive'][:, None]).T; Y = ((M - w64 * H['cdiag'][:, None]) * (H['alive'] / H['g'].clamp_min(1e-3))[:, None]).T; S1, S2, S3, S4 = H['S']
            loc = loc + (4 * S1[None, :] * Y ** 3 * Wt + 6 * S2[None, :] * Y ** 2 * Wt ** 2 + 4 * S3[None, :] * Y * Wt ** 3 + S4[None, :] * Wt ** 4).sum(1)
        for j in range(l):
            if l - j > amax or l - j < amin: continue
            if o['k4all'] and j == l - 1 and (l - 1) in st['hub']: continue
            B = st['births'][j]; k0 = stops[j]; K = (stops[j + 1] - k0) // 2; assert K == len(B['tp']), (l, j, K, len(B['tp']))
            gap = (B['tp'] - B['tm'])[None, :]; pplus = pp[:, k0:k0 + K]; pminus = pp[:, k0 + K:k0 + 2 * K]
            Wt = (pplus - pminus) / gap; Y = (B['tp'][None, :] * pminus - B['tm'][None, :] * pplus) / gap; S1, S2, S3, S4 = B['S']
            loc = loc + (4 * S1[None, :] * Y ** 3 * Wt + 6 * S2[None, :] * Y ** 2 * Wt ** 2 + 4 * S3[None, :] * Y * Wt ** 3 + S4[None, :] * Wt ** 4).sum(1)
        if raw: return loc      # the hub terms themselves, isotropic part not taken out (k4dump)
        al = mu / vv.clamp_min(1e-12).sqrt(); w4 = ((al * al - 1) * torch.exp(-al * al / 2)) ** 2 + 1e-12; iso = vv * vv
        beta = (w4 * loc * iso).sum() / (w4 * iso * iso).sum(); dk = loc - beta * iso
        if amin == 1: st['stat'][l] = (float(beta), float((dk * dk * w4).sum().sqrt() / (w4 * iso * iso).sum().sqrt()))
        return dk
    def k3parts(l):      # the bank's per-unit third cumulant sum_r lam_r pp_ir^3 by cohort age band, and the bank's weighted square sum_r lam_r pp_ir^2 (pp_r . mu-free: plain) by band
        pp = st['pp'].to(torch.float64); lam = st['lam'].to(torch.float64); stops = st['stops']; out = {}
        for nm_, a0_, a1_ in (('a12', 1, 2), ('a34', 3, 4), ('a56', 5, 6), ('a7', 7, 99)):
            acc = torch.zeros(1024, dtype=torch.float64); acc2 = torch.zeros(1024, dtype=torch.float64)
            for j in range(l):
                if not a0_ <= l - j <= a1_ or j + 1 >= len(stops): continue
                P_ = pp[:, stops[j]:stops[j + 1]]; L_ = lam[stops[j]:stops[j + 1]]; acc = acc + (P_ ** 3) @ L_; acc2 = acc2 + (P_ ** 2) @ L_.abs()
            out[f'k3{nm_}_{l}'] = acc.numpy().copy(); out[f'm2{nm_}_{l}'] = acc2.numpy().copy()
        return out
    def pin_hook(l, pin, ctx):
        if o['k4dump'] and l >= 1:      # k4dump=1: the young pairs' local quartic per unit by age band (raw hub terms), with the layer's mean, variance and isotropic kurtosis: k4d_rank<rank>_<tag>_<arm>.npz
            vv_ = torch.diagonal(ctx['Cpre']); K4D.setdefault(name, {}).update({f'mu_{l}': pin[0].detach().numpy().copy(), f'v_{l}': vv_.detach().numpy().copy(), f'k4iso_{l}': (ctx['k4'] * torch.ones(1024, dtype=torch.float64)).detach().numpy().copy(), f'resp_{l}': ctx['response_mask'].detach().numpy().copy()})
            for nm_, a0_, a1_ in (('a12', 1, 2), ('a34', 3, 4), ('a56', 5, 6), ('a7', 7, 99)): K4D[name][f'{nm_}_{l}'] = local(l, vv_, pin[0], a0_, a1_, raw=True).detach().numpy().copy()
            if o['k4dump'] >= 2: K4D[name].update(k3parts(l)); K4D[name].update({f'k3star_{l}': torch.diagonal(ctx['sp']).detach().numpy().copy(), f'k3b_{l}': (3 * vv_ * ctx['bpre']).detach().numpy().copy(), f'k3all_{l}': torch.diagonal(pin[2]).detach().numpy().copy(), f'bpre_{l}': ctx['bpre'].detach().numpy().copy()})
        if l >= 1 and o['c'] != 0.0 and o['pair'] and o['lmin'] <= l <= o['lmax']:
            dk = o['c'] * local(l, torch.diagonal(ctx['Cpre']), pin[0]); st['dk'] = dk
            k22 = pin[4]
            if o['k22']:
                ratio = ((pin[3] + dk) / pin[3].clamp_min(1e-30)).clamp_min(0.0).sqrt(); k22 = k22 * ratio[:, None] * ratio[None, :]
            # the card computes the pair map twice at layers 2-14: the first call (pin) gives the mean, the plane target and the radial;
            # the covariance-knob hook (gn3) recomputes the covariance output from ctx['k4'].
            if o['k4mo']:      # k4mo=1: the whole local term for the MEAN ONLY (per-unit uses of the pair map), the covariance keeps the isotropic kurtosis
                cd.mk4[l] = cd.mk4.get(l, 0) + dk
            else:
                if o['covk4']: ctx['k4'] = ctx['k4'] + dk
                if o['meank4']: pin = (pin[0], pin[1], pin[2], pin[3] + dk, k22, pin[5])
        if l >= 1 and o['agem'] > o['age'] and o['pair'] and o['lmin'] <= l <= o['lmax']:      # agem=<A>: the pairs of ages age+1..A too, for the MEAN ONLY, at strength cm (default c)
            dkm = (o['c'] if o['cm'] < 0 else o['cm']) * local(l, torch.diagonal(ctx['Cpre']), pin[0], int(o['age']) + 1, o['agem'])
            cd.mk4[l] = cd.mk4.get(l, 0) + dkm
        return pin
    pc.wrap(cd, 'pair_in', pin_hook)
    def b_hook(l, b, ctx):
        if o['bsrc'] and l >= 1 and 'dk' in st: b = b + o['bsrc'] * ctx['d'] * st['dk'] / (6 * ctx['vv'].clamp_min(1e-12))
        if o['bs'] != 1.0: b = b + (o['bscale'] if o['bscale'] > -9 else pc.DIALS['bscale']) * (o['bs'] - 1) * ctx['radial'] * ctx['d'] * ctx['vv'] / 6     # scale of the isotropic Sym(C x b) source (the physicist's bsrc)
        return b
    pc.wrap(cd, 'b_post', b_hook)
    def k415(l, k4, ctx):
        if o['k4dump']:      # the final layer's parts (no pair map there)
            K4D.setdefault(name, {}).update({'mu_15': ctx['mu'].detach().numpy().copy(), 'v_15': ctx['vv'].detach().numpy().copy(), 'k4iso_15': (k4 * torch.ones(1024, dtype=torch.float64)).detach().numpy().copy()})
            for nm_, a0_, a1_ in (('a12', 1, 2), ('a34', 3, 4), ('a56', 5, 6), ('a7', 7, 99)): K4D[name][f'{nm_}_15'] = local(15, ctx['vv'], ctx['mu'], a0_, a1_, raw=True).detach().numpy().copy()
            if o['k4dump'] >= 2: K4D[name].update(k3parts(15)); K4D[name]['k3all_15'] = ctx['k3'].detach().numpy().copy()
        if o['c'] != 0.0 and o['final'] and o['lmax'] >= 15: k4 = k4 + o['c'] * local(15, ctx['vv'], ctx['mu'])
        if o['agem'] > o['age'] and o['final'] and o['lmax'] >= 15: k4 = k4 + (o['c'] if o['cm'] < 0 else o['cm']) * local(15, ctx['vv'], ctx['mu'], int(o['age']) + 1, o['agem'])
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
    if o['nt4m']:      # ORACLE, MEAN ONLY: the second-chaos kurtosis of the TRUE matrix, k4_i += nt4m (6 / n) R_ii, for the per-unit uses of the pair map (mean, post variance, post k3 / k4)
        # and the final layer's mean: registered in cd.mk4 (see MEAN-ONLY KURTOSIS below): the covariance sees the card's kurtosis, the mean the corrected one.
        # nt4ml: 0 layers 1-15, 1 layers 1-14 only, 2 the final layer only.
        ORAm_ = ORACLE if not int(o['nto']) else np.load(HERE / f"t19_oracle_rank{rank:04d}_{NTO_NAMES[int(o['nto'])]}.npz")
        RD_ = {l_: torch.from_numpy(np.diag(ORAm_[f'R_L{l_}']).astype(np.float64).copy()) for l_ in range(1, 16) if f'R_L{l_}' in ORAm_.files}
        def nt4m_pin(l, pin, ctx):
            if 1 <= l <= 14 and l in RD_ and int(o['nt4ml']) != 2: cd.mk4[l] = cd.mk4.get(l, 0) + o['nt4m'] * (6.0 / 1024) * RD_[l]
            return pin
        pc.wrap(cd, 'pair_in', nt4m_pin)
        if int(o['nt4ml']) != 1 and 15 in RD_ and o['mk4hi'] >= 15: pc.wrap(cd, 'k4_15', lambda l, k4, ctx: k4 + o['nt4m'] * (6.0 / 1024) * RD_[15])
    if o['nt4'] or o['nt31'] or o['ntf'] or o['nt22']:
        ORA = ORACLE if not int(o['nto']) else np.load(HERE / f"t19_oracle_rank{rank:04d}_{NTO_NAMES[int(o['nto'])]}.npz")
        NTOR.install(cd, torch, {l: ORA[f'R_L{l}'] for l in range(1, 16)}, k4=o['nt4'], k31=o['nt31'], feed=o['ntf'], rank=int(o['ntr']), replace=bool(o['ntx']), k22=o['nt22'])
    if o['nc'] or o['ncpas']:
        NCARRY.install(cd, torch, strength=o['nc'], rank=int(o['ncr']), k21=bool(o['nck']), start=int(o['nclo']), stop=int(o['nchi']),
                       skip=tuple(n_ for i_, n_ in enumerate(('gain', 'rows', 'vtilt', 'own31', 'k3c', 'k3off', 'k3prod', 'k3el', 'k3x', 'convonly', 'nobirth', 'notrans')) if int(o['ncskip']) >> i_ & 1),
                       conv=''.join(ch for i_, ch in enumerate('bsm') if int(o['ncv']) >> i_ & 1), k31v=bool(o['nc31']),
                       conv_age=int(o['ncage']), state_rank=int(o['ncsr']), law_start=int(o['ncl0']), partner=int(o['ncpart']), forms=int(o['ncforms']), oneside=(int(o['ncos']), int(o['ncosk'])), ownrow=float(o['ncown']), bank_exact=bool(o['ncbx']),
                       basis={0: 'TC', 1: 'TGRC', 2: 'TGR', 3: 'TGC', 4: 'GRC', 5: 'T'}[int(o['ncbas'])], ritz=bool(o['ncritz']), law_ck=int(o['ncck']), passive=bool(o['ncpas']))
    if o['nc6'] >= 0:      # AUDIT of the lead's pair-carried law (physicist/ncar6_audit.py = a copy of O/ncar6.py that dumps its matrices): strength nc6 (0 = passive), level r31,
        import ncar6_audit as N6      # with the side table the lead's harness builds (pushed columns, their weights, the births t+, t-, selected hubs of every layer)
        st6 = dict(births={}, tmp={}, calls=[]); push0_ = cd.shared['push']
        def push6(t_, layer, w, p, lam, previous, Q, F, core, stops, rank=384, push0_=push0_, st6=st6):
            r = push0_(t_, layer, w, p, lam, previous, Q, F, core, stops, rank=rank); st6['pp'] = r[0]; st6['lam_in'] = lam; st6['old'] = int(r[4]); return r
        cd.shared['push'] = push6
        bd0_ = cd.env['_bd_update']
        def bd6(bp, Cpre, K21, aux, d, alive, births, a, t_, s_, coefficients, bd0_=bd0_, st6=st6):
            st6['tmp'] = dict(plus=t_.detach(), minus=s_.detach()); st6['bp'] = bp.detach(); r = bd0_(bp, Cpre, K21, aux, d, alive, births, a, t_, s_, coefficients); st6['stored'] = r[0].detach(); return r
        cd.env['_bd_update'] = bd6
        sel0_ = cd.select
        def select6(score, sel0_=sel0_, st6=st6):
            r = sel0_(score); st6['calls'].append(r[0].detach()); return r
        cd.select = select6
        def out6(l, out, ctx, st6=st6):
            if l <= 14:
                sel = st6['calls'][-1]; f = lambda x: x.detach().to(torch.float64)[sel]
                st6['births'][l] = dict(tp=f(st6['tmp']['plus']), tm=f(st6['tmp']['minus']), sel=sel)
                a_, b_ = st6['stored'].to(torch.float64)[:, sel], st6['bp'].to(torch.float64)[:, sel]      # the newborn direction s_h against the law's c_h = g o Co[:, h]
                st6.setdefault('scos', {})[l] = (float(((a_ * b_).sum(0) / (a_.norm(dim=0) * b_.norm(dim=0)).clamp_min(1e-300)).mean()), float(((a_ - b_).norm(dim=0) / b_.norm(dim=0).clamp_min(1e-300)).mean()), int(sel.numel()))
            return out
        pc.wrap(cd, 'state_out', out6)
        AUD6[name] = dict(dump={}, st=st6)
        N6.install(cd, torch, st6, k31=float(o['nc6']), forms=int(o['nc6f']), level='read' if o['nc6lev'] else 'r31', dump=AUD6[name]['dump'], ca=0.75, arest=bool(o['nc6ar']), gain=int(o['nc6g']), dfit=int(o['nc6df']), dscale=float(o['nc6ds']), k22=float(o['nc6k22']), pmproj=int(o['nc6pm']), pk=int(o['nc6pk']), k4=float(o['nc6k4']), k4last=float(o['nc6k4l']), k4mo=int(o['nc6k4mo']))
    if o['ncb'] or o['nfo']:      # the bank channel as a module with the lead's carrier interface (physicist/ncar_bank.py, ncar_fold.py), fed through a side table as the lead's and the trainer's harnesses do
        import ncar_bank as NCB
        import ncar_fold as NFO
        stb = {}; push0_ = cd.shared['push']
        def pushb(t_, layer, w, p, lam, previous, Q, F, core, stops, rank=384, push0_=push0_, stb=stb):
            r = push0_(t_, layer, w, p, lam, previous, Q, F, core, stops, rank=rank); stb['pp'] = r[0]; stb['lam_in'] = lam; stb['old'] = int(r[4]); return r
        cd.shared['push'] = pushb
        if o['ncb']: NCB.install(cd, torch, stb, k31=o['ncb'], forms=int(o['ncforms']), conv=int(o['ncbconv']), transport=o['ncbt'], rank=int(o['ncr']), columns=bool(o['ncbcol']), top=int(o['ncbtop']))
        if o['nfo']:
            NFCHECK[name] = {} if o['nfocheck'] else None
            NFO.install(cd, torch, stb, strength=o['nfo'], forms=int(o['nfoforms']), level=('pair', 'read', 'r31')[int(o['nfolev'])], rows=('all', 'resp')[int(o['nforows'])], top=int(o['ncbtop']), check=NFCHECK[name], dfit=int(o['nfod']), dscale=float(o['nfods']),
                        vbeta=bool(o['nfovb']), dresp=bool(o['nfodr']), transport=float(o['nfot']), source=float(o['nfosrc']), k22=float(o['nfok22']), k4=float(o['nfok4']))
    if o['k3s'] != 1.0 or o['k3sl'] or o['k3s15'] != 1.0:      # a knob on the per-unit third cumulant as the mean and the post variance use it: the diagonal of the K21 plane x (k3s + k3sl * layer)
        def k3s_pin(l, pin, ctx):      # at layers 1-14 (pair map input), and x k3s15 x that at layer 15 (hook k3_15); the plane off the diagonal is not touched
            k21_ = pin[2]
            if l < 1 or not torch.is_tensor(k21_) or k21_.ndim != 2: return pin
            f_ = o['k3s'] + o['k3sl'] * l; k21n_ = k21_ + torch.diag(torch.diagonal(k21_) * (f_ - 1.0)); return (pin[0], pin[1], k21n_) + tuple(pin[3:])
        pc.wrap(cd, 'pair_in', k3s_pin); pc.wrap(cd, 'k3_15', lambda l, k3, ctx: k3 * (o['k3s'] + o['k3sl'] * 15) * o['k3s15'])
    if any(o[k_] for k_ in PBK):      # THE PHYSICS BASE AS ONE MODULE (physicist/pbase.py): pbm pbc pbt pbf = its so2m so2c so2t so2f; pbbsk pbgsk pbgk4 pbgbk; pbx1 pbx2 = the cross form (specific planes)
        PBASE.install(cd, torch, None, **{n_: o[k_] for k_, n_ in zip(PBK, ('so2m', 'so2c', 'so2t', 'so2f', 'bsk', 'gsk', 'gk4', 'gbk', 'x1', 'x2'))})
    if o['bk3a'] or o['bk3b'] or o['bk22'] or o['bk4'] or o['bkbb']:      # THE CROSS FORM OF THE COLLECTIVE MODE AND THE THIRD CUMULANT (physicist T33): kappa4 contains x Sym(b (x) kappa3), b = the card's
        # pushed isotropic third-order vector bpre (bk3mu=1: mu / n instead). (3,1) plane: bk3a * k3_i b_j + bk3b * b_i K21_ij through remainder31 (theory for a common multiplicative
        # fluctuation 1.5 and 4.5; the truth planes of networks 35 / 100 read 2.2 and 4.5); (2,2) plane: bk22 * (b_i K21_ji + b_j K21_ij) (theory 3); per unit: k4_i += bk4 * b_i k3_i (theory 6).
        def bk3_pin(l, pin, ctx):
            if not int(o['bk3lo']) <= l <= int(o['bk3hi']) or not torch.is_tensor(pin[2]) or pin[2].ndim != 2: return pin
            mu_, C_, k21_, k4_, k22_, rem_ = pin; f64_ = torch.float64; N_ = 1024; eye_ = torch.eye(N_, dtype=torch.bool); v_ = torch.diagonal(C_).to(f64_); s_ = v_.sqrt()
            b_ = (mu_.to(f64_) / N_) if o['bk3mu'] else ctx['bpre'].to(f64_); K_ = k21_.to(f64_); k3_ = torch.diagonal(K_)
            if o['bk3s']:      # bk3s=1: only the SPECIFIC third cumulant (bank + star planes, without the b-form v_i b_j + 2 C_ij b_i; k3 without 3 v_i b_i): the truth planes want no term of
                K_ = (ctx['cp'] + ctx['sp']).to(f64_); k3_ = torch.diagonal(K_)      # second order in b (T33c: coefficients 3.2 / 10 on the specific forms, about 0 on v_i b_i b_j)
            if o['bk3a'] or o['bk3b'] or o['bkbb']:
                P_ = o['bk3a'] * torch.outer(k3_, b_) + o['bk3b'] * b_[:, None] * K_ + o['bkbb'] * torch.outer(v_ * b_, b_); add_ = (P_ / (v_[:, None] * s_[:, None] * s_[None, :])).masked_fill(eye_, 0.); rem_ = add_ if rem_ is None else rem_ + add_
            if o['bk22'] and torch.is_tensor(k22_) and k22_.ndim == 2:
                X_ = b_[:, None] * K_.T + b_[None, :] * K_; k22_ = k22_ + o['bk22'] * X_.masked_fill(eye_, 0.).to(k22_.dtype)
            if o['bk4']:
                dk_ = o['bk4'] * b_ * k3_; k4_ = k4_ + dk_.to(k4_.dtype) if torch.is_tensor(k4_) else k4_ + dk_
            return (mu_, C_, k21_, k4_, k22_, rem_)
        pc.wrap(cd, 'pair_in', bk3_pin)
    if o['k3l'] or o['k3ldump']:      # the lead's K31L (O/k31l.py through my audit copy): the young pairs' local quartic as a (3,1) plane; k3lc=1 fourth-chaos part only; k3lr=1 folded into the read;
        import k31l_audit as K3L      # k3la / k3lam = oldest / youngest cohort age; k3lrow=1 row part taken out, 2 row part only; k3ldump=1 keeps the raw planes (k3l_rank<rank>_<tag>_<arm>.npz)
        K3LD[name] = {} if o['k3ldump'] else None
        K3L.install(cd, torch, st, k31=float(o['k3l']), age=int(o['k3la']), lo=1, hi=14, forms=int(o['k3lf']), chaos4=bool(o['k3lc']), level='read' if o['k3lr'] else 'r31', dump=K3LD[name], agemin=int(o['k3lam']), rowrm=int(o['k3lrow']))
    if o['tk21'] or o['tk31'] or o['tk22'] or o['tkdump'] or o['tkb'] or o['tkbl'] or o['tks21'] or o['tks31']:      # ORACLE: the Monte Carlo truth of a whole pair-map plane (8M samples, networks 35 and 100, layers 2-14) in place of the card's,
        # off the diagonal (the diagonals are the per-unit inputs: the budget arms); tkrows=1: kink rows only. ONLY THE COVARIANCE OUTPUT of the pair map is changed (the
        # difference of two extra pair-map calls is added to the covariance): the post K21 target, the star and the births keep the card's own planes.
        # tkdraw: 0 the first draw, 1 the second draw (network 35 only), 2 the mean of both (16M samples). tkm (K21 and K31 only): 0 the whole plane from the truth; 1 only the
        # part proportional, row by row, to the covariance (K21: a_i C_ij, K31: a_i rho_ij) from the truth, the rest of the plane the card's own (with its default knobs);
        # 2 the reverse: the card's row coefficients, the truth's rest. TKD[name][layer] = the comparison of the truth's and the card's two parts.
        # tkb=<x>, tkbl=<y>: NO truth plane: the card's own b-form v_i bpre_j + 2 C_ij bpre_i is added to the K21 plane at strength x + y * layer, in the covariance output only (tkbt=1: only the
        # part v_i bpre_j); works on any network. tks21=<x> / tks31=<x>: NO truth plane: the card's own K21 plane (with its knobs) / own K31 slot on the rows chosen by tkrows is multiplied by x
        # in the covariance output only (a dial test: which row classes want more or less of the card's own plane).
        NEEDT = bool(o['tk21'] or o['tk31'] or o['tk22'])
        # tkf: 0 the stored 8M planes; 1 / 2 / 3 this lane's 64M planes of layers 2-7 (t24c_merge.py: all blocks / seed 1 / seed 2), only those layers are then substituted
        PLN = (np.load(HERE / ('t24_planes_rank%04d_%s.npz' % (rank, {1: 'all', 2: 'A', 3: 'B'}[int(o['tkf'])]))) if o['tkf'] else (pc.planes(rank, '_noise2') if o['tkdraw'] == 1 else pc.planes(rank))) if NEEDT else None; PLN2 = pc.planes(rank, '_noise2') if o['tkdraw'] == 2 else None; TPL = {}; TKD[name] = {}
        V3_ = dict(pc.LC.gl.V3); V3_.update({k: o[n_] for k, n_ in (('k21n', 'kn'), ('iso', 'kiso'), ('k31', 'k31')) if o[n_] > -9})      # the knob constants this arm runs with: the truth replaces
        # the card's plane AS ITS KNOBS MAKE IT (the knob hook stays on and its increment is taken out of the substituted plane), so no knob has to be switched off in an oracle arm
        def tpl_pin(l, pin, ctx):
            TPL['pin'] = pin; return pin
        pc.wrap(cd, 'pair_in', tpl_pin)
        def tpl_cov(l, C, ctx):
            if not 2 <= l <= 14 or (NEEDT and f'K21_{l}' not in PLN.files) or (o['tkl4'] and l not in (2, 5, 9, 12)) or not int(o['tkl0']) <= l <= int(o['tkl1']): return C      # tkl4=1: only the four layers the second draw has; tkl0..tkl1: layer range of the substitution
            mu, Cpre, k21_, k4v, k22_, rem = TPL['pin']; N_ = 1024; eye_ = torch.eye(N_, dtype=torch.bool); v_ = torch.diagonal(Cpre); s_ = v_.sqrt(); pair = ctx['pair']
            al_ = mu / s_; rs_ = {0: torch.ones(N_, dtype=torch.bool), 1: al_.abs() < 1.5, 2: al_.abs() >= 1.5, 3: al_ >= 1.5, 4: al_ <= -1.5, 5: ctx['response_mask'].bool(), 6: ~ctx['response_mask'].bool()}[int(o['tkrows'])]
            rowsel = rs_[:, None].expand(N_, N_); sel = rowsel & ~eye_      # tkrows: 0 all rows, 1 kink rows |mu| < 1.5 sigma, 2 the others, 3 mu >= 1.5 sigma, 4 mu <= -1.5 sigma, 5 the card's response rows, 6 the others
            T64_ = lambda k: torch.from_numpy(PLN[f'{k}_{l}'].astype(np.float64)) if PLN2 is None else (torch.from_numpy(PLN[f'{k}_{l}'].astype(np.float64)) + torch.from_numpy(PLN2[f'{k}_{l}'].astype(np.float64))) / 2
            k21t, k22t, remt = k21_, k22_, rem; f64_ = torch.float64; tkd = TKD[name].setdefault(l, {})
            def rowp(X, Bm):      # the part of X proportional, row by row, to the off-diagonal of Bm, and the row coefficients
                Bo = Bm.masked_fill(eye_, 0.); a_ = (X.masked_fill(eye_, 0.) * Bo).sum(1) / (Bo * Bo).sum(1); return a_[:, None] * Bo, a_
            def cmp_(key, Tt, Cc, Bm):      # truth against the card: row coefficients (correlation, ratio of the means, rms ratio) and the rest (cosine, norm ratio), energy shares of the truth
                Pt, at = rowp(Tt, Bm); Pc, ac = rowp(Cc, Bm); Rt = (Tt - Pt).masked_fill(eye_, 0.); Rc = (Cc - Pc).masked_fill(eye_, 0.); cc = lambda x, y: float((x * y).sum() / (x.norm() * y.norm() + 1e-300))
                tkd[key] = dict(row_corr=round(float(torch.corrcoef(torch.stack((at, ac)))[0, 1]), 4), row_mean_card_over_truth=round(float(ac.mean() / at.mean()), 4), row_rms_card_over_truth=round(float(ac.norm() / at.norm()), 4),
                                rest_cos=round(cc(Rt, Rc), 4), rest_norm_card_over_truth=round(float(Rc.norm() / Rt.norm()), 4), truth_row_share=round(float(Pt.norm() ** 2 / (Pt.norm() ** 2 + Rt.norm() ** 2)), 4),
                                whole_cos=round(cc(Tt.masked_fill(eye_, 0.), Cc.masked_fill(eye_, 0.)), 4))
                return Pt, Pc
            if o['tk21']:
                Tt = T64_('K21'); kc = k21_.to(f64_); kink_ = ((mu / s_).abs() < 1.5).to(f64_)
                kx_ = ctx['k21'].to(f64_); cx_ = ctx['Cpre'].to(f64_); kinkx_ = (ctx['aux']['alpha'].abs() < 1.5).to(f64_)      # the knob hook works on the card's own plane (before any pair_in hook)
                kce = kc + ((V3_['iso'] * kinkx_) * torch.diagonal(kx_) / torch.diagonal(cx_))[:, None] * cx_ + (V3_['k21n'] * (1 - kinkx_))[:, None] * kx_      # the card's plane as its knobs make it
                Pt, Pc = cmp_('K21', Tt, kce, Cpre.to(f64_))
                m21_ = o['tkm21'] if o['tkm21'] >= 0 else o['tkm']      # tkm21 / tkm31: tkm for one plane only (so that one arm can take the K21 column part and the K31 row part)
                if m21_: Tt = (kce - Pc + Pt) if m21_ == 1 else (Pc + Tt - Pt)
                D_ = (Tt - (kc if o['tkk'] else kce)).masked_fill(eye_, 0.)      # tkk=1: the knobs' own increments STAY on top of the truth (the difference is taken against the card's plane without its knobs)
                if o['tkcol']:      # tkcol=1: only the part of (truth - card) with the COLUMN VECTOR mu (the scale mode): row by row its projection on mu_j in the covariance's column weights
                    # (gate_j / sigma_j)^2, q_i mu_j; tkcol=2: the rest (the difference without that part). TKD: the part's share of the weighted difference.
                    mu64_ = mu.to(f64_); wc2_ = (ctx['aux']['gate'].to(f64_) / s_.to(f64_)) ** 2; cw_ = (wc2_ * mu64_)[None, :].expand(N_, N_).masked_fill(eye_, 0.)
                    q_ = (D_ * cw_).sum(1) / (cw_ * mu64_[None, :]).sum(1); P_ = (q_[:, None] * mu64_[None, :]).masked_fill(eye_, 0.)
                    wr_ = (ctx['aux']['pdf'].to(f64_) / (2 * v_.to(f64_)))[:, None] * wc2_.sqrt()[None, :]; tkd['col'] = dict(share=round(float(((P_ * wr_) ** 2).sum() / ((D_ * wr_) ** 2).sum()), 4))
                    D_ = P_ if o['tkcol'] == 1 else D_ - P_
                k21t = torch.where(sel, kc + o['tk21'] * D_, kc)
            if o['tks21']:
                kc = k21_.to(f64_); kx_ = ctx['k21'].to(f64_); cx_ = ctx['Cpre'].to(f64_); kinkx_ = (ctx['aux']['alpha'].abs() < 1.5).to(f64_)
                kce = kc + ((V3_['iso'] * kinkx_) * torch.diagonal(kx_) / torch.diagonal(cx_))[:, None] * cx_ + (V3_['k21n'] * (1 - kinkx_))[:, None] * kx_
                k21t = torch.where(sel, kc + (o['tks21'] - 1.0) * kce, kc)
            if o['tks31']:
                rho = (Cpre / torch.outer(s_, s_)).masked_fill(eye_, 0.); rowf = cd.env.get('_ROWF', [None])[0]; k4c = k4v if (torch.is_tensor(k4v) and k4v.ndim) else k4v * torch.ones(N_, dtype=torch.float64)
                internal = ((k4c / v_ ** 2) * (rowf(mu, torch.diagonal(k21_), v_) if rowf is not None else 1.0))[:, None] * rho; rem0 = torch.zeros((N_, N_), dtype=torch.float64) if rem is None else rem
                vx_ = torch.diagonal(ctx['Cpre']).to(f64_); rhox_ = (ctx['Cpre'] / torch.outer(vx_.sqrt(), vx_.sqrt())).masked_fill(eye_, 0.)
                knob31 = (V3_['k31'] * (ctx['k4'].expand(vx_.shape) / vx_ ** 2) * (rowf(ctx['mu'], torch.diagonal(ctx['k21']), vx_) if rowf is not None else 1.0))[:, None] * rhox_
                remt = torch.where(sel, rem0 + (o['tks31'] - 1.0) * (internal + knob31 + rem0), rem0)
            if o['tkb'] or o['tkbl']:
                bp_ = ctx['bpre'].to(f64_); xb_ = o['tkb'] + o['tkbl'] * l; vv_ = torch.diagonal(ctx['Cpre']).to(f64_)
                k21t = k21t.to(f64_) + xb_ * (vv_[:, None] * bp_[None, :] + (0.0 if o['tkbt'] else 2.0) * ctx['Cpre'].to(f64_) * bp_[:, None]).masked_fill(eye_, 0.)      # off the diagonal only
            if o['tk22']: k22t = torch.where(sel, o['tk22'] * T64_('K22') + (1 - o['tk22']) * k22_.to(torch.float64), k22_.to(torch.float64))
            if o['tk31']:      # the whole K31 slot = truth: remainder31 = T31_true - (what the pair map builds itself from k4 and the row factor); any carried plane is replaced on these rows
                rho = (Cpre / torch.outer(s_, s_)).masked_fill(eye_, 0.); rowf = cd.env.get('_ROWF', [None])[0]; k4c = k4v if (torch.is_tensor(k4v) and k4v.ndim) else k4v * torch.ones(N_, dtype=torch.float64)
                internal = ((k4c / v_ ** 2) * (rowf(mu, torch.diagonal(k21_), v_) if rowf is not None else 1.0))[:, None] * rho
                T31t = (T64_('K31') / (v_[:, None] * s_[:, None] * s_[None, :])).masked_fill(eye_, 0.); rem0 = torch.zeros((N_, N_), dtype=torch.float64) if rem is None else rem
                vx_ = torch.diagonal(ctx['Cpre']).to(f64_); rhox_ = (ctx['Cpre'] / torch.outer(vx_.sqrt(), vx_.sqrt())).masked_fill(eye_, 0.)      # the knob hook's own increment: from the card's k4 and plane before any pair_in hook
                knob31 = (V3_['k31'] * (ctx['k4'].expand(vx_.shape) / vx_ ** 2) * (rowf(ctx['mu'], torch.diagonal(ctx['k21']), vx_) if rowf is not None else 1.0))[:, None] * rhox_
                tot = internal + knob31 + rem0; Pt, Pc = cmp_('K31', T31t, tot, rho.to(f64_))      # the card's whole slot as its knob makes it
                m31_ = o['tkm31'] if o['tkm31'] >= 0 else o['tkm']
                if m31_: T31t = (tot - Pc + Pt) if m31_ == 1 else (Pc + T31t - Pt)
                remt = torch.where(sel, rem0 + o['tk31'] * (T31t - tot), rem0)
            if o['tkdump']:      # the card's own planes of this layer, for the offline comparison with the truth planes (t22_planes.py)
                kc_ = k21_.to(f64_); kink_ = (al_.abs() < 1.5).to(f64_); rho_ = (Cpre / torch.outer(s_, s_)).masked_fill(eye_, 0.); rowf_ = cd.env.get('_ROWF', [None])[0]
                k4c_ = k4v if (torch.is_tensor(k4v) and k4v.ndim) else k4v * torch.ones(N_, dtype=f64_); int_ = ((k4c_ / v_ ** 2) * (rowf_(mu, torch.diagonal(k21_), v_) if rowf_ is not None else 1.0))[:, None] * rho_
                f32_ = lambda x: x.detach().to(torch.float32).numpy()
                TKDUMP.setdefault(name, {}).update({f'mu_{l}': mu.detach().numpy(), f'v_{l}': v_.detach().numpy(), f'K21c_{l}': f32_(kc_), f'C_{l}': f32_(Cpre), f'T31int_{l}': f32_(int_), f'k4_{l}': k4c_.detach().numpy(),
                                                    f'T31rem_{l}': f32_(rem if rem is not None else torch.zeros((N_, N_))), f'K22c_{l}': f32_(k22_ if (torch.is_tensor(k22_) and k22_.ndim == 2) else torch.zeros((N_, N_))),
                                                    f'cp_{l}': f32_(ctx['cp']), f'sp_{l}': f32_(ctx['sp']), f'bpre_{l}': ctx['bpre'].detach().numpy(), f'resp_{l}': ctx['response_mask'].detach().numpy(), f'alive_{l}': ctx['alive'].detach().numpy()})
            call = lambda a_, b_, c_: pair(mu, Cpre, a_, k4v, b_, 'covariance')[1] if c_ is None else pair(mu, Cpre, a_, k4v, b_, 'covariance', remainder31=c_)[1]
            return C + (call(k21t, k22t, remt) - call(k21_, k22_, rem))
        pc.wrap(cd, 'cov_pm', tpl_cov)
    if o['rg']:      # ROW GAIN OF THE BANK READ, the law the true K21 planes of networks 35 / 100 show (physicist T55b): the read's plane on the response rows, off the diagonal, times
        # (1 + r_i),  r_i = rg x { rgm [0.004 + 0.0022 (l - 3)] - rgs [0.068 + 0.033 (l - 3)] (log sigma_i - <log sigma>) + rgt (t4_i - <t4>) + rgq (qb_i - <qb>) },  layers rglo..rghi;
        # < > = mean over the response rows weighted by the row's weight in the plane error (pdf_i / (2 v_i))^2 sum_j (cp_ij gate_j / sigma_j)^2; t4 = the card's k4 / v^2 as the pair map
        # gets it; qb = the bank plane's own column along mu per unit variance. Truth fits: rgt about +2 (joint) to +4.5 (alone), rgq about -10 (joint) to -20 (alone).
        # rgw=0: in the COVARIANCE OUTPUT only (difference of two pair-map calls, like the truth-plane oracle); rgw=1: the plane itself for every consumer (pair map input).
        RGI[name] = {}
        def rg_gain(l, pin, ctx):
            mu_, C_, k21_, k4_ = pin[0], pin[1], pin[2], pin[3]; N_ = 1024; f64_ = torch.float64; eye_ = torch.eye(N_, dtype=torch.bool); v_ = torch.diagonal(C_).to(f64_); s_ = v_.sqrt(); al_ = mu_.to(f64_) / s_
            ph_ = torch.exp(-al_ * al_ / 2) / math.sqrt(2 * math.pi); gt_ = torch.erfc(-al_ / math.sqrt(2)) / 2; wc_ = gt_ / s_; cp_ = ctx['cp'].to(f64_).masked_fill(eye_, 0.); resp_ = ctx['response_mask'].bool()
            w_ = ((ph_ / (2 * v_)) ** 2 * ((cp_ * wc_[None, :]) ** 2).sum(1)) * resp_.to(f64_); ctr_ = lambda x: x - (w_ * x).sum() / w_.sum().clamp_min(1e-300)
            c_ = wc_ * wc_ * mu_.to(f64_); qb_ = ((cp_ * c_[None, :]).sum(1) / ((c_ * mu_).sum() - c_ * mu_)) / v_; t4_ = (k4_ * torch.ones(N_, dtype=f64_)) / v_ ** 2
            r_ = o['rg'] * (o['rgm'] * (0.004 + 0.0022 * (l - 3)) - o['rgs'] * (0.068 + 0.033 * (l - 3)) * ctr_(s_.log()) + o['rgt'] * ctr_(t4_) + o['rgq'] * ctr_(qb_)) * resp_.to(f64_)
            RGI[name][l] = (round(float((w_ * r_).sum() / w_.sum()), 5), round(float(((w_ * r_ * r_).sum() / w_.sum()).sqrt()), 5)); return (r_[:, None] * cp_)
        def rg_pin(l, pin, ctx):
            RGS['pin'] = pin
            if not int(o['rglo']) <= l <= int(o['rghi']) or not int(o['rgw']) or not torch.is_tensor(pin[2]) or pin[2].ndim != 2: return pin
            return (pin[0], pin[1], pin[2] + rg_gain(l, pin, ctx).to(pin[2].dtype)) + tuple(pin[3:])
        RGS = {}; pc.wrap(cd, 'pair_in', rg_pin)
        def rg_cov(l, C, ctx):
            if not int(o['rglo']) <= l <= int(o['rghi']) or int(o['rgw']) or l < 2: return C
            mu, Cpre, k21_, k4v, k22_, rem = RGS['pin']; pair = ctx['pair']; k21t = k21_.to(torch.float64) + rg_gain(l, RGS['pin'], ctx)
            call = lambda a_: pair(mu, Cpre, a_, k4v, k22_, 'covariance')[1] if rem is None else pair(mu, Cpre, a_, k4v, k22_, 'covariance', remainder31=rem)[1]
            return C + (call(k21t) - call(k21_))
        pc.wrap(cd, 'cov_pm', rg_cov)
    if o['ncdumpR']:      # keep the pre-activation covariance and mean of every layer (for the offline study of dumped carrier matrices)
        CAP = {}
        def cap_hook(l, pin, ctx, CAP=CAP):
            CAP[l] = (pin[1].detach().to(torch.float32).clone(), pin[0].detach().to(torch.float64).clone()); return pin
        pc.wrap(cd, 'pair_in', cap_hook)
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
    if o['budget']:      # T8 budget on THIS card: an input of the per-unit mean formula (variance, skew, kurtosis) replaced by 32M Monte Carlo truth at every layer 1-15
        TSA, TSB = pc.pu(rank, 'P32M'), pc.pu(rank, 'indep32M'); f6 = lambda z, k: torch.from_numpy(np.ascontiguousarray(z[f'mc_pre_{k}'].astype(np.float64)))
        TS = {'A': {k: f6(TSA, k) for k in ('v', 'k3', 'k4')}, 'B': {k: f6(TSB, k) for k in ('v', 'k3', 'k4')}}; TS['AB'] = {k: (TS['A'][k] + TS['B'][k]) / 2 for k in ('v', 'k3', 'k4')}
        bctl = dict(arm='', tset=None); bst = {}; NS_ = cd.env['nonlinear_scalar']
        pc.wrap(cd, 'k4_15', lambda layer, value, ctx: bst.__setitem__('k4', value) or value); pc.wrap(cd, 'k3_15', lambda layer, value, ctx: bst.__setitem__('k3', value) or value)
        def bpin(l, pin, ctx):      # what the pair map of the layer really receives (after every other hook)
            bst['pin'] = (torch.diagonal(pin[1]).detach(), torch.diagonal(pin[2]).detach(), pin[3].detach() if torch.is_tensor(pin[3]) else pin[3]); return pin
        pc.wrap(cd, 'pair_in', bpin)
        def bdelta(layer, d, ctx):
            if layer == 0 or not bctl['arm']: return d
            host, mu_ = ctx['host'], ctx['mu']
            if layer == 15: v_, k3_, k4_ = ctx['vv'], bst['k3'], bst['k4']
            else: v_, k3_, k4_ = bst['pin']
            k4_ = k4_ if (torch.is_tensor(k4_) and k4_.ndim) else (k4_ * torch.ones(1024, dtype=torch.float64))
            bst.setdefault('dev', []).append(float((NS_(mu_, v_, k3_, k4_)[0] - host).abs().max()))
            tt = TS[bctl['tset']]; tv, tk3, tk4 = tt['v'][layer], tt['k3'][layer], tt['k4'][layer]; arm_ = bctl['arm']
            v2 = tv if 'v' in arm_ else v_; g1 = tk3 / tv ** 1.5 if 's' in arm_ else k3_ / v_ ** 1.5; g2 = tk4 / tv ** 2 if 'k' in arm_ else k4_ / v_ ** 2
            return d + (NS_(mu_, v2, g1 * v2 ** 1.5, g2 * v2 ** 2)[0] - host)
        pc.wrap(cd, 'delta', bdelta)
    pred = pc.run(cd); mse = pc.mse(pred, tp); PRED[name] = pred; INFO[name] = getattr(cd, 'so2_info', {}); NTI[name] = getattr(cd, 'ntor_info', {})
    if o['budget']:
        e0 = pred[15] - tp[15]; E0 = float((e0 * e0).sum()); bo = {}; pred2 = pc.run(cd); bo['rerun_max_abs_diff'] = float(np.abs(pred2 - pred).max()); PB = {}
        for arm_, ts_ in (('v', 'A'), ('v', 'B'), ('s', 'AB'), ('k', 'AB'), ('sk', 'AB'), ('vsk', 'A'), ('vsk', 'B')):
            bctl.update(arm=arm_, tset=ts_); PB[(arm_, ts_)] = pc.run(cd)[15] - tp[15]
        bctl.update(arm='', tset=None)
        bo['variance_exact_pct'] = round(100 * (float((PB[('v', 'A')] * PB[('v', 'B')]).sum()) / E0 - 1), 2); bo['skew_exact_pct'] = round(100 * (float((PB[('s', 'AB')] ** 2).sum()) / E0 - 1), 2)
        bo['kurtosis_exact_pct'] = round(100 * (float((PB[('k', 'AB')] ** 2).sum()) / E0 - 1), 2); bo['all_three_exact_pct'] = round(100 * (float((PB[('vsk', 'A')] * PB[('vsk', 'B')]).sum()) / E0 - 1), 2)
        bo['skew_and_kurtosis_exact_pct'] = round(100 * (float((PB[('sk', 'AB')] ** 2).sum()) / E0 - 1), 2); bo['own_final_x1e9'] = round(E0 / 1024 * 1e9, 4)
        bo['formula_check_max_dev'] = max(bst.get('dev', [0.0])); AUDOUT[name] = dict(AUDOUT.get(name) or {}, budget=bo)
        for k_, v_ in PB.items(): PRED[name + '_exact_' + k_[0] + '_' + k_[1]] = np.tile(v_ + tp[15], (16, 1))
    if o['ncdumpR'] and hasattr(cd, 'ncarry_R'):      # the carrier's network-specific matrix and the card's covariance and mean at layers 2..14 -> ncR_rank<rank>_<tag>_<arm>.npz
        np.savez(HERE / f'ncR_rank{rank:04d}_{tag}_{name}.npz', **{f'R_L{l_}': cd.ncarry_R[l_].to(torch.float32).numpy() for l_ in cd.ncarry_R if 2 <= l_ <= 14},
                 **{f'C_L{l_}': CAP[l_][0].numpy() for l_ in CAP if 1 <= l_ <= 14}, **{f'mu_L{l_}': CAP[l_][1].numpy() for l_ in CAP if 1 <= l_ <= 14})
    if NFCHECK.get(name): NTI[name] = dict(NTI[name], cols={'fold_check': {str(k): {kk: round(vv, 4) for kk, vv in v.items()} for k, v in NFCHECK[name].items()}})
    if RGI.get(name): AUDOUT[name] = dict(AUDOUT.get(name) or {}, row_gain_mean_rms={str(k_): v_ for k_, v_ in RGI.pop(name).items()})
    if cd.mk4dev: AUDOUT[name] = dict(AUDOUT.get(name) or {}, mk4_check=dict(mean_max_rel_dev=max(x[0] for x in cd.mk4dev), variance_max_rel_dev=max(x[1] for x in cd.mk4dev), calls=len(cd.mk4dev)))
    if name in AUD6:      # the lead's matrix against the full-matrix carrier (if carried on this card) and against the Monte Carlo oracle, layer by layer, off the diagonal
        aud = {}; Rf_ = getattr(cd, 'ncarry_R', {}); off = ~torch.eye(1024, dtype=torch.bool)
        for l_, d_ in AUD6[name]['dump'].items():
            if o['k4dump']: K4D.setdefault(name, {}).update({f'diagB_{l_}': torch.diagonal(d_['B']).numpy().copy(), f'diagR_{l_}': torch.diagonal(d_['R']).numpy().copy()})
            R6 = d_['R'].to(torch.float64); row = dict(d_['coef']); row['newborn_direction'] = AUD6[name]['st'].get('scos', {}).get(l_ - 1)
            if l_ in Rf_:
                Rf = Rf_[l_]; row['cos_with_full_matrix'] = float((R6[off] * Rf[off]).sum() / (R6[off].norm() * Rf[off].norm())); row['size_over_full_matrix'] = float(R6[off].norm() / Rf[off].norm())
                row['full_matrix_multiple_of_C'] = cd.ncarry_a.get(l_)
            if ORACLE is not None:
                Ro = torch.from_numpy(ORACLE[f'R_L{l_}'].astype(np.float64)); eo = float((Ro[off] ** 2).sum())
                row['share_of_oracle'] = (2 * float((R6[off] * Ro[off]).sum()) - float((R6[off] ** 2).sum())) / eo; row['cos_with_oracle'] = float((R6[off] * Ro[off]).sum() / (R6[off].norm() * Ro[off].norm()))
                for nm_, d6_ in (('diag', torch.diagonal(R6)), ('diagB', torch.diagonal(d_['B']).to(torch.float64))):      # the matrix's diagonal (forms removed / raw) against the oracle's R_ii, both minus their means
                    dc_ = d6_ - d6_.mean(); oc_ = torch.diagonal(Ro) - torch.diagonal(Ro).mean(); row[nm_ + '_corr_with_oracle'] = float((dc_ * oc_).sum() / (dc_.norm() * oc_.norm())); row[nm_ + '_oracle_on_card_coef'] = float((dc_ * oc_).sum() / (dc_ * dc_).sum())
                if l_ in Rf_: row['full_matrix_share_of_oracle'] = (2 * float((Rf_[l_][off] * Ro[off]).sum()) - float((Rf_[l_][off] ** 2).sum())) / eo
            aud[str(l_)] = {k_: ([round(float(x), 5) for x in v_] if isinstance(v_, (tuple, list)) else (None if v_ is None else round(float(v_), 5))) for k_, v_ in row.items()}
        AUDOUT[name] = dict(AUDOUT.get(name) or {}, **aud); AUD6[name] = None
    if K4D.get(name): np.savez(HERE / f'k4d_rank{rank:04d}_{tag}_{name}.npz', **K4D.pop(name))
    if K3LD.get(name): np.savez(HERE / f'k3l_rank{rank:04d}_{tag}_{name}.npz', **{f'P_{l_}': v_.numpy() for l_, v_ in K3LD.pop(name).items()})
    if TKDUMP.get(name): np.savez(HERE / f'tkd_rank{rank:04d}_{tag}_{name}.npz', **TKDUMP.pop(name))
    if TKD.get(name): AUDOUT[name] = dict(AUDOUT.get(name) or {}, truth_planes={str(k): v for k, v in TKD[name].items()})
    if hasattr(cd, 'ncar_info') and cd.ncar_info.get('zcheck'): AUDOUT[name] = dict(AUDOUT.get(name) or {}, zcheck={str(k): {kk: [float('%.5g' % x) for x in vv] for kk, vv in v.items()} for k, v in cd.ncar_info['zcheck'].items()})
    if hasattr(cd, 'ncarry_R') and ORACLE is not None:      # the carrier's matrix against the Monte Carlo oracle of the same network, layer by layer (off-diagonal)
        fid = {}
        for l_, R_ in cd.ncarry_R.items():
            T_ = torch.as_tensor(np.asarray(ORACLE[f'R_L{l_}'], dtype=np.float64)); T_ = T_ - torch.diag(torch.diagonal(T_)); fid[l_] = round(float((2 * (R_ * T_).sum() - (R_ * R_).sum()) / (T_ * T_).sum()), 3)
        NTI[name] = {'fidelity': fid}; NTI[name].update(cd.ncarry_info)
    return name, o, mse, st['stat']


base = None; PRED = {}; INFO = {}; NTI = {}; NFCHECK = {}; AUD6 = {}; AUDOUT = {}; TKD = {}; TKDUMP = {}; K3LD = {}; K4D = {}; RGI = {}
for spec in ARMS:
    name, o, mse, stat = run(spec)
    if base is None: base = mse
    print(json.dumps(dict(rank=rank, tag=tag, arm=name, opts=o, final_x1e9=round(float(mse[15]) * 1e9, 4), final_vs_first_arm_pct=round(100 * (float(mse[15]) / float(base[15]) - 1), 3),
                          all_layer_vs_first_arm_pct=round(100 * (float(mse.mean()) / float(base.mean()) - 1), 3),
                          per_layer_vs_first_arm_pct=[round(100 * (float(mse[i]) / float(base[i]) - 1), 2) for i in range(16)], per_layer_x1e9=[round(float(x) * 1e9, 2) for x in mse],
                          beta={k: round(v[0], 4) for k, v in stat.items() if k in (5, 9, 13, 15)}, dk_rel={k: round(v[1], 3) for k, v in stat.items() if k in (5, 9, 13, 15)},
                          so2={k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in INFO[name].items() if k in (2, 5, 9, 13, 14, 15)},
                          ntor={str(k): ({kk: round(vv, 5) for kk, vv in v.items()} if k not in ('fidelity', 'cols') else v) for k, v in NTI[name].items() if k in (2, 5, 9, 13, 14, 'fidelity', 'cols')},
                          audit=AUDOUT.get(name), wall=round(time.time() - t0, 1))), flush=True)
np.savez(HERE / f'k4l_pred_rank{rank:04d}_{tag}.npz', truth=tp, **PRED)
