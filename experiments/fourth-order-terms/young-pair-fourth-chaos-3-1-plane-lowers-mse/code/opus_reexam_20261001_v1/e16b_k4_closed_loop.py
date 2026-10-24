#!/usr/bin/env python3
"""TAG opus-reexam E16b: closed-loop screen of a NON-ISOTROPIC per-unit fourth cumulant built from the exact hub coefficients
(current torch card, learned parts ON, no refit; final-layer MSE against the organizers' truth).
The card's per-unit fourth cumulant is radial * v^2 (one number per layer). Every column pair of the bank gives, elementwise,
   k4_local(i) = sum over pairs  4 S1 Y_i^3 Wt_i + 6 S2 Y_i^2 Wt_i^2 + 4 S3 Y_i Wt_i^3 + S4 Wt_i^4       (e15_local_feed.py, e16_local_k4.py)
and the arm replaces  k4 -> k4 + c * (k4_local - beta v^2),  beta = the projection of k4_local on v^2 weighted by the size of the
kurtosis term of the mean formula ((alpha^2 - 1) pdf)^2, so the trained isotropic level stays where it is. Applied at the pair map
of layers 1-14 (also ctx['k4']) and at the final layer's mean.
arms: name:c=<float>,age=<max column age, 99 = all>[,final=0/1][,pair=0/1][,f4=<theta of the physicist's F4, installed first>][,on=0 learned mean correction off]
      [,bsrc=<x>: the Sym(C x b) source radial d v/6 gets + x d dk/(6 v)] [,k22=1: the isotropic (2,2) plane scaled by sqrt(ratio_i ratio_j), ratio = new k4 / old k4]
      [,d_<dial>=<delta>: shift one of the 12 physics dials (X1 X0 X2 X3 bscale s_carry s_new c0 c1 k4a k4b k4c) by delta]
      [,e2=<x>: the missing second-order skew term of the mean formula, x k3^2 He4(alpha) pdf / (72 sigma^5), layers 1-15]
      [,a3=<x>,a3min=<l>: pairs of ages above `age` and up to 4 enter with relative weight x at layers >= l]
      [,cv=<x>,tau=<y>: E30, the fourth cumulant made from the third at earlier layers, carried as one weight per bank column (retention y x theory)]
      [,c2k=<b>: E33, per-unit k4 += (6 b / n) (C^2)_ii at layers 1-14, isotropic part projected out (Monte Carlo b = 0.23-0.29)]
      [,nc=<x>,nck=<K>,ncm1=0/1,nct=,ncs=,nca=,ncb=,ncconv=,nclo=,nchi=,ncdump=1: NCAR (O/ncar.py), the vector carrier of the norm coupling as an extra K31 plane]
      [,br=<x>: birth coefficient d (1 + x radial / 3) instead of d (1 + radial / 3)]
      [,edge=<x>: learned covariance edge increment (layers 8-14) times x]
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
    name, _, opts = spec.partition(':'); o = dict(c=0.0, age=99, final=1, pair=1, f4=0.0, on=1, bsrc=0.0, k22=0.0, lmin=1, lmax=15, covk4=1, meank4=1, bs=1.0, ds=1.0, rcx=1.0, edge=1.0, e2=0.0, br=1.0, a3=0.0, a3min=1, cv=0.0, tau=1.0, dump=0, c2k=0.0, nc=-1.0, nck=8, ncm1=0, nct=0.9, ncs=1.0, nca=0.5, ncb=0.25, ncconv=1.0, nclo=2, nchi=14, ncdump=0, nclaw=2, ncq1=0.0, ncsw=0, nck21=1, ncplane=0, ncskip=0, ncy=0.0, ncyp=0, ncyforms=1, ncyrank=0, ora=0.0, orak=16, orapart=0, orarank=0, ncr=32, ncp=24, nclean=1, nccl=1, ncstart=0, ncpass=2, nchu=0, nctop=0, ncrow=1.0, ncysr=0, ncysk=0, ncyst=1.0, ncpr=0, ncx=0, ncbd=1, ncbi=1.0, ncrw=1.0, ncbk=0.0, ncbf=4, ncbtop=0, ncbconv=1, ncdg=1.0, ncar=1, nclev=0, ncdf=0, ncds=1.0, nccb=0.0, nccr=0.0, ncpt=0, ncpd=0, ncpm=0, ncc2=0, ncc2s=0, k3l=0.0, k3la=2, k3lf=4, k3lc=0, k3lr=0, k3lm=0, k3ln=1, nck4=0.0, nck4f=0.0, k3c=0.0, k3cc=1, k3cf=0, k3ca=1, k3dg=0, k3m=0.0, k3m1=1.0, k3m2=1.0, k3m3=1.0, k3mf=0, kg=0.0, kgs=0.0, kgl=1, rf=0.0, mxp=0.0, mxpl=14, mxpn=0, ncrk=1.0, ncrn=1.0, sc=0.0, scm=0, sclo=0, schi=14, scf=0.0, nck3=0.0, nck3o=0.0, nckq=1.0, nckc=1, od=0.0, odlo=0, odhi=14, odfull=0, odhalf=0, rg=0.0, rga=0.068, rgb=0.033, rglo=3, rghi=14, rgm=0.0)
    dial_over = {}
    for kv in filter(None, opts.split(',')):
        k, v = kv.split('=')
        if k.startswith('d_'): dial_over[k[2:]] = float(pc.DIALS[k[2:]]) + float(v)      # d_<dial>=<delta>: one of the 12 physics dials shifted by delta
        else: o[k] = float(v)
    cd = pc.build(WEIGHTS, learned=bool(o['on']), dials=dial_over or None); st = dict(births={}, tmp={}, calls=[], stat={}, stat_cv={}, taus={})
    if o['br'] != 1.0:
        # birth coefficient d (1 + br * radial / 3): the radial / 3 part is the exchange pairing 2 C_ic C_jc of the isotropic fourth-order feed
        # (the physicist's fit of the true feed gives that pairing a weight near 0). The step's source is re-executed with that one line changed.
        import inspect, textwrap, types
        owner = next(k for k in type(cd).__mro__ if 'step' in k.__dict__)
        src = textwrap.dedent(inspect.getsource(owner.step)); line = 'birth = d * (1 + radial / 3)'
        assert src.count(line) == 1
        g = dict(sys.modules[owner.__module__].__dict__); exec(src.replace(line, 'birth = d * (1 + %r * radial / 3)' % float(o['br'])), g)
        cd.step = types.MethodType(g['step'], cd)
    if o['f4']: F4.install(cd, torch, float(o['f4']), age=2, anchor='cheap')
    push0 = cd.shared['push']
    def push(t, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
        r = push0(t, layer, w, p, lam, previous, Q, F, core, stops, rank=rank)
        stp = (0,) + tuple(int(x) for x in stops)
        if o['mxp'] and 2 <= layer <= int(o['mxpl']) and (layer - 1) in st['births'] and 'g' in st['births'][layer - 1] and 'mx_mu' in st:
            # MX1 AS A PERMANENT COLUMN SHIFT (lead 17:45Z): the mixed term's fold P_ri -> P_ri - (mu_i / v_i) sum_a W_ai d_a Y_ai pp_ra is LINEAR in the column, so it is
            # the same push with the weights W_ai (1 - mxp (mu_i / v_i) (d_a / g_a) Y_ai) on the hub rows a: no product beyond the push itself. The newborn cohort is not shifted.
            B_ = st['births'][layer - 1]; sel_ = B_['sel']; k0_ = stp[layer - 1]; K_ = (stp[layer] - k0_) // 2
            if K_ and K_ == int(sel_.numel()):
                gap_ = (B_['tp'] - B_['tm'])[None, :]; p64 = p.to(t.float64)
                Sb = (B_['tp'][None, :] * p64[:, k0_ + K_:k0_ + 2 * K_] - B_['tm'][None, :] * p64[:, k0_:k0_ + K_]) / gap_          # the hubs' base vectors, post space
                Y_ = w.to(t.float64).T @ Sb                                                                                         # pushed base: Y_ai, (next units) x hubs
                fac = 1 - o['mxp'] * (st['mx_mu'] / st['mx_v'].clamp_min(1e-12))[None, :] * (B_['d'] / B_['g'].clamp_min(1e-6))[:, None] * Y_.T
                w_eff = w.clone(); w_eff[sel_, :] = (w.to(t.float64)[sel_, :] * fac).to(w.dtype)
                r2 = push0(t, layer, w_eff, p, lam, previous, Q, F, core, stops, rank=rank)
                keep = k0_ if not o['mxpn'] else r[0].shape[1]                                                                      # mxpn=1: the newborn cohort is shifted too
                r = (t.cat((r2[0][:, :keep], r[0][:, keep:]), dim=1), r2[1]) + tuple(r2[2:])
        st['pp'] = r[0]; st['stops'] = stp; st['lam_in'] = lam; st['old'] = int(r[4]); return r
    cd.shared['push'] = push
    if o['rg'] or o['rgm']:
        # RG (lead 20:03Z, after the physicist's 20:00Z open-loop law): a ROW GAIN on the bank read's plane, installed UNDER the modules (their fold terms are not scaled):
        # plane_ij <- plane_ij (1 + r_i), r_i = -rg (rga + rgb (l - 3)) (log sigma_i - its mean over the response rows) - rgm (m_i - mean), m_i = the plane's row sum against
        # mu over v_i, standardised over the rows; layers rglo..rghi. The gain is centred, so the plane's mean level (which the dials hold) does not move.
        read_rg0 = cd.shared['read']; old_c_rg = cd.hooks.get('cpre'); old_mu_rg = cd.hooks.get('mu')
        def mu_rg(l, mu, ctx):
            if old_mu_rg is not None: mu = old_mu_rg(l, mu, ctx)
            st['rg_mu'] = mu.detach().to(torch.float64); return mu
        def c_rg(l, C, ctx):
            if old_c_rg is not None: C = old_c_rg(l, C, ctx)
            st['rg_l'] = l; st['rg_v'] = torch.diagonal(C).detach().to(torch.float64).clamp_min(1e-12); return C
        cd.hooks['mu'], cd.hooks['cpre'] = mu_rg, c_rg
        def read_rg(tt, pp, lam, response_indices, indices, U, F, old_columns):
            plane = read_rg0(tt, pp, lam, response_indices, indices, U, F, old_columns); l_ = st.get('rg_l')
            if l_ is None or not (o['rglo'] <= l_ <= o['rghi']) or response_indices.numel() < 3: return plane
            v_ = st['rg_v'][response_indices]; ls_ = 0.5 * torch.log(v_); r_ = -o['rg'] * (o['rga'] + o['rgb'] * (l_ - 3)) * (ls_ - ls_.mean())
            if o['rgm']:
                m_ = (plane.to(torch.float64) @ st['rg_mu'][indices]) / v_; m_ = (m_ - m_.mean()) / m_.std().clamp_min(1e-30); r_ = r_ - o['rgm'] * m_
            st.setdefault('rg_stat', {})[l_] = (float(r_.std()), float(r_.abs().max()))
            return plane * (1 + r_).to(plane.dtype)[:, None]
        cd.shared['read'] = read_rg
    bd0 = cd.env['_bd_update']
    def bd(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
        st['tmp'] = dict(plus=t.detach(), minus=s.detach()); r = bd0(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients)
        L_ = st.get('sc_layer')
        if o['sc'] and L_ is not None and o['sclo'] <= L_ <= o['schi']:
            # SC (lead 19:27Z): THE NEWBORN COLUMNS FROM THE POST COVARIANCE. The card pushes two n x n matrices with the next layer's weights: the post
            # covariance C (covariance sandwich) and the stored hub columns s_h = g o Cpre[:, h] + learned second-order change. To first order
            # C[:, h] = g_h s_h, so with s'_h = r_h x (C[:, h] without its own entry) the pushed columns are an elementwise function of the product
            # C W the sandwich makes anyway: one full product per layer less (4.15% of the S40 bill). scm 0: r_h = 1 / g_h; scm 1: r_h = the least-squares
            # scalar of the card's own stored column on C's column (free: per-hub sums); sc = mixing weight (1 = replace); scf > 0: the fraction scf
            # of hubs with the largest relative misfit keep the card's column (a partial product).
            f64 = torch.float64; al_ = alive.to(f64)[:, None]; old_ = (bp * alive[:, None]).to(torch.float32).to(f64); card_ = r[0].to(f64)
            Cp_ = st['sc_C'].to(f64); Cp_ = (Cp_ - torch.diag(torch.diagonal(Cp_))) * al_
            if int(o['scm']) == 0: rr_ = 1.0 / aux['gate'].to(f64).clamp_min(1e-3)
            else: rr_ = (card_ * Cp_).sum(0) / (Cp_ * Cp_).sum(0).clamp_min(1e-30)
            S1_ = Cp_ * rr_[None, :]
            mis_ = ((card_ - S1_) ** 2).sum(0) / (card_ ** 2).sum(0).clamp_min(1e-30)
            use_ = torch.ones(card_.shape[1], dtype=f64)
            if o['scf'] > 0:
                kk_ = int(round(float(o['scf']) * card_.shape[1])); bb_ = births.to(f64)
                if kk_ > 0: use_[torch.argsort(-(mis_ * bb_))[:kk_]] = 0.0
            mix_ = float(o['sc']) * use_[None, :]
            new32_ = ((1 - mix_) * card_ + mix_ * S1_).to(torch.float32); new_ = new32_.to(f64); diff_ = new_ - old_
            a64_ = a.to(torch.float32).to(f64); t64_ = (t * alive).to(torch.float32).to(f64); s64_ = (s * alive).to(torch.float32).to(f64)
            wgt_ = (a64_ * (t64_ - s64_)) * births; qw_ = (a64_ * (t64_ * t64_ - s64_ * s64_)) * births
            dl_ = (2 * old_ * diff_ + diff_ * diff_) * wgt_[None, :] + qw_[:, None] * diff_.T
            dl_ = dl_ - torch.diag(torch.diagonal(dl_))
            bsel_ = births.to(f64) > 0
            st.setdefault('sc_stat', {})[L_] = (float(((card_ - S1_)[:, bsel_] ** 2).sum().sqrt() / (card_[:, bsel_] ** 2).sum().sqrt()), float(((card_ - old_)[:, bsel_] ** 2).sum().sqrt() / (card_[:, bsel_] ** 2).sum().sqrt()), float(use_[bsel_].mean()))
            r = (new32_, dl_.to(r[1].dtype))
        return r
    cd.env['_bd_update'] = bd
    if o['od']:               # ORACLE (lead 19:55Z): the post covariance's DIAGONAL at truth (32M-sample all-layer truth of the covhunter, ranks 35 / 100 / 37 / 39), layers odlo..odhi,
        # mixing weight od; odfull = 1: the whole post covariance at truth (the known covariance lever, for scale); odhalf = 1 / 2: the truth's half A / B instead of the whole.
        # The ceiling of a learned per-unit readout for the post variance (batch 4, o1): what the diagonal part of the covariance error is worth.
        MC_ = pc.mcall(rank); st['od_stat'] = {}; sfx_ = {0: '', 1: '_A', 2: '_B'}[int(o['odhalf'])]
        def od_cov(l, C, ctx):
            if not (o['odlo'] <= l <= o['odhi']): return C
            T_ = torch.from_numpy(np.asarray(MC_[f'cov_post{sfx_}_{l}'], dtype=np.float64)); Cd = C.detach().to(torch.float64); dT, dC = torch.diagonal(T_), torch.diagonal(Cd)
            ok = dT > 1e-9; rel = ((dC - dT)[ok] / dT[ok]); st['od_stat'][l] = (float(rel.pow(2).mean().sqrt()), float(rel.mean()), float(((Cd - T_) - torch.diag(dC - dT)).norm() / (T_ - torch.diag(dT)).norm()))
            if o['odfull']: return ((1 - o['od']) * Cd + o['od'] * T_).to(C.dtype)
            return (Cd + o['od'] * torch.diag(dT - dC)).to(C.dtype)
        pc.wrap(cd, 'cov_out', od_cov)
    if o['sc']:
        def sc_cov(l, C, ctx):
            st['sc_layer'] = l; st['sc_C'] = C.detach(); return C
        pc.wrap(cd, 'cov_out', sc_cov)
    sel0 = cd.select
    def select(score):
        r = sel0(score); st['calls'].append(r[0].detach()); return r
    cd.select = select
    def local(l, vv, mu):
        pp = st['pp'].to(torch.float64); stops = st['stops']; loc = torch.zeros(1024, dtype=torch.float64)
        for j in range(l):
            wage = 1.0
            if l - j > o['age']:
                if not (o['a3'] and l - j <= 4 and l >= o['a3min']): continue
                wage = o['a3']                                    # ages beyond `age` up to 4, at layers >= a3min, with relative weight a3
            B = st['births'][j]; k0 = stops[j]; K = (stops[j + 1] - k0) // 2; assert K == len(B['tp']), (l, j, K, len(B['tp']))
            gap = (B['tp'] - B['tm'])[None, :]; pplus = pp[:, k0:k0 + K]; pminus = pp[:, k0 + K:k0 + 2 * K]
            Wt = (pplus - pminus) / gap; Y = (B['tp'][None, :] * pminus - B['tm'][None, :] * pplus) / gap; S1, S2, S3, S4 = B['S']
            loc = loc + wage * (4 * S1[None, :] * Y ** 3 * Wt + 6 * S2[None, :] * Y ** 2 * Wt ** 2 + 4 * S3[None, :] * Y * Wt ** 3 + S4[None, :] * Wt ** 4).sum(1)
        al = mu / vv.clamp_min(1e-12).sqrt(); w4 = ((al * al - 1) * torch.exp(-al * al / 2)) ** 2 + 1e-12; iso = vv * vv
        beta = (w4 * loc * iso).sum() / (w4 * iso * iso).sum(); dk = loc - beta * iso
        st['stat'][l] = (float(beta), float((dk * dk * w4).sum().sqrt() / (w4 * iso * iso).sum().sqrt()))
        return dk
    def conv(l, vv, mu):
        # E30: the fourth cumulant made from the THIRD at earlier layers and carried on the bank's own columns. With R = the centred squared norm of
        # a layer's post-activations, N_ab = sum_c kappa4(x_a, x_b, x_c, x_c) gets at every layer the source g_a g_b sum_c beta_c K3pre[a, b, c],
        # beta = 2 m (1 - g): on the bank that is one weight per column, c_r += lam_r (pp_r . beta), and N is transported with the columns times
        # tau = 2 mean(g - m d). For Gaussian weights the per-unit fourth cumulant of the next layer has the part (12 / n) w_i' N w_i =
        # (12 / n) sum_r c_r pp_ri^2 (the second-chaos part); its isotropic projection is taken out as for the hub terms.
        c = st.get('cacc')
        if c is None or c.numel() == 0: return torch.zeros(1024, dtype=torch.float64)
        pp = st['pp'].to(torch.float64); assert pp.shape[1] == c.numel(), (l, pp.shape, c.numel())
        q = (12.0 / 1024) * ((pp * pp) @ c)
        st.setdefault('qdump', {})[l] = q.clone()
        al = mu / vv.clamp_min(1e-12).sqrt(); w4 = ((al * al - 1) * torch.exp(-al * al / 2)) ** 2 + 1e-12; iso = vv * vv
        beta = (w4 * q * iso).sum() / (w4 * iso * iso).sum(); dq = q - beta * iso
        st['stat_cv'][l] = (float(beta), float((dq * dq * w4).sum().sqrt() / (w4 * iso * iso).sum().sqrt()))
        return dq
    def pin_cv(l, pin, ctx):
        if l >= 1 and o['cv'] and o['lmin'] <= l <= o['lmax']:
            dq = o['cv'] * conv(l, torch.diagonal(ctx['Cpre']), pin[0])
            ctx['k4'] = ctx['k4'] + dq; pin = (pin[0], pin[1], pin[2], pin[3] + dq, pin[4], pin[5])
        return pin
    pc.wrap(cd, 'pair_in', pin_cv)
    def pin_c2(l, pin, ctx):
        # E33: the C^2 part of the norm coupling. Monte Carlo (E32) finds Ntilde = a tr C x C + b C^2 with b = 0.23-0.29 where the isotropic model
        # has b = 2 a (about 0.015): per unit that is k4_i += (6 b / n) (C^2)_ii, a free row energy of the covariance. Layers 1-14 (the final layer
        # has no covariance matrix). The projection on v^2 is taken out as for the hub terms. c2k = the value of b used.
        if l >= 1 and o['c2k'] and o['lmin'] <= l <= o['lmax']:
            Cp = ctx['Cpre']; vv = torch.diagonal(Cp); q = (6.0 * o['c2k'] / 1024) * (Cp * Cp).sum(1)
            al = pin[0] / vv.clamp_min(1e-12).sqrt(); w4 = ((al * al - 1) * torch.exp(-al * al / 2)) ** 2 + 1e-12; iso = vv * vv
            dq = q - ((w4 * q * iso).sum() / (w4 * iso * iso).sum()) * iso
            ctx['k4'] = ctx['k4'] + dq; pin = (pin[0], pin[1], pin[2], pin[3] + dq, pin[4], pin[5])
        return pin
    pc.wrap(cd, 'pair_in', pin_c2)
    def k415_cv(l, k4, ctx):
        if o['cv'] and o['lmax'] >= 15: k4 = k4 + o['cv'] * conv(15, ctx['vv'], ctx['mu'])
        return k4
    pc.wrap(cd, 'k4_15', k415_cv)
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
            st['births'][l] = dict(tp=f(st['tmp']['plus']), tm=f(st['tmp']['minus']), S=S, sel=sel)
            if o['mxp']: st['births'][l].update(d=dj, g=gt)                                           # for the shifted push: the hubs' pdf / sigma and gates
            if o['k3c'] or o['k3dg'] or o['k3m']: st['births'][l].update(d=dj, Ch=ctx['Cpre'].detach().to(torch.float64)[sel][:, sel])      # for k31c.py: the hubs' pdf / sigma and their covariance block
        if o['cv'] and l <= 14:
            aux = ctx['aux']; f64 = lambda x: x.detach().to(torch.float64)
            g, host, dd = f64(aux['gate']), f64(ctx['host']), f64(aux['pdf']) / f64(aux['sigma'])
            pp = st['pp'].to(torch.float64); lam_all = f64(out[3]); nold = pp.shape[1]
            c_old = st.get('cacc'); c_old = torch.zeros(nold, dtype=torch.float64) if c_old is None else c_old
            assert c_old.numel() == nold, (l, c_old.numel(), nold)
            tau = o['tau'] * 2 * float((g - host * dd).mean()); st['taus'][l] = tau
            c_new = tau * c_old + lam_all[:nold] * (pp.T @ (2 * host * (1 - g))) if nold else c_old
            st['cacc'] = torch.cat((c_new, torch.zeros(lam_all.numel() - nold, dtype=torch.float64)))
        st['calls'].clear()
        return out
    pc.wrap(cd, 'state_out', out_hook)
    if o['ora']:              # ORACLE split: the true network-specific matrix (physicist's t19_oracle_rank<r>.npz) as the K31 extra plane, cut by the top-K
        # eigenvectors of the card's own covariance: orapart 0 whole, 1 one-sided block (I-P) R P + P R (I-P), 2 two-sided block P R P, 3 complement (I-P) R (I-P),
        # 4 one-sided block without mode 1, 5 one-sided block of mode 1 only; orarank > 0: the part is cut to its orarank largest |eigenvalue| directions
        ORA = np.load(HERE / f'physicist/t19_oracle_rank{rank:04d}.npz'); old_pin_o = cd.hooks.get('pair_in')
        def pin_ora(l, pin, ctx):
            if old_pin_o is not None: pin = old_pin_o(l, pin, ctx)
            if not (1 <= l <= 14) or f'R_L{l}' not in ORA: return pin
            mu_, Cpre_, k21_, k4_, k22_, rem_ = pin; C_ = Cpre_.to(torch.float64); R_ = torch.from_numpy(ORA[f'R_L{l}'].astype(np.float64)); R_ = (R_ + R_.T) / 2
            part = int(o['orapart'])
            if part:
                lam_, V_ = torch.linalg.eigh(C_); U_ = V_[:, -int(o['orak']):].flip(1); PR = U_ @ (U_.T @ R_); PRP = (PR @ U_) @ U_.T
                if part == 1: R_ = PR + PR.T - 2 * PRP
                elif part == 2: R_ = PRP
                elif part == 3: R_ = R_ - PR - PR.T + PRP
                elif part in (4, 5):
                    u1 = U_[:, :1]; Us = U_[:, 1:] if part == 4 else u1; q = R_ @ Us; q = q - U_ @ (U_.T @ q); R_ = Us @ q.T; R_ = R_ + R_.T
            if o['orarank']:
                le, Ve = torch.linalg.eigh(R_); oo = le.abs().argsort(descending=True)[:int(o['orarank'])]; R_ = (Ve[:, oo] * le[oo][None, :]) @ Ve[:, oo].T
            s_ = torch.diagonal(C_).clamp_min(1e-10).sqrt(); rm = float(o['ora']) * (3.0 / 1024) * (R_ - torch.diag(torch.diagonal(R_))) / (s_ ** 3)[:, None] / s_[None, :]
            return (mu_, Cpre_, k21_, k4_, k22_, rm if rem_ is None else rem_ + rm)
        cd.hooks['pair_in'] = pin_ora
    if o['ncy']:              # the physicist's research carrier (full matrices; O/physicist/ncarry.py), for parity with ncar3: partner = ncyp, forms = ncyforms
        import ncarry
        ncarry.install(cd, torch, strength=float(o['ncy']), rank=int(o['ncyrank']), k21=True, start=1, stop=14, conv='b', partner=int(o['ncyp']), forms=int(o['ncyforms']),
                       state_rank=int(o['ncysr']), skip=('k3prod',) if o['ncysk'] else ())
    if o['ncbk']:             # the physicist's BANK CHANNEL (second weight per bank column; frozen copy ncar_bank_frozen_s2s.py) as a K31 extra plane, strength ncbk
        import ncar_bank_frozen_s2s as ncar_bank
        ncar_bank.install(cd, torch, st, k31=float(o['ncbk']), lo=1, hi=14, forms=int(o['ncbf']), conv=int(o['ncbconv']), columns=True, top=int(o['ncbtop']))
    if o['nc'] >= 0 and int(o['nclaw']) == 6:      # NCAR6: the whole lean law on the bank's own column pairs (O/ncar6.py)
        import ncar6
        st['ncdump'] = {} if o['ncdump'] else None
        ncar6.install(cd, torch, st, k31=float(o['nc']), lo=int(o['nclo']), hi=int(o['nchi']), forms=int(o['ncbf']), births=float(o['ncbi']), rows=float(o['ncrw']), diag=float(o['ncdg']),
                      conv=float(o['ncconv']), ca=float(o['nca']), arest=bool(o['ncar']), k21=bool(o['nck21']), start=int(o['ncstart']), dump=st['ncdump'], level='read' if o['nclev'] else 'r31', dfit=int(o['ncdf']), dscale=float(o['ncds']), cenb=float(o['nccb']), cenr=float(o['nccr']), ptop=int(o['ncpt']), pdiag=int(o['ncpd']), pmu=int(o['ncpm']), c2k=int(o['ncc2']), c2sw=int(o['ncc2s']), k4d=float(o['nck4']), k4f=float(o['nck4f']), rowk=float(o['ncrk']), rown=float(o['ncrn']), k3d=float(o['nck3']), k3o=float(o['nck3o']), kq=float(o['nckq']), kcov=int(o['nckc']))
    elif o['nc'] >= 0 and int(o['nclaw']) == 5:      # NCAR5: structured low-rank state on pushed columns, no probes (O/ncar5.py)
        import ncar5
        st['ncdump'] = {} if o['ncdump'] else None
        ncar5.install(cd, torch, st, k31=float(o['nc']), r=int(o['ncr']), K=int(o['nck']), lo=int(o['nclo']), hi=int(o['nchi']), births=float(o['ncbi']), rows=float(o['ncrw']),
                      k21=bool(o['nck21']), cleanup=bool(o['nccl']), ca=float(o['nca']), sweeps=int(o['ncsw']), extra=int(o['ncx']), bankdiag=bool(o['ncbd']), dump=st['ncdump'])
    elif o['nc'] >= 0 and int(o['nclaw']) == 4:      # NCAR4: sketched rank-r state (O/ncar4.py)
        import ncar4
        st['ncdump'] = {} if o['ncdump'] else None
        ncar4.install(cd, torch, st, k31=float(o['nc']), r=int(o['ncr']), k=int(o['ncp']), K=int(o['nck']), conv=int(o['ncconv']), k21=bool(o['nck21']), lean=bool(o['nclean']),
                      lo=int(o['nclo']), hi=int(o['nchi']), dump=st['ncdump'], cleanup=bool(o['nccl']), start=int(o['ncstart']), passes=int(o['ncpass']), hutch=bool(o['nchu']), convtop=int(o['nctop']), rowdial=float(o['ncrow']), probe='carried' if o['ncpr'] else 'random')
    elif o['nc'] >= 0 and int(o['nclaw']) == 3:      # NCAR3: the complete law on K vectors (O/ncar3.py)
        import ncar3
        st['ncdump'] = {} if o['ncdump'] else None
        SK = {0: (), 1: ('k3c',), 2: ('gain',), 3: ('vtilt',), 4: ('rows', 'vtilt', 'own31')}[int(o['ncskip'])]
        ncar3.install(cd, torch, st, k31=float(o['nc']), K=int(o['nck']), conv=int(o['ncconv']), k21=bool(o['nck21']), plane='full' if o['ncplane'] else 'one',
                      lo=int(o['nclo']), hi=int(o['nchi']), sweeps=int(o['ncsw']), skip=SK, dump=st['ncdump'])
    elif o['nc'] >= 0:        # NCAR: free-running vector carrier of the network-specific norm coupling, used as an extra K31 plane (strength nc; 0 = carried but not used)
        import ncar
        st['ncdump'] = {} if o['ncdump'] else None
        ncar.install(cd, torch, st, k31=float(o['nc']), K=int(o['nck']), skip1=not bool(o['ncm1']), ct=float(o['nct']), cs=float(o['ncs']), ca=float(o['nca']), cb=float(o['ncb']),
                     lo=int(o['nclo']), hi=int(o['nchi']), conv=float(o['ncconv']), dump=st['ncdump'], law=int(o['nclaw']), q1c=float(o['ncq1']), sweeps=int(o['ncsw']))
    if o['k3l']:              # K31L: the young pairs' hub-local quartic form as a (3,1) plane (O/k31l.py), strength k3l, ages <= k3la, forms k3lf, k3lc=1: fourth-chaos part only
        import k31l
        k31l.install(cd, torch, st, k31=float(o['k3l']), age=int(o['k3la']), lo=1, hi=14, forms=int(o['k3lf']), chaos4=bool(o['k3lc']), level='read' if o['k3lr'] else 'r31', means={0: 'all', 1: 'resp', 2: 'iid'}[int(o['k3lm'])], agemin=int(o['k3ln']))
    if o['k3c']:              # K31C: the two-hub (chain-tree) part of the fresh fourth cumulant as a (3,1) plane (O/k31c.py), strength k3c, k3cc=1: fourth-chaos part only
        import k31c
        k31c.install(cd, torch, st, k31=float(o['k3c']), lo=1, hi=14, chaos4=bool(o['k3cc']), forms=int(o['k3cf']), age=int(o['k3ca']))
    if o['mxp']:              # the next layer's mean and variance are needed at its push: taken from the incoming state
        def sin_hook(l, state, ctx):
            if 1 <= l <= 15:
                w64 = cd.ws[l]; st['mx_mu'] = (w64.T @ state[0].to(torch.float64)).detach(); st['mx_v'] = ((w64.T @ state[1].to(torch.float64)) * w64.T).sum(1).detach()
            return state
        pc.wrap(cd, 'state_in', sin_hook)
    if o['kg'] or o['kgs']:   # K21 GAIN: the pair map's K21 gets + kg x (bank plane) on the kink rows (|alpha| < 1.5) and + kgs x (star plane) on all rows, from layer kgl on
        def kg_hook(l, pin, ctx):   # (truth planes, network 35: the card's K21 rest on the kink rows has cosine 0.93 with the truth and 0.82 of its norm at depth)
            if l < int(o['kgl']) or l > 14: return pin
            al = (pin[0] / torch.diagonal(pin[1]).clamp_min(1e-12).sqrt()).abs(); kink = (al < 1.5).to(pin[2].dtype)
            return (pin[0], pin[1], pin[2] + o['kg'] * kink[:, None] * ctx['cp'].to(pin[2].dtype) + o['kgs'] * ctx['sp'].to(pin[2].dtype), pin[3], pin[4], pin[5])
        pc.wrap(cd, 'pair_in', kg_hook)
    if o['k3m']:              # K31M: the mixed terms (third cumulant x curvature x covariance) as a (3,1) plane, research form (O/k31m.py)
        import k31m
        k31m.install(cd, torch, st, k31=float(o['k3m']), terms=(float(o['k3m1']), float(o['k3m2']), float(o['k3m3'])), lo=2, hi=14, forms=int(o['k3mf']))
    if o['k3dg']:             # K31DIAG: passive dump of candidate (3,1) planes for the comparison with the truth planes (O/k31diag.py)
        import k31diag
        st['k3dg'] = {}; k31diag.install(cd, torch, st, layers=(3, 5, 9, 12, 14), out=st['k3dg'])
    if o['ds'] != 1.0: pc.wrap(cd, 'delta', lambda l, d, ctx: d if l == 0 else d * o['ds'])      # shrink the learned mean correction (layers 1-15)
    if o['rf']:               # response-row floor of the bank read (card: .01 = rows with a gate between 0.01 and 0.99); rows of the read = bill
        cd.knobs['resp_floor'] = float(o['rf']); st['rows'] = {}
        sf0 = cd.select_floor
        def sf(score, floor):
            r = sf0(score, floor); st['rows'][len(st['rows'])] = int(r[0].numel()); return r
        cd.select_floor = sf
    if o['rcx'] != 1.0: cd.knobs['recenter'] = float(cd.knobs['recenter']) * o['rcx']               # recentring strength times x
    if o['edge'] != 1.0: pc.wrap(cd, 'edge', lambda l, inc, ctx: inc * o['edge'])                # learned covariance edge (layers 8-14) times x
    if o['e2']:
        # second-order Edgeworth term of the mean formula, missing in the card: (k3^2 / 72) E f^(6) = e2 * k3^2 He4(alpha) pdf / (72 sigma^5), added to the mean
        # through the learned-correction slot (so the covariance is recentred for it like for any mean shift)
        def e2_hook(l, d, ctx):
            if l == 0: return d
            aux = ctx['aux']; a = aux['alpha']; k3 = aux['pre_k3']
            return d + o['e2'] * (k3 * k3 / 72) * (a ** 4 - 6 * a * a + 3) * aux['pdf'] / aux['sigma'] ** 5
        pc.wrap(cd, 'delta', e2_hook)
    pred = pc.run(cd); mse = pc.mse(pred, tp)
    if o['nc'] >= 0 and o['ncdump'] and st.get('ncdump'):
        np.savez(HERE / f'ncar_dump_rank{rank:04d}_{name}.npz', **{f'{k}_{l}': v[k].numpy() for l, v in st['ncdump'].items() for k in v})
    if o['k3dg'] and st.get('k3dg'): np.savez(HERE / f'k31diag_rank{rank:04d}_{name}.npz', **{f'{k}_{l}': v[k] for l, v in st['k3dg'].items() for k in v})
    if o['dump']: np.savez(HERE / f'e30_dump_rank{rank:04d}_{name}.npz', **{f'q_{l}': v.numpy() for l, v in st.get('qdump', {}).items()})
    if o['rf']: print('ROWS', name, list(st['rows'].values()), file=sys.stderr, flush=True)
    if (o['rg'] or o['rgm']) and st.get('rg_stat'): print('RG', name, {k: tuple(round(x, 4) for x in v) for k, v in st['rg_stat'].items()}, file=sys.stderr, flush=True)      # per layer: rms and largest row gain
    if o['od'] and st.get('od_stat'): print('OD', name, {k: tuple(round(x, 5) for x in v) for k, v in st['od_stat'].items()}, file=sys.stderr, flush=True)      # per layer: relative rms error of the card's post variances, their mean relative error, relative Frobenius error of the off-diagonal
    if o['sc'] and st.get('sc_stat'): print('SC', name, {k: tuple(round(x, 4) for x in v) for k, v in st['sc_stat'].items()}, file=sys.stderr, flush=True)      # per layer: |card - C-based| / |card|, |card - base| / |card|, share of hubs replaced
    return name, o, mse, st['stat'], dict(cv={k: (round(v[0], 5), round(v[1], 4)) for k, v in st['stat_cv'].items() if k in (5, 9, 13, 15)}, tau={k: round(v, 3) for k, v in st['taus'].items() if k in (4, 9, 13)})


base = None
for spec in ARMS:
    name, o, mse, stat, extra = run(spec)
    if base is None: base = mse
    print(json.dumps(dict(rank=rank, tag=tag, arm=name, opts=o, final_x1e9=round(float(mse[15]) * 1e9, 4), final_vs_first_arm_pct=round(100 * (float(mse[15]) / float(base[15]) - 1), 3),
                          all_layer_vs_first_arm_pct=round(100 * (float(mse.mean()) / float(base.mean()) - 1), 3),
                          per_layer_vs_first_arm_pct=[round(100 * (float(mse[i]) / float(base[i]) - 1), 2) for i in range(16)], per_layer_x1e9=[round(float(x) * 1e9, 2) for x in mse],
                          beta={k: round(v[0], 4) for k, v in stat.items() if k in (5, 9, 13, 15)}, dk_rel={k: round(v[1], 3) for k, v in stat.items() if k in (5, 9, 13, 15)},
                          cv=extra, wall=round(time.time() - t0, 1))), flush=True)
