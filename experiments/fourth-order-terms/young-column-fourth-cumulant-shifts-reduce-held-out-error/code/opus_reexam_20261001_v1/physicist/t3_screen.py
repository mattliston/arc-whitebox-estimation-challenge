#!/usr/bin/env python3
"""TAG opus-physicist T3 screen: closed-loop test of ZERO-COST completions of the third-order births named by the T2 budget, on the current
torch card (pcard.build), no refit. Arms are '<name>:<key>=<value>,...' with keys
   on=0|1      learned mean correction on / off (layers 1-15)
   bsrc=x      the b source radial*d*v/6 scaled by x (physics of the feed: about 0.75)
   kw=x        the fourth-cumulant part of the birth weight, d(1 + x radial/3) instead of d(1 + radial/3)  (legs scaled by the square root)
   l22=x       new leg -(d_i/4) eta_c K22[i,c], K22 = radial (v_i v_c + 2 C_ic^2)/3 (the elliptical partner of the b source), strength x
   vsk=x       vertex skew: birth weight d_c -> d_c (1 - x skew_c (alpha_c^3 - 3 alpha_c)/6)
   cp4=x       FOURTH-ORDER WEIGHT ON THE BANK'S COLUMNS (T4): every column carries A = E f'''/birth = -alpha/(sigma (1 + radial/3)) of its
               hub at birth (lam4 = x lam A); at each later layer the YOUNG carried columns (not yet in the old-tier frame) are shifted in
               place, g o pp -> g o pp + x A (d/2) o pp o pp: the feed of that fourth cumulant into births (zero extra columns)
   cp4anchor=1 with cp4: the plane of the added tensor is removed from the star (plane stays anchored to the pair-map target; costly, test only)
   cp4anchor=2 with cp4: CHEAP anchor: the plane of the added tensor is approximated by the b-carrier form v_i beta_k + 2 C_ik beta_i with
               beta_k = sum_i v_i P[i,k] / sum_i v_i^2 (two matrix-vector passes over the young columns) and its exact diagonal
               3 sum_r lam4_r u_ri^2 om_ri; only that is removed from the star (elementwise n^2)
   cp4age=k    with cp4: shift only columns at most k layers old (default: all columns not yet in the old-tier frame)
   cp4shuf=1   CONTROL with cp4: the birth scalar A is permuted among the hubs of each cohort (same permutation for the +- partners)
   cp4hi=x     with cp4: strength x at layers >= 8 (cp4 at layers 0-7)
   cp4off=e    with cp4: birth scalar A + e / sigma_c (a positive offset common to all hubs; the free fit wants e about 0.15-0.25)
   cp4planes=x TEST ONLY (costs two plane reads natively): the same column fourth cumulant also supplies the pair map's COVARIANCE output:
               K31c[a,c] = sum_r lam_r A_r pp_ra^3 pp_rc and K22c[a,c] = sum_r lam_r A_r pp_ra^2 pp_rc^2 over the columns <= cp4age layers
               old, each with its isotropic-like part removed by least squares (forms (k4_a/v_a) C_ac; v_a v_c and C_ac^2), strength x,
               added through the linear series terms of the covariance (as the card's covariance knobs are)
   cp4pl=31|22 with cp4planes: only the K31 part or only the K22 part
   cp4a2=e     with cp4: birth scalar A + e alpha_c^2 / sigma_c
   f4b=c       F4b (module f4b.py): the exact local fourth cumulant S1..S4 of every young hub fed by two in-place shifts per pair, strength c
               (physical value 1), cheap anchor; f4bage=k (default 2); f4bmode=1: rule A through the same code (must equal cp4=c);
               f4bterms=124: keep only those S_m; f4banchor=1: the b-like part of the added tensor is taken out of the carried b vector
               (plane and distinct content) instead of out of the star (plane only); f4blo=l, f4bhi=l: shifts at the end of steps lo..hi only; f4bbl=L: b anchor at steps <= L, star anchor after;
               f4bc1..f4bc4, f4bt2..f4bt4 (with f4b=1): mixture, S_m = f4bc_m x exact + f4bt_m x F4's rule (missing keys are 0)
   X1,X0,X2,bscale,s_new,s_carry,c0,c1,k4a,k4b,k4c = value   any of the card's 12 dials
Reported per arm: final-layer and all-layer MSE (1e-9) against the organizers' truth; K21 kink-block plane error and per-unit k3 error
at layers 6/10/12/14 (noise-free: products of the two independent 8M truth blocks of t2_mc.py, when that file exists for the network).
usage: t3_screen.py <rank> <threads> <out tag> <arm> [<arm> ...]   -> appends to t3_screen_rank{rank}_{tag}.jsonl, writes t3_pred_rank{rank}_{tag}.npz"""
import sys, json, time, gc, math, resource
import numpy as np, torch
import pcard as pc
import t2_lib as L
import f4b as F4B
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2]); tag = sys.argv[3]; ARMS = sys.argv[4:]
W = pc.weights(rank); tp = pc.truth(rank); N = 1024
mcf = pc.HERE / f't2mc_rank{rank:04d}.npz'; TRUTH = {}
if mcf.exists():
    MCF = np.load(mcf); MC = pc.mcall(rank)
    for l in [int(x) for x in MCF['layers']]:
        Lx = l + 1; CL = MC[f'cov_pre_{Lx}'].astype(np.float64)
        TRUTH[Lx] = [L.k21_from_sums(MCF[f'M21_zn_L{l}_b{b}'].astype(np.float64), MCF[f'S1_zn_L{l}_b{b}'], MCF[f'S2_zn_L{l}_b{b}'], CL) for b in (0, 1)]
    VT = {Lx: np.diag(MC[f'cov_pre_{Lx}']).astype(np.float64) for Lx in TRUTH}
DIAL_KEYS = set(pc.DIALS)


def make_bd(D, X, side):
    """JT.make_bd (the trainer's three-direction version) plus the optional l22 leg and a per-hub weight factor."""
    def diag0(x): return torch.diagonal_scatter(x, torch.zeros(x.shape[0], dtype=x.dtype))
    def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
        n = int(bp.shape[0]); c = coefficients.to(torch.float64)
        alpha, sigma, g = aux['alpha'], aux['sigma'], aux['gate']; eta = alpha / sigma
        half_d = 0.5 * d; negative_half_g = -0.5 * g; negative_half_d = -half_d
        denominator = torch.clamp_min(torch.sum(bp * bp, dim=0), 1e-20)
        change = torch.zeros((n, n), dtype=torch.float64)
        add = {0: D['X0'], 2: D['X2']}
        for j in range(3):
            if j == 0: direction = half_d[:, None] * K21
            elif j == 1: direction = (negative_half_g[:, None] * K21.T) * eta[None, :]
            else: direction = (negative_half_d[:, None] * (Cpre * Cpre)) * eta[None, :]
            direction = diag0(direction)
            if j == 1:
                change = change + direction * D['X1']
            else:
                raw = direction
                projection = torch.sum(bp * direction, dim=0) / denominator
                direction = direction - bp * projection[None, :]
                change = change + direction * c[j] + raw * add[j]
        radial = side['radial']; v = sigma * sigma
        if X.get('l22', 0.0):
            K22 = radial * (torch.outer(v, v) + 2 * Cpre * Cpre) / 3
            change = change + X['l22'] * diag0((-0.25 * d)[:, None] * K22 * eta[None, :])
        f = torch.ones(n, dtype=torch.float64)
        if 'kw' in X: f = f * (1 + X['kw'] * radial / 3) / (1 + radial / 3)
        if X.get('vsk', 0.0):
            skew = aux['pre_k3'] / (sigma * sigma * sigma)
            f = f * (1 - X['vsk'] * skew * (alpha ** 3 - 3 * alpha) / 6).clamp(0.25, 4.0)
        if 'kw' in X or X.get('vsk', 0.0):
            change = (bp + change) * f.sqrt()[None, :] - bp
        change = diag0(change)
        old32 = (bp * alive[:, None]).to(torch.float32); new32 = ((bp + change) * alive[:, None]).to(torch.float32)
        old = old32.to(torch.float64); new = new32.to(torch.float64); difference = new - old
        a64 = a.to(torch.float32).to(torch.float64)
        t64 = (t * alive).to(torch.float32).to(torch.float64); s64 = (s * alive).to(torch.float32).to(torch.float64)
        weight = (a64 * (t64 - s64)) * births; quadratic_weight = (a64 * (t64 * t64 - s64 * s64)) * births
        delta = ((2 * old * difference + difference * difference) * weight[None, :] + quadratic_weight[:, None] * difference.T)
        return new32, diag0(delta)
    return _bd_update


def run(spec):
    name, _, kv = spec.partition(':'); kw = dict(x.split('=') for x in kv.split(',') if x)
    learned = kw.pop('on', '1') == '1'
    dials = {k: float(kw.pop(k)) for k in list(kw) if k in DIAL_KEYS}; X = {k: float(v_) for k, v_ in kw.items()}
    assert set(X) <= {'bsrc', 'kw', 'l22', 'vsk', 'cp4', 'cp4anchor', 'cp4age', 'cp4shuf', 'cp4hi', 'cp4off', 'cp4a2', 'cp4planes', 'cp4pl', 'f4b', 'f4bage', 'f4bmode', 'f4bterms', 'f4banchor', 'f4blo', 'f4bhi', 'f4bbl', 'f4bc1', 'f4bc2', 'f4bc3', 'f4bc4', 'f4bt2', 'f4bt3', 'f4bt4', 'f4be2', 'f4bi2', 'f4bown', 'f4bbm', 'f4bcoll', 'f4bcollage'}, X
    cd = pc.build(W, learned=learned, dials=dials); side = {}; cap = {}
    if set(X) & {'kw', 'l22', 'vsk'}:
        cd.env['_bd_update'] = make_bd({k: float(v_) for k, v_ in pc.LC.JT.DIALS_T.items()}, X, side)
    if 'f4b' in X:
        mix = any(k in X for k in ('f4bc1', 'f4bc2', 'f4bc3', 'f4bc4', 'f4bt2', 'f4bt3', 'f4bt4'))
        F4B.install(cd, torch, X['f4b'], age=int(X.get('f4bage', 2)), mode='mix' if mix else ('ruleA' if X.get('f4bmode', 0) else 'exact'),
                    cs=tuple(X.get(k, 0.0) for k in ('f4bc1', 'f4bc2', 'f4bc3', 'f4bc4')), ths=tuple(X.get(k, 0.0) for k in ('f4bt2', 'f4bt3', 'f4bt4')), even2=X.get('f4be2', 1.0), iso2=X.get('f4bi2', 0.0), form='own' if X.get('f4bown', 0) else 'ride', metric='birth' if X.get('f4bbm', 0) else 'gram', coll=X.get('f4bcoll', 0.0), coll_age=int(X.get('f4bcollage', 99)),
                    terms=tuple(int(ch) for ch in str(int(X.get('f4bterms', 1234)))), anchor=int(X['f4bbl']) if 'f4bbl' in X else ('b' if X.get('f4banchor', 0) else 'star'), layers=(int(X.get('f4blo', 1)), int(X.get('f4bhi', 14))))
    def pair_in(l, pin, ctx):
        side['radial'] = ctx['radial']
        if l in TRUTH: cap[l] = (ctx['k21'].numpy().copy(), ctx['response_mask'].numpy() > .5, ctx['alive'].numpy() > .5, torch.diagonal(pin[1]).numpy().copy())
        return pin
    pc.wrap(cd, 'pair_in', pair_in)
    if 'cp4' in X:
        th_lo = X['cp4']; th_hi = X.get('cp4hi', th_lo); anchor = int(X.get('cp4anchor', 0)); maxage = int(X.get('cp4age', 99)); s2 = dict(A=torch.zeros(0, dtype=torch.float32))
        push0 = cd.shared['push']
        def push(t, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
            r = push0(t, layer, w, p, lam, previous, Q, F, core, stops, rank=rank); s2['pp'] = r[0]; s2['oc'] = int(r[4]); s2['lam_in'] = lam; s2['stops_in'] = stops; return r
        cd.shared['push'] = push
        if X.get('cp4planes', 0.0):
            xp = X['cp4planes']; which = int(X.get('cp4pl', 0)); derivatives = cd.env['derivatives']; eyeN = torch.eye(N, dtype=torch.bool)
            def planes_cov(l, C, ctx):
                if l < 2 or l > 14: return C
                A_ = s2['A']; pp = s2['pp']; lam_ = s2['lam_in']; stops_ = s2['stops_in']; nc = int(pp.shape[1])
                oc = max(s2['oc'], int(stops_[l - maxage - 1]) if l - maxage >= 1 else 0)
                if nc <= oc: return C
                aux = ctx['aux']; Cpre = ctx['Cpre']; sg = aux['sigma']; al = aux['alpha']; v = torch.diagonal(Cpre); k4 = ctx['k4']
                P = pp[:, oc:].to(torch.float64); c4 = (lam_[oc:nc] * A_[oc:]).to(torch.float64)
                K31c = (P ** 3) @ (c4[:, None] * P.T); K22c = (P ** 2) @ (c4[:, None] * (P ** 2).T)
                Aj, _ = derivatives(al, aux['pdf'], aux['gate'], aux['gaussian_mean'] / sg, 5); A1, A2, A3, A4, A5 = Aj[1:6]
                off = (~eyeN).to(torch.float64); w31 = (al.abs() * aux['pdf'] / v)[:, None] * aux['gate'][None, :] * off; w22 = torch.outer(aux['pdf'] / sg, aux['pdf'] / sg) * off
                iso31 = (k4 / v)[:, None] * Cpre
                a31 = ((K31c * iso31) * w31 ** 2).sum() / ((iso31 * iso31) * w31 ** 2).sum(); K31p = (K31c - a31 * iso31) * off
                B = torch.stack(((torch.outer(v, v) * w22).flatten(), (Cpre * Cpre * w22).flatten()), 1)
                co = torch.linalg.lstsq(B, (K22c * w22).flatten()[:, None]).solution.flatten(); K22p = (K22c - co[0] * torch.outer(v, v) - co[1] * Cpre * Cpre) * off
                ss = torch.outer(sg, sg); rho = (Cpre / ss) * off; rho2 = rho * rho / 2
                dT31 = xp * K31p / (v[:, None] * ss); dT22 = xp * K22p / torch.outer(v, v)
                t31a = torch.outer(A3, A1) + rho * torch.outer(A4, A2) + rho2 * torch.outer(A5, A3)
                t22 = torch.outer(A2, A2) + rho * torch.outer(A3, A3) + rho2 * torch.outer(A4, A4)
                Xm = torch.zeros_like(C)
                if which in (0, 31): Xm = Xm + dT31 * t31a / 6
                d22 = dT22 * t22 / 4 if which in (0, 22) else torch.zeros_like(C)
                dcov = ((Xm + Xm.T) + d22) * ss
                return C + dcov * off
            pc.wrap(cd, 'cov_pm', planes_cov)
        sel0 = cd.select
        def select(score):
            r = sel0(score); s2['sel'] = r[0]; return r
        cd.select = select
        def cp4(l, out, ctx):
            if l > 14: return out
            th = th_hi if l >= 8 else th_lo
            aux = ctx['aux']; p, lam, star, alive = out[2], out[3], out[4], out[6]
            A = s2['A']; pp = s2['pp']; oc = s2['oc']; nc = int(pp.shape[1]); assert A.numel() == nc, (l, A.numel(), nc)
            stops = out[12]                                                       # cohort j occupies columns stops[j-1]..stops[j]; cohort j is l - j layers old
            if maxage < 99 and l - maxage >= 1: oc = max(oc, int(stops[l - maxage - 1]))
            if nc > oc and th != 0.0:
                dh = ((aux['pdf'] / aux['sigma'] / 2) * alive).to(torch.float32)
                old_cols = p[:, oc:nc]; om = dh[:, None] * (pp[:, oc:] * pp[:, oc:])
                new_cols = old_cols + om * (th * A[oc:])[None, :]
                if anchor == 1:
                    lm = lam[oc:nc].to(torch.float64); o64, n64 = old_cols.to(torch.float64), new_cols.to(torch.float64)
                    dP = ((n64 * n64) * lm[None, :]) @ n64.T - ((o64 * o64) * lm[None, :]) @ o64.T
                    star = star - (3 * dP.T - 2 * torch.diag(torch.diagonal(dP))).to(star.dtype)
                elif anchor == 2:
                    C = out[1]; v = torch.diagonal(C); c_ = (lam[oc:nc] * th * A[oc:]).to(torch.float64); u64, o64 = old_cols.to(torch.float64), om.to(torch.float64)
                    uu = u64 * u64; uo = u64 * o64
                    vP = o64 @ (c_ * (v @ uu)) + 2 * (u64 @ (c_ * (v @ uo)))
                    beta = vP / (v * v).sum()
                    dP = v[:, None] * beta[None, :] + 2 * C * beta[:, None]
                    dP = torch.diagonal_scatter(dP, 3 * (uu * o64) @ c_)
                    star = star - (3 * dP.T - 2 * torch.diag(torch.diagonal(dP))).to(star.dtype)
                p = torch.cat((p[:, :oc], new_cols, p[:, nc:]), dim=1)
            sel = s2['sel']; An = -aux['alpha'][sel] / (aux['sigma'][sel] * (1 + ctx['radial'] / 3))
            An = (An + X.get('cp4off', 0.0) / aux['sigma'][sel] + X.get('cp4a2', 0.0) * aux['alpha'][sel] ** 2 / aux['sigma'][sel]).to(torch.float32)
            if X.get('cp4shuf', 0): An = An[torch.randperm(An.numel(), generator=torch.Generator().manual_seed(1000 + l))]
            s2['A'] = torch.cat((A, An, An))
            return out[:2] + (p, lam, star) + out[5:]
        pc.wrap(cd, 'state_out', cp4)
    if 'bsrc' in X:
        bs = float(pc.LC.JT.DIALS_T['bscale'])
        pc.wrap(cd, 'b_post', lambda l, b, ctx: b + bs * (X['bsrc'] - 1) * ctx['radial'] * ctx['d'] * ctx['vv'] / 6)
    t0 = time.time(); pred = pc.run(cd); mse = pc.mse(pred, tp)
    rec = dict(rank=rank, arm=name, spec=spec, learned=learned, final_x1e9=round(float(mse[15]) * 1e9, 4), all_layer_x1e9=round(float(mse.mean()) * 1e9, 4),
               per_layer_x1e9=[round(float(x) * 1e9, 2) for x in mse])
    for Lx, (k21, resp, alive, v) in sorted(cap.items()):
        blk = np.outer(resp, alive) & ~np.eye(N, dtype=bool); T0, T1 = TRUTH[Lx]
        rec[f'plane_L{Lx}'] = round(100 * math.sqrt(max(float(((k21 - T0)[blk] * (k21 - T1)[blk]).sum()), 0.) / float((T0[blk] * T1[blk]).sum())), 3)
        k3, a0, a1 = np.diag(k21)[alive], np.diag(T0)[alive], np.diag(T1)[alive]
        rec[f'k3_L{Lx}'] = round(100 * math.sqrt(max(float(((k3 - a0) * (k3 - a1)).sum()), 0.) / float((a0 * a1).sum())), 3)
        rec[f'v_L{Lx}'] = round(100 * float(np.sqrt(np.mean((v / VT[Lx] - 1) ** 2))), 4)
    if 'f4b' in X:
        rec['f4b_rho'] = {l: round(i['rho'], 3) for l, i in cd.f4b_info.items()}; rec['f4b_masked'] = sum(i['masked'] for i in cd.f4b_info.values())
        rec['f4b_beta_over_b'] = {l: round(i['beta_norm_over_b'], 3) for l, i in cd.f4b_info.items()}; rec['f4b_cos_beta_b'] = {l: round(i['cos_beta_b'], 3) for l, i in cd.f4b_info.items()}
    rec['wall'] = round(time.time() - t0, 1); rec['rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)
    PRED[name] = pred.astype(np.float64)
    del cd; gc.collect()
    return rec


out = pc.HERE / f't3_screen_rank{rank:04d}_{tag}.jsonl'; PRED = {}
for spec in ARMS:
    rec = run(spec)
    with out.open('a') as f: f.write(json.dumps(rec) + '\n')
    print(json.dumps({k: v_ for k, v_ in rec.items() if k != 'per_layer_x1e9'}), flush=True)
np.savez(pc.HERE / f't3_pred_rank{rank:04d}_{tag}.npz', truth=tp, **PRED)       # predictions (16, 1024) per arm, for the shadow test
