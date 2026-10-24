#!/usr/bin/env python3
"""TAG fable-physicist (RF6-11): which CONSUMER of young third-order content carries the Tucker hypersensitivity?
Card: the analyst's s6bcard (covhunter's HookCard + v3 knobs + L15 cut) with B2 ep1 literals, built on RouteCard, a copy of
covstep.HookCard.step with every hook call kept, plus a carried difference dP = (projected - exact) young columns (post space).
From layer L0 the NEWBORN cohort is projected once at birth: its covariance legs st' onto the common top-r eigenspace of all young
pairs' sum (|lam||e'|)^2 st' st'^T (pinned legs and unit bonds exact; rho = u/gam recorded at birth). The card's own state stays EXACT;
dP is pushed and gated with the same maps (the projected cohorts, born >= L0 >= 9, never join the every3 old tier).
Each consumer of the pushed columns reads pp (exact) or pp + W^T dP according to ROUTE:
  none  no consumer (control: must reproduce the card)     all  every consumer (the birth-only Tucker arm)
  a     the plane read (-> cp -> K21 for the pair map, births, features, edges, and the star residual)
  b     the k3 diagonal (per-unit skew; also the L15 bank k3)
  c     the omitted-radial source (energy, column and its own diagonal)
  d     every consumer, but only for cohorts aged >= 2 (each cohort's first read is exact; the damage via the carry only)
Diagnostic (DIAG=1, r35 truth planes public_g19_attribution_v1/mps_planes_rank0035.npz): at L10 and L14, on the kink block (response
rows x live columns, off-diagonal): the card's OWN K21 error vs the 8M truth and the plane change the projection induces, both
relative to the truth block norm; the same for the k3 diagonal (alive units; truth = the diagonal of the 8M K21 plane).
usage: route_eval.py <route> <r> <L0> <nets e.g. valid:8 or 35> <threads> <tag> [diag 0|1]"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_covhunter_v1'))
import numpy as np, torch
import covstep as cs
import s6bcard
ROUTE, R, L0, NETS, threads, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
DIAG = len(sys.argv) > 7 and sys.argv[7] == '1'
assert ROUTE in ('none', 'all', 'a', 'b', 'c', 'd')
torch.set_num_threads(threads); torch.set_grad_enabled(False); t = torch
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
nets = inputs['valid'][:int(NETS.split(':')[1])] if NETS.startswith('valid:') else [int(x) for x in NETS.split(',')]
TRUTH = np.load(P2 / 'public_g19_attribution_v1' / 'mps_planes_rank0035.npz') if DIAG else None


def cohorts(stops):
    out, first = [], 0
    for s in stops:
        s = int(s); out.append((first, s)); first = s
    return out


class RouteCard(cs.HookCard):
    def src(self, pp, dPP, consumer):
        if dPP is None or ROUTE == 'none':
            return pp
        if ROUTE == 'all' or ROUTE == consumer:
            return pp + dPP
        if ROUTE == 'd':
            return pp + dPP * self._age2[None, :]
        return pp

    def step(self, layer, state):
        t, f = self.t, self.env
        self.call_layer, self.slot = layer, 0
        state = self.h('state_in', layer, state, {})
        m, C, p, lam, star, b, previous_alive, radial, hstate, Q, F, core, stops, previous = state
        w, w64 = self.w[layer], self.ws[layer]
        mu = self.h('mu', layer, w64.T @ m, {})
        pp, U, F, core, old_columns = self.shared['push'](
            t, layer, w, p, lam, previous, Q, F, core, stops, rank=cs.TIER_RANK)
        dP = getattr(self, '_dP', None)
        dPP = (w[previous, :].T @ dP[previous, :]) if (dP is not None and dP.shape[1] == p.shape[1] and bool(dP.abs().sum() > 0)) else None
        if dPP is not None:
            age2 = t.ones(p.shape[1], dtype=pp.dtype)
            coh = cohorts(stops)
            if coh: age2[coh[-1][0]:coh[-1][1]] = 0.0        # the cohort born at layer-1 is at its first read
            self._age2 = age2
        bpre = w64.T @ b
        if layer == 15:
            pushed = C.to(t.float32) @ w
            vv = (w64 * pushed.to(t.float64)).sum(0)
            vv = self.h('vv15', layer, vv, {})
            self.variance(layer, vv)
            p64 = self.src(pp, dPP, 'b').to(t.float64)
            k3b = ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)
            k3s = f['_flms_star_diagonal'](w, star)
            k3 = k3b + k3s + 3 * vv * bpre
            k3 = self.h('k3_15', layer, k3, dict(bank=k3b, star=k3s, bterm=3 * vv * bpre, vv=vv, mu=mu, radial=radial))
            k4 = self.h('k4_15', layer, radial * vv * vv, dict(vv=vv, mu=mu, radial=radial, k3=k3))
            host, aux = f['nonlinear_scalar'](mu, vv, k3, k4)
            x = f['final_scalar_features'](layer, host, aux, w)
            delta, hs = f['gnn_step'](layer, x, aux['sigma'], w, hstate, self.P)
            delta = self.h('delta', layer, delta, dict(host=host, aux=aux, mu=mu, vv=vv))
            out = (host + delta.to(t.float64), C, p, lam, star, b, previous_alive,
                   radial, hs, Q, F, core, stops, previous)
            return self.h('state_out', layer, out, dict(host=host, aux=aux, mu=mu, vv=vv, k3=k3))
        Cpre = ((w.T @ w) if layer == 0 else (w.T @ (C.to(t.float32) @ w))).to(t.float64)
        Cpre = self.h('cpre', layer, Cpre, {})
        U_ = lambda name: self.h('cpre_use', layer, Cpre, dict(use=name))
        vv = t.diagonal(U_('v'))
        self.variance(layer, vv)
        gate = t.erfc(-(mu / vv.clamp_min(1e-10).sqrt()) / math.sqrt(2)) / 2
        indices, alive = self.select(gate)
        live = alive[:, None] * alive[None, :]
        response_score = t.where(alive.bool(), t.minimum(gate, 1 - gate), -t.ones_like(gate))
        if self.knobs['resp_floor'] == .01:
            response_indices, response_mask = self.select(response_score)
        else:
            response_indices, response_mask = self.select_floor(response_score, self.knobs['resp_floor'])
        omitted = alive * (1 - response_mask)
        cp = t.zeros((1024, 1024), dtype=t.float64)
        if lam.numel():
            plane = self.shared['read'](t, self.src(pp, dPP, 'a'), lam, response_indices, indices, U, F, old_columns).to(t.float64)
            cp = cp.index_put((response_indices[:, None], indices[None, :]), plane)
            p64 = self.src(pp, dPP, 'b').to(t.float64)
            cpdiag = ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)
            cp = self.diagonal(cp, cpdiag)
            if DIAG and layer in (10, 14) and TRUTH is not None:
                tb = t.as_tensor(TRUTH[f'K21_{layer}'], dtype=t.float64)[response_indices][:, indices]
                offd = (response_indices[:, None] != indices[None, :]).to(t.float64)
                nt = float((tb * offd).norm())
                rec = dict(truth_block_norm=nt)
                if dPP is not None:
                    plane_x = self.shared['read'](t, pp, lam, response_indices, indices, U, F, old_columns).to(t.float64)
                    plane_p = self.shared['read'](t, pp + dPP, lam, response_indices, indices, U, F, old_columns).to(t.float64)
                    rec['projection_change'] = float(((plane_p - plane_x) * offd).norm()) / nt
                al_ = alive.bool(); kt = t.as_tensor(np.diagonal(TRUTH[f'K21_{layer}']).copy(), dtype=t.float64)[al_]
                rec['k3_truth_norm'] = float(kt.norm())
                if dPP is not None:
                    q = (pp + dPP).to(t.float64); cpd_p = ((q * q * q) * lam.to(t.float64)).sum(1)
                    rec['k3_projection_change'] = float((cpd_p - cpdiag)[al_].norm()) / float(kt.norm())
                self._k3_truth = kt; self._al = al_
                self._diag_pending = (layer, rec, tb, offd, nt)
        sp = t.zeros_like(Cpre) if layer == 0 else f['_flms_star'](w, star)
        k21 = cp + sp + vv[:, None] * bpre[None, :] + 2 * U_('k21b') * bpre[:, None]
        if getattr(self, '_diag_pending', None) is not None and self._diag_pending[0] == layer:
            L_, rec, tb, offd, nt = self._diag_pending
            kc = k21[response_indices][:, indices]
            rec['card_own_error'] = float(((kc - tb) * offd).norm()) / nt
            rec['k3_card_own_error'] = float((t.diagonal(k21)[self._al] - self._k3_truth).norm()) / float(self._k3_truth.norm())
            LOG.setdefault(self._net, {})[L_] = rec; self._diag_pending = None
        k4 = radial * vv * vv
        Ck22 = U_('k22')
        k22 = radial * (t.outer(vv, vv) + 2 * Ck22 * Ck22) / 3
        pair = f['_nonlinear_first_gaussian'] if layer == 0 else f['_nonlinear_pair_batch']
        ctx = dict(mu=mu, Cpre=Cpre, k21=k21, k4=k4, k22=k22, radial=radial, gate=gate, alive=alive, response_mask=response_mask,
                   cp=cp, sp=sp, bpre=bpre, pair=pair, m_in=m, C_in=C, hstate=hstate, w=w)
        pin = self.h('pair_in', layer, (mu, U_('pair'), k21, k4, k22, None), ctx)
        if pin[5] is None:
            host, C, target, new_radial, aux = pair(pin[0], pin[1], pin[2], pin[3], pin[4], 'covariance')
        else:
            host, C, target, new_radial, aux = pair(pin[0], pin[1], pin[2], pin[3], pin[4], 'covariance', remainder31=pin[5])
        if 'cpre_use' in self.hooks:
            aux = dict(aux); aux['pre_cov'] = U_('feat')
        ctx.update(host=host, aux=aux, target=target)
        C = self.h('cov_pm', layer, C, ctx)
        if lam.numel() and self.knobs['omitted_radial']:
            aa = (aux['gate'] - host * aux['pdf'] / aux['sigma']) * omitted
            beta = 2 * (aux['gaussian_mean'] - host * aux['gate']) * alive
            q64 = self.src(pp, dPP, 'c').to(t.float64)
            qdiag = ((q64 * q64 * q64) * lam.to(t.float64)).sum(1)
            energy = (q64 * q64).T @ aa
            column = q64.T @ beta
            extra = 6 * ((lam.to(t.float64) * energy * column).sum() - (qdiag * aa * beta).sum())
            new_radial = new_radial + extra / (t.trace(C) ** 2 + 2 * (C * C).sum())
        if 8 <= layer <= 14:
            context = {'left': self.edge['context_left'], 'right': self.edge['context_right']}
            C = C + self.h('edge', layer, f['_message_edge_source'](layer, host, k21, aux, w, hstate, self.P, context), ctx)
        ctx['C_edge'] = C
        x = f['features'](layer, host, C, k21, aux, w)
        delta, hs = f['gnn_step'](layer, x, aux['sigma'], w, hstate, self.P)
        delta = delta.to(t.float64)
        delta = self.h('delta', layer, delta, ctx)
        al = self.knobs['recenter']
        if al == 1.0 and self.knobs['recenter_part'] == 'all':
            C = C - t.outer(host, delta) - t.outer(delta, host) - t.outer(delta, delta)
        else:
            rc = t.outer(host, delta) + t.outer(delta, host) + t.outer(delta, delta)
            if self.knobs['recenter_part'] == 'diag':
                rc = t.diag(t.diagonal(rc))
            elif self.knobs['recenter_part'] == 'off':
                rc = rc - t.diag(t.diagonal(rc))
            C = C - al * rc
        m = host + delta
        ctx.update(delta=delta, m_out=m)
        C = self.h('cov_out', layer, C, ctx)
        g = aux['gate']
        d = aux['pdf'] / aux['sigma']
        birth = d * (1 + radial / 3)
        b = g * bpre + radial * d * vv / 6
        if self.knobs['bfix_gamma']:
            b = b + self.knobs['bfix_gamma'] * (new_radial - radial) / 6 * m
        b = self.h('b_post', layer, b, dict(m=m, host=host, g=g, d=d, vv=vv, bpre=bpre, radial=radial, new_radial=new_radial))
        selected, births = self.select(t.minimum(g, 1 - g))
        inherited = g[:, None] ** 2 * g[None, :] * cp
        bp = self.diagonal(g[:, None] * U_('births'), 0.)
        trace_post = t.diagonal(C)[:, None] * b[None, :] + 2 * C * b[:, None]
        beta = 2 * (aux['gaussian_mean'] - host * g)
        innovation = t.diagonal(target) - t.diagonal(inherited) - t.diagonal(trace_post)
        safe_birth = birth.clamp_min(1e-12)
        u = t.maximum(t.minimum(beta / safe_birth, 8 * vv), -8 * vv)
        disc = 4 * innovation / safe_birth - 3 * u * u
        gap = t.minimum(t.maximum(disc, .01 * vv * vv).sqrt(), 16 * vv)
        plus, minus = (u + gap) / 2, (u - gap) / 2
        coefficient = birth / gap
        born = bp * bp * (birth * births)[None, :] + (beta * births)[:, None] * bp.T
        born = self.diagonal(born, birth * births * (3 * u * u + gap * gap) / 4)
        c = self.cstar[layer] if layer == 0 else f['_bdnh_coefficients'](hs, self.A, self.cstar[layer])
        stored, increment = f['_bd_update'](bp, U_('bd'), k21, aux, d, alive, births, coefficient, plus, minus, c)
        born = born + increment
        cp_post = (inherited + born) * live
        positive = self.diagonal(stored, (plus * alive).to(t.float32))
        negative = self.diagonal(stored, (minus * alive).to(t.float32))
        gate32 = alive.to(t.float32)[:, None] * g.to(t.float32)[:, None]
        p = t.cat((gate32 * pp, positive[:, selected], negative[:, selected]), dim=1)
        nsel = int(selected.numel())
        dP_new = t.zeros_like(p)
        if dPP is not None:
            dP_new[:, :pp.shape[1]] = gate32 * dPP                               # the carried difference follows the same gate
        lam = t.cat((lam, coefficient[selected].to(t.float32), -coefficient[selected].to(t.float32)))
        if old_columns:
            Q = alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * U)
        stops = stops + (int(lam.numel()),)
        # record rho for the newborn pairs; project the newborn cohort at birth (layer >= L0)
        a0 = int(p.shape[1]) - 2 * nsel
        if nsel:
            Pp, Pm = p[:, a0:a0 + nsel].to(t.float64), p[:, a0 + nsel:].to(t.float64)
            idx = (Pp - Pm).abs().argmax(0); plus_ = Pp[idx, t.arange(nsel)]; minus_ = Pm[idx, t.arange(nsel)]
            gp = plus_ - minus_; ok = gp.abs() > 1e-30
            ratio = t.where(ok, (plus_ + minus_) / t.where(ok, gp, t.ones_like(gp)), t.zeros_like(gp))
            if not hasattr(self, '_rho'): self._rho = {}
            for j, v in enumerate(ratio.tolist()): self._rho[a0 + j] = v
        if layer >= L0 and nsel:
            n_old = int(F.shape[1]) if F.dim() == 2 else 0
            ip, im = [], []
            for (s0, s1) in cohorts(stops):
                if s0 < n_old: continue
                k = (s1 - s0) // 2
                ip += list(range(s0, s0 + k)); im += list(range(s0 + k, s1))
            ip_t, im_t = t.tensor(ip), t.tensor(im)
            Xp, Xm = p[:, ip_t].to(t.float64), p[:, im_t].to(t.float64)
            rr = t.tensor([self._rho[i] for i in ip], dtype=t.float64)
            ep = (Xp - Xm) / 2; st = (Xp + Xm) / 2 - rr[None, :] * ep
            wgt = lam[ip_t].to(t.float64).abs() * ep.norm(dim=0)
            X = st * wgt[None, :]; G = X @ X.T
            ev, V = t.linalg.eigh((G + G.T) / 2); V = V.flip(1); Ur = V[:, :min(R, V.shape[1])]
            new = t.tensor([i >= a0 for i in ip])
            stp = Ur @ (Ur.T @ st[:, new])
            ipn, imn = ip_t[new], im_t[new]
            dP_new[:, ipn] = (stp + rr[new][None, :] * ep[:, new] + ep[:, new]).to(p.dtype) - p[:, ipn]
            dP_new[:, imn] = (stp + rr[new][None, :] * ep[:, new] - ep[:, new]).to(p.dtype) - p[:, imn]
        self._dP = dP_new
        rest = target - cp_post - trace_post
        star = (3 * rest.T - 2 * t.diag(t.diagonal(rest))).to(t.float32)
        out = (m, C, p, lam, star, b, alive, new_radial, hs, Q, F, core, stops, indices)
        return self.h('state_out', layer, out, ctx)


cs.HookCard = RouteCard                                    # s6bcard.card builds cs.HookCard at call time
LOG = {}
p0, a0_, c0, e0 = s6bcard.payload(torch)
sv = torch.load(HERE / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
for k in p0: p0[k].copy_(sv['p'][k])
a0_.copy_(sv['a']); c0.copy_(sv['c'])
for k in e0: e0[k].copy_(sv['e'][k])
t0 = time.time(); res = {}
for rk in nets:
    W = np.load(STORE / Path(man[rk]['path']).parent / 'weights.npy')
    truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rk:04d}.npz')['mc_post'][15].astype(np.float64)
    card = s6bcard.card(torch, W, p0, a0_, c0, e0); card._net = str(rk)
    pred = card.run(False).numpy().astype(np.float64)
    res[str(rk)] = float(((pred[15] - truth) ** 2).mean())
out = dict(route=ROUTE, r=R, L0=L0, mean=float(np.mean(list(res.values()))), valid=res, diag=LOG, wall=round(time.time() - t0, 1))
(HERE / 'route').mkdir(exist_ok=True); (HERE / 'route' / f'{tag}.json').write_text(json.dumps(out))
print(json.dumps({k: out[k] for k in ('route', 'r', 'L0', 'mean', 'wall')} | ({'diag': LOG} if LOG else {})), flush=True)
