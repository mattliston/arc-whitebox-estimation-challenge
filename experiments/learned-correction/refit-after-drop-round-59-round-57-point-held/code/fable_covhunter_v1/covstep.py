"""TAG fable-covhunter (CH-2): the torch research card (fable_refit_v1/model_bank_research.py, gamma-free bank port) with its step copied
VERBATIM plus named hook points and knobs, for per-layer injection and attribution of the carried-covariance error. With no hooks and
default knobs, HookCard reproduces model.Card bit-exactly (checked by parity() below).
Hooks: self.hooks[name] = fn(layer, value, ctx) -> value, ctx = dict of the step's locals at that point.
  'state_in'  value = incoming state tuple (resets of m/C to truth at l-1)
  'mu', 'cpre', 'vv15'  the pre-activation mean, covariance (L0-14) and variance (L15) right after they are formed (adjoint leaves)
  'k3_15', 'k4_15'  the final layer's per-unit k3 (ctx: bank / star / bterm parts) and k4 (= radial vv^2) before the scalar map
  'b_post'    the carried b vector right after its recursion (and the b-fix), before births / trace_post use it
  'cpre_use'  value = Cpre at each CONSUMER (ctx['use'] in v, k21b, k22, pair, feat (features+edge pre_cov), births, bd)
  'pair_in'   value = (mu, Cpre, k21, k4, k22, remainder31) for the MAIN pair-map call
  'cov_pm'    value = post covariance straight out of the pair map (cov-only channel swaps)
  'edge'      value = the message-edge increment (L8-14)
  'delta'     value = the GNN mean correction delta
  'cov_out'   value = the carried covariance after recentering
  'state_out' value = outgoing state tuple (post swaps)
Knobs: self.knobs['recenter'] = alpha (C -= alpha*(h d^T + d h^T + d d^T)), ['recenter_part'] in all|diag|off,
  ['resp_floor'] = response-row select floor (bank .01), ['omitted_radial'] = bool, ['bfix_gamma'] = gain b-fix (S2: 0.1)."""
import math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import json
import numpy as np, torch
import model_bank_research as model

STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
MAN = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
TIER_RANK = model.TIER_RANK


def weights(rank):
    return np.load(STORE / Path(MAN[rank]["path"]).parent / "weights.npy")


def truthpost(rank):
    return np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)


class HookCard(model.Card):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.hooks, self.knobs = {}, dict(recenter=1.0, recenter_part='all', resp_floor=.01, omitted_radial=True, bfix_gamma=0.0)

    def observe(self, kind, layer, record):   # no checkpoint-replay asserts: steps are re-run from modified states
        pass

    def h(self, name, layer, value, ctx):
        fn = self.hooks.get(name)
        return value if fn is None else fn(layer, value, ctx)

    def select_floor(self, score, floor):
        t = self.t
        count = min(1024, ((int((score.detach() >= floor).sum()) + 31) // 32) * 32)
        ids = t.argsort(-score.detach(), stable=True)[:count]
        alive = t.zeros_like(score).scatter(0, ids, t.ones(count, dtype=score.dtype))
        self.slot += 1
        return ids, alive

    def step(self, layer, state):
        t, f = self.t, self.env
        self.call_layer, self.slot = layer, 0
        state = self.h('state_in', layer, state, {})
        m, C, p, lam, star, b, previous_alive, radial, hstate, Q, F, core, stops, previous = state
        w, w64 = self.w[layer], self.ws[layer]
        mu = self.h('mu', layer, w64.T @ m, {})
        pp, U, F, core, old_columns = self.shared['push'](
            t, layer, w, p, lam, previous, Q, F, core, stops, rank=TIER_RANK)
        bpre = w64.T @ b
        if layer == 15:
            pushed = C.to(t.float32) @ w
            vv = (w64 * pushed.to(t.float64)).sum(0)
            vv = self.h('vv15', layer, vv, {})
            self.variance(layer, vv)
            p64 = pp.to(t.float64)
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
        U_ = lambda name: self.h('cpre_use', layer, Cpre, dict(use=name))   # consumer-split leaves (identity by default)
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
            plane = self.shared['read'](t, pp, lam, response_indices, indices, U, F, old_columns).to(t.float64)
            cp = cp.index_put((response_indices[:, None], indices[None, :]), plane)
            p64 = pp.to(t.float64)
            cpdiag = ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)
            cp = self.diagonal(cp, cpdiag)
        sp = t.zeros_like(Cpre) if layer == 0 else f['_flms_star'](w, star)
        k21 = cp + sp + vv[:, None] * bpre[None, :] + 2 * U_('k21b') * bpre[:, None]
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
            # Exact parent omitted_radial_source algebra, with current host.
            aa = (aux['gate'] - host * aux['pdf'] / aux['sigma']) * omitted
            beta = 2 * (aux['gaussian_mean'] - host * aux['gate']) * alive
            energy = (p64 * p64).T @ aa
            column = p64.T @ beta
            extra = 6 * ((lam.to(t.float64) * energy * column).sum() - (cpdiag * aa * beta).sum())
            new_radial = new_radial + extra / (t.trace(C) ** 2 + 2 * (C * C).sum())
        if 8 <= layer <= 14:
            context={'left':self.edge['context_left'],'right':self.edge['context_right']}
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
        if self.knobs['bfix_gamma']:   # gain b-fix, partition mode, mean m (model_bank_bfix.py, S2): same expression
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
        p = t.cat((alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * pp),
                   positive[:, selected], negative[:, selected]), dim=1)
        lam = t.cat((lam, coefficient[selected].to(t.float32), -coefficient[selected].to(t.float32)))
        if old_columns:
            Q = alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * U)
        stops = stops + (int(lam.numel()),)
        rest = target - cp_post - trace_post
        star = (3 * rest.T - 2 * t.diag(t.diagonal(rest))).to(t.float32)
        out = (m, C, p, lam, star, b, alive, new_radial, hs, Q, F, core, stops, indices)
        return self.h('state_out', layer, out, ctx)

    def run_states(self, start=0, state=None, stop=16):
        """run layers start..stop-1 from `state` (initial if None); returns (outputs list of m per layer, final state)."""
        state = self.initial() if state is None else state
        outs = []
        for layer in range(start, stop):
            state = self.step(layer, state)
            outs.append(state[0])
        return outs, state


def make(rank, p=None, a=None, c=None, e=None, w=None):
    if p is None:
        p, a, c, e = model.torch_payload(torch, np, False)
    w = weights(rank) if w is None else w
    return HookCard(torch, w, p, a, c, e)


def final_mse(pred15, truth):
    x = pred15.clamp_min(0).to(torch.float32).numpy().astype(np.float64)
    return float(((x - truth[15]) ** 2).mean())


def parity(rank=35):
    p, a, c, e = model.torch_payload(torch, np, False)
    w = weights(rank)
    with torch.no_grad():
        ref = model.Card(torch, w, p, a, c, e).run(False)
        card = HookCard(torch, w, p, a, c, e)
        outs, _ = card.run_states()
        mine = torch.stack(outs).clamp_min(0).to(torch.float32)
    return float((ref - mine).abs().max())


if __name__ == '__main__':
    torch.set_num_threads(3)
    print('max abs diff vs model.Card (all layers):', parity(int(sys.argv[1]) if len(sys.argv) > 1 else 35))
