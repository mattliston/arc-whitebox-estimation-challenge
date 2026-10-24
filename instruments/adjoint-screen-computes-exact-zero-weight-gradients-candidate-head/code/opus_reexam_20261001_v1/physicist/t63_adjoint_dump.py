#!/usr/bin/env python3
"""TAG opus-physicist T63: ADJOINT DUMP OF THE S42 TORCH CARD = the exact zero-weight gradient of ANY new head-input column, for the open-loop screen of head inputs
(lead 10-10 02:11Z / 02:13Z). Same card, request and engine as T60 (the trainer's S42 card: his request a_0hS41_u2_r0287, frozen otr_m f2dfccba), run in the trainer's
GRADIENT mode (forward with layer checkpoints + backward of 1e8 x the final-layer MSE against the 1e9-sample truth, exactly his zero-weight runs).
What is added: at every layer 1-15 a zero per-unit bias leaf Z_l (1024 x 128, float32) on the first layer of the learned mean correction, at the place where the head's
new columns enter (enc0_b + z @ HEADU + Z_l; with Z = 0 nothing changes). After the backward, G_l = dL/dZ_l. A new column x (per layer, standardised as the head
would get it) with first-layer weights u at zero has the exact gradient dL/du = sum_l x_l' G_l: G is the whole zero-weight test for every candidate at once.
CERTIFICATES printed: sum_l z_l' G_l against the trainer's own HEADU gradient (z = the module's 21 columns, recorded in the forward pass), and sum_l H_l' G_l against the
gradient of the head's existing first-layer weights (H = its 126 inputs).
Also dumped (forward pass only; the replay of a checkpointed layer is skipped, since the side tables then hold another layer's state): everything T60 dumps, the batch-6
columns (b6cols.py), and the top-8 modes of the post covariance each layer hands on (eigenvalues; loadings (W_l' u_k)_i, sign fixed by sum(u_k) > 0).
-> adump_rank<rank>.npz ; usage: t63_adjoint_dump.py <rank> [threads = 4]"""
import hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
rank = int(sys.argv[1]); threads = str(int(sys.argv[2])) if len(sys.argv) > 2 else '4'
FROZEN = HERE / 'otr_frozen_f2dfccba'; SHA = 'f2dfccbadf7d90d727d9283cf6454bddcb5ef8912ebe020f7b3bf2c9e1a1cb6e'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]
sys.path.insert(0, str(O / 'trainer')); sys.path.insert(0, str(FROZEN))
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == FROZEN and hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest() == SHA, E.M.__file__
from otr_common import load_params, load_pl
torch = E.torch; t = torch; f64 = t.float64; n = 1024; t0 = time.time(); SCALE = 1e8; NMODE = 8
REQ = json.loads((O / 'trainer/run_M/claimed/a_0hS41_u2_r0287.w0.json').read_text())
E.set_parameters(*load_params(torch, REQ['params'])); E.PL.set_values({f: [float(x) for x in v] for f, v in load_pl(torch, REQ['params']).items()})
E.STAR_ATTACH = False; E.KEEP_FINAL = False; E.STAR_SS = REQ['ss']; E.M.F4C = bool(REQ['f4c']); E.M.NCAR = REQ['ncar']; E.M.HEADF = REQ['hf']
if hasattr(E.M, 'RG'): E.M.RG = False
hu = np.array(json.loads(REQ['hu']) if isinstance(REQ['hu'], str) else REQ['hu'], dtype=np.float64); HU = np.zeros((len(E.M.HEAD_COLUMNS), 128)); HU[:hu.shape[0]] = hu
E.M.head_weights(HU.tolist())
D = {}; ST = {}; ZL = {}; MODES = {}
cell = lambda fn, name: fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents
BANDS = (('1', 1, 1), ('2', 2, 2), ('3', 3, 3), ('4', 4, 4), ('56', 5, 6), ('7', 7, 99))


def per_age(l, side, v, mu):
    """per-unit sums over the bank's cohorts by pair age, from the columns pushed into layer l (as T60)"""
    pp = side['pp'].detach().to(f64); lam = side['lam_in'].detach().to(f64)[:pp.shape[1]]; starts = side['starts']; births = side['births']; s = v.sqrt(); out = {}
    for nm, a0, a1 in BANDS:
        q = t.zeros(n, dtype=f64); f = t.zeros(n, dtype=f64); c = t.zeros(n, dtype=f64); e = t.zeros(n, dtype=f64); sg = t.zeros(n, dtype=f64)
        for j in range(l):
            if not a0 <= l - j <= a1 or j + 1 >= len(starts): continue
            k0, k1 = starts[j], starts[j + 1]
            if k1 <= k0: continue
            P_ = pp[:, k0:k1]; L_ = lam[k0:k1]; c = c + (P_ ** 3) @ L_; e = e + (P_ ** 2) @ L_.abs(); sg = sg + (P_ ** 2) @ L_
            B = births.get(j); K = (k1 - k0) // 2
            if B is None or not K or len(B['tp']) != K: continue
            gap = (B['tp'] - B['tm']).to(f64)[None, :]; pl, pm = pp[:, k0:k0 + K], pp[:, k0 + K:k0 + 2 * K]
            Ev = (pl - pm) / gap; Y = (B['tp'].to(f64)[None, :] * pm - B['tm'].to(f64)[None, :] * pl) / gap; S1, S2, S3, S4 = (x.to(f64)[None, :] for x in B['S'])
            q = q + (4 * S1 * Y ** 3 * Ev + 6 * S2 * Y ** 2 * Ev ** 2 + 4 * S3 * Y * Ev ** 3 + S4 * Ev ** 4).sum(1)
            AY = 3 * S1 * Y * Y * Ev + 3 * S2 * Y * Ev * Ev + S3 * Ev ** 3; AE = S1 * Y ** 3 + 3 * S2 * Y * Y * Ev + 3 * S3 * Y * Ev * Ev + S4 * Ev ** 3
            mY2, mYE, mE2 = (Y * Y).mean(0)[None, :], (Y * Ev).mean(0)[None, :], (Ev * Ev).mean(0)[None, :]
            AY = AY - 3 * (S1 * (mY2 * Ev + 2 * mYE * Y) + S2 * (mE2 * Y + 2 * mYE * Ev) + S3 * mE2 * Ev)
            AE = AE - 3 * (S1 * mY2 * Y + S2 * (mY2 * Ev + 2 * mYE * Y) + S3 * (mE2 * Y + 2 * mYE * Ev) + S4 * mE2 * Ev)
            f = f + (AY * Y + AE * Ev).sum(1)
        out.update({f'q{nm}': q / v ** 2, f'f{nm}': f / v ** 2, f'c{nm}': c / (v * s), f'e{nm}': e / v, f's{nm}': sg / v})
    return out


def pushes(l, cd, v):
    """(W o W) pushes into layer l of per-unit numbers the layer l-1 handed on (as T60), and the loadings of the previous layer's top post-covariance modes"""
    prev = ST.get(l - 1)
    if prev is None: return {}
    W = cd.ws[l].to(f64); W2 = W ** 2
    out = {'pv': (W2.T @ prev['pv']) / v, 'pk3': W2.T @ prev['pk3'], 'pk4': W2.T @ prev['pk4'], 'pd': W2.T @ prev['d'], 'pdd': (W2.T @ prev['dd']) / v}
    if (l - 1) in MODES:
        lam, U = MODES[l - 1]; L = W.T @ U      # (n, NMODE): loading of unit i on mode k
        out['mload'] = L / v.sqrt()[:, None]; out['mshare'] = (L * L) * lam[None, :] / v[:, None]
    return out


def hooks(cd):
    side = cd.m_side
    pin0 = cd.hooks['pair_in']

    def pin_hook(l, pin, ctx):
        pin = pin0(l, pin, ctx)
        if l < 1 or side.get('again'): return pin
        mu, C, K21, k4 = pin[0].detach().to(f64), pin[1].detach().to(f64), pin[2].detach().to(f64), pin[3]
        v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); k4v = (k4.detach().to(f64) if t.is_tensor(k4) and k4.ndim else float(k4) * t.ones(n, dtype=f64))
        g = ctx['gate'].detach().to(f64); eye = t.eye(n, dtype=t.bool); Ko = K21.masked_fill(eye, 0.); cp = ctx['cp'].detach().to(f64).masked_fill(eye, 0.)
        c = (g / s) ** 2 * mu; den = (c * mu).sum() - c * mu
        rec = dict(mu=mu, v=v, k3=t.diagonal(K21).clone(), k4=k4v, gate=g, resp=ctx['response_mask'].detach().to(f64), ls=s.log(),
                   qb=((cp @ c) / den) / v, qk=((Ko @ c) / den) / v, kmu=(Ko @ mu) / (v * s), kg=(Ko @ g) / (v * s))
        rec.update(per_age(l, side, v, mu)); rec.update(pushes(l, cd, v))
        D.setdefault(l, {}).update(rec)
        return pin
    cd.hooks['pair_in'] = pin_hook

    k3h0 = cd.hooks.get('k3_15')

    def k3_15(l, k3, ctx):
        k3 = k3 if k3h0 is None else k3h0(l, k3, ctx)
        if side.get('again'): return k3
        vv, mu = ctx['vv'].detach().to(f64), ctx['mu'].detach().to(f64)
        rec = dict(mu=mu, v=vv, k3=k3.detach().to(f64).clone(), ls=vv.clamp_min(1e-12).sqrt().log())
        rec.update(per_age(15, side, vv.clamp_min(1e-12), mu)); rec.update(pushes(15, cd, vv.clamp_min(1e-12)))
        D.setdefault(15, {}).update(rec); return k3
    cd.hooks['k3_15'] = k3_15
    k4h0 = cd.hooks.get('k4_15')

    def k4_15(l, k4, ctx):
        k4 = k4 if k4h0 is None else k4h0(l, k4, ctx)
        if side.get('again'): return k4
        D.setdefault(15, {})['k4'] = (k4.detach().to(f64) if t.is_tensor(k4) and k4.ndim else float(k4) * t.ones(n, dtype=f64)).clone(); return k4
    cd.hooks['k4_15'] = k4_15

    gnn_m = cd.env['gnn_step']; cols_fn = cell(gnn_m, 'columns') if 'columns' in gnn_m.__code__.co_freevars else None

    def gnn_step(l, x, sigma, W, state, P):
        if 1 <= l <= 15 and not side.get('again'):
            with t.no_grad():
                xx = x.detach().to(t.float32); xn = t.clamp((xx - P['centers'][l]) / P['scales'][l], -8, 8)
                if l == 15: xn = xn * t.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=t.float32)[None, :]
                g = xn[:, 5:6]; src = state.detach() @ P['msg_w'].T + P['msg_b']; cm = int(P['msg_w'].shape[0]) // 3
                w = W.detach().to(t.float32); w2 = w * w; w3 = w2 * w
                m1 = w.T @ src[:, :cm]; m2 = w2.T @ src[:, cm:2 * cm]; m3 = w3.T @ src[:, 2 * cm:]
                e1 = w2.T @ (src[:, :cm] ** 2); e2 = (w2 * w2).T @ (src[:, cm:2 * cm] ** 2); e3 = (w3 * w3).T @ (src[:, 2 * cm:] ** 2)
                msgs = t.clamp(t.cat([g * m1, g * m2, g * m3, t.log1p(t.clamp(e1, min=0)), t.log1p(t.clamp(e2, min=0)), t.log1p(t.clamp(e3, min=0))], 1), -8, 8)
                r = D.setdefault(l, {}); r.update(x=xx.to(f64), xn=xn.to(f64), msgs=msgs.to(f64), state_in=state.detach().to(f64), sigma=sigma.detach().to(f64))
                if cols_fn is not None: r['cols'] = cols_fn(l).to(f64)
        out = gnn_m(l, x, sigma, W, state, P)
        if 1 <= l <= 15 and not side.get('again'): D[l].update(delta=out[0].detach().to(f64), state_out=out[1].detach().to(f64))
        return out
    cd.env['gnn_step'] = gnn_step

    out0 = cd.hooks['state_out']

    def state_out(l, out, ctx):
        out = out0(l, out, ctx)
        if side.get('again'): return out
        with t.no_grad():
            m = out[0].detach().to(f64); r = D.setdefault(l, {}); r['m'] = m
            if 'host' in ctx: r['host'] = ctx['host'].detach().to(f64)
            if l <= 14:
                aux = ctx['aux']; Cp = out[1].detach().to(f64); dl = ctx.get('delta')
                z_ = t.zeros(n, dtype=f64); pk3_ = aux.get('post_k3'); pk4_ = aux.get('post_k4')
                ST[l] = dict(pv=t.diagonal(Cp).clone(), pk3=(pk3_.detach().to(f64) if t.is_tensor(pk3_) else z_), pk4=(pk4_.detach().to(f64) * t.ones(n, dtype=f64) if t.is_tensor(pk4_) else z_),
                             d=(aux['pdf'] / aux['sigma']).detach().to(f64), dd=(dl.detach().to(f64) ** 2 if dl is not None else t.zeros(n, dtype=f64)))
                ev, U = t.linalg.eigh(0.5 * (Cp + Cp.T)); ev, U = ev.flip(0)[:NMODE], U.flip(1)[:, :NMODE]
                U = U * t.where(U.sum(0) < 0, -1.0, 1.0).to(f64)[None, :]
                MODES[l] = (ev, U); r['post_modes'] = ev; r['post_trace'] = t.trace(Cp).reshape(1)
        return out
    cd.hooks['state_out'] = state_out


sys.path.append(str(HERE)); import b6cols as B6      # appended: nothing of the card resolves to this lane
install0 = E.M.install
B6CAP = {}


def install(cd):
    base = cd.env['gnn_step']

    def gnn_zl(l, x, sigma, W, state, P):      # the zero per-unit bias leaf, INSIDE the module's head-column wrapper (it hands enc0_b on as n x 128)
        if 1 <= l <= 15:
            if l not in ZL: ZL[l] = t.zeros((n, 128), dtype=t.float32, requires_grad=True)
            P = dict(P, enc0_b=P['enc0_b'] + ZL[l])
        return base(l, x, sigma, W, state, P)
    cd.env['gnn_step'] = gnn_zl
    install0(cd); hooks(cd); B6.install(cd, torch)
    side = cd.m_side; pin_b6, k3_b6, k4_b6 = cd.hooks['pair_in'], cd.hooks['k3_15'], cd.hooks['k4_15']
    B6CAP.clear(); cd.b6cols = B6CAP
    # batch-6 hooks only in the forward pass (in a replay the side tables hold another layer's state); the module's own hooks always run
    pin_mod = cell(cell(pin_b6, 'pin0'), 'pin0')
    cd.hooks['pair_in'] = lambda l, pin, ctx: pin_mod(l, pin, ctx) if side.get('again') else pin_b6(l, pin, ctx)
    k3_mod = cell(cell(k3_b6, 'k3h0'), 'k3h0'); k4_mod = cell(cell(k4_b6, 'k4h0'), 'k4h0')
    cd.hooks['k3_15'] = lambda l, k3, ctx: (k3 if k3_mod is None else k3_mod(l, k3, ctx)) if side.get('again') else k3_b6(l, k3, ctx)
    cd.hooks['k4_15'] = lambda l, k4, ctx: (k4 if k4_mod is None else k4_mod(l, k4, ctx)) if side.get('again') else k4_b6(l, k4, ctx)
E.M.install = install
rec = E.run_network(rank, 'grad', SCALE)
G = {l: (z.grad.detach().clone() if z.grad is not None else t.zeros((n, 128))) for l, z in ZL.items()}
hug = rec['hu_grads'].double(); ew = rec['grads']['p.enc0_w'].double()
cert_hu = sum(D[l]['cols'].T @ G[l].double() for l in sorted(G) if 'cols' in D.get(l, {}))
cert_w = sum(G[l].double().T @ t.cat([D[l]['xn'], D[l]['msgs']], 1) for l in sorted(G))
rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))
out = {'final_x1e9': np.array(rec['mse'] * 1e9), 'per_layer_x1e9': np.array(rec['mse_layers']) * 1e9, 'scale': np.array(SCALE),
       'hu_grads': hug.numpy(), 'enc0_w_grad': ew.numpy(), 'cert_hu': np.array(rel(cert_hu, hug)), 'cert_w': np.array(rel(cert_w, ew))}
for l, g in G.items(): out[f'G_{l}'] = g.numpy().astype(np.float32)
for l, r in B6CAP.items():
    for k, x in r.items(): out[f'b6_{k}_{l}'] = x.cpu().numpy().astype(np.float64)
for l, r in D.items():
    for k, x in r.items(): out[f'{k}_{l}'] = x.cpu().numpy().astype(np.float32 if x.ndim == 2 and k not in ('mload', 'mshare') else np.float64)
np.savez(HERE / f'adump_rank{rank:04d}.npz', **out)
ref = O / f'trainer/run_M/results/a_0hS41_u2_r{rank:04d}.json'
refv = json.loads(ref.read_text())['mse'] * 1e9 if ref.exists() else None
print(json.dumps(dict(TAG='physicist T63 adjoint dump', rank=rank, final_x1e9=round(rec['mse'] * 1e9, 6), trainer_eval_x1e9=refv, cert_hu_rel=rel(cert_hu, hug), cert_w_rel=rel(cert_w, ew),
                      hu_grad_norm=float(hug.norm()), G_layers=sorted(G), seconds=round(rec['seconds'], 1), wall=round(time.time() - t0, 1))), flush=True)
