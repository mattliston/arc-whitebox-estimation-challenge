#!/usr/bin/env python3
"""TAG opus-physicist T60: PER-UNIT DUMP OF THE S42 TORCH CARD for the open-loop search of new inputs of the learned mean correction (lead 10-02 19:36Z and 10-10 02:11Z).
The card is the trainer's S42 card exactly as he judged it (request of a_0hS41_u2_r0287: parameter file c3q4zd6_optimum.pt = first dials, f4c 1, ss 0, hf 2, law module n6y with
the table (dfit 2), k31ly instances for ages 1 / 2 / 3 with strengths 0 / 1 / 0, head column 6 at the step-2 weights), built and run by the trainer's own engine (otr_engine.run_network,
forward only) with a frozen copy of his card module (otr_frozen_f2dfccba/otr_m.py; the head weights padded with zero rows to the module's 21 columns, which changes nothing).
Hooks added on top of the card (they read, they never change a number; the run's final-layer MSE is printed beside the trainer's stored one for the parity):
  per layer 1-14, from what the pair map receives: mean mu, variance v, skew k3, kurtosis k4 (as the mean formula gets them), the gate, the response rows;
  per layer 1-15, the head's inputs: the 30 raw features x, the 96 message columns (formed from the previous state as the head forms them), the 21 head columns of the module
  (column 6 = the S42 input), the previous and the new state (64 each), the learned correction delta, the post mean m;
  CANDIDATES (layer l, before the head runs; per unit):
    ls      log sigma_i
    qb, qk  the bank plane's / the whole K21 plane's row sum against c_j = (gate_j / sigma_j)^2 mu_j, over ((c . mu) v_i)      (the row-gain law's third input and its whole-plane twin)
    kmu, kg the K21 plane's row sums against mu and against the gate, over v_i sigma_i
    by pair AGE a = 1, 2, 3, 4, 5-6, 7+ (cohorts of the bank as the card holds them, old tier included):
      q_a   the local quartic (K4L's hub terms) per unit, raw, over v^2          f_a   the diagonal of the fourth-chaos plane of those pairs (k31l's chaos4 form) over v^2
      c_a   sum_r lam_r pp_ir^3 over the cohort's columns, over v sigma       e_a   sum_r |lam_r| pp_ir^2 over v        s_a   sum_r lam_r pp_ir^2 over v
    (W o W) pushes into layer l of per-unit numbers of layer l-1: post variance pv, post skew pk3, post kurtosis pk4 (as the pair map of l-1 hands them on), the kink density d,
      the learned correction squared dd, each sum_a W_ai^2 u_a, over v_i (pv, dd) or as it is
  layer 15: the same where it exists (no plane at 15: qb, qk, kmu, kg absent).
-> s42dump_rank<rank>.npz ; usage: t60_s42_dump.py <rank> [threads = 4]"""
import hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
rank = int(sys.argv[1]); threads = str(int(sys.argv[2])) if len(sys.argv) > 2 else '4'; CERT = len(sys.argv) > 3 and sys.argv[3].startswith('cert'); CL = int(sys.argv[3][4:]) if CERT and len(sys.argv[3]) > 4 else 9      # cert<l>: + pbase so2m 1e-4 at layer l only (the host's change there certifies b6cols' m2)
FROZEN = HERE / 'otr_frozen_f2dfccba'; SHA = 'f2dfccbadf7d90d727d9283cf6454bddcb5ef8912ebe020f7b3bf2c9e1a1cb6e'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]
sys.path.insert(0, str(O / 'trainer')); sys.path.insert(0, str(FROZEN))
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == FROZEN and hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest() == SHA, E.M.__file__
from otr_common import load_params, load_pl
torch = E.torch; t = torch; f64 = t.float64; n = 1024; t0 = time.time()
REQ = json.loads((O / 'trainer/run_M/claimed/a_0hS41_u2_r0287.w0.json').read_text())
E.set_parameters(*load_params(torch, REQ['params'])); E.PL.set_values({f: [float(x) for x in v] for f, v in load_pl(torch, REQ['params']).items()})
E.STAR_ATTACH = False; E.KEEP_FINAL = True; E.STAR_SS = REQ['ss']; E.M.F4C = bool(REQ['f4c']); E.M.NCAR = REQ['ncar']; E.M.HEADF = REQ['hf']
if hasattr(E.M, 'RG'): E.M.RG = False
hu = np.array(json.loads(REQ['hu']) if isinstance(REQ['hu'], str) else REQ['hu'], dtype=np.float64); HU = np.zeros((len(E.M.HEAD_COLUMNS), 128)); HU[:hu.shape[0]] = hu
E.M.head_weights(HU.tolist())
if CERT:
    E.M.MODULES['pbase'] = ('pbase_frozen_s4a', dict({k: 0.0 for k in ('so2m', 'so2c', 'so2t', 'so2f', 'bsk', 'gsk', 'gk4', 'gbk', 'x1', 'x2')}, so2k=1, so2p=123456, so2td=0, lo=1, hi=14))
    E.M.NCAR = list(REQ['ncar']) + [{'module': 'pbase', 'so2m': 1e-4, 'lo': CL, 'hi': CL}]
D = {}; ST = {}
cell = lambda fn, name: fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents
BANDS = (('1', 1, 1), ('2', 2, 2), ('3', 3, 3), ('4', 4, 4), ('56', 5, 6), ('7', 7, 99))


def per_age(l, side, v, mu):
    """per-unit sums over the bank's cohorts by pair age, from the columns pushed into layer l"""
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
    """(W o W) pushes into layer l of per-unit numbers the layer l-1 handed on"""
    prev = ST.get(l - 1)
    if prev is None: return {}
    W2 = cd.ws[l].to(f64) ** 2
    return {'pv': (W2.T @ prev['pv']) / v, 'pk3': W2.T @ prev['pk3'], 'pk4': W2.T @ prev['pk4'], 'pd': W2.T @ prev['d'], 'pdd': (W2.T @ prev['dd']) / v}


def hooks(cd):
    side = cd.m_side
    pin0 = cd.hooks['pair_in']

    def pin_hook(l, pin, ctx):
        pin = pin0(l, pin, ctx)
        if l < 1: return pin
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
        vv, mu = ctx['vv'].detach().to(f64), ctx['mu'].detach().to(f64)
        rec = dict(mu=mu, v=vv, k3=k3.detach().to(f64).clone(), ls=vv.clamp_min(1e-12).sqrt().log())
        rec.update(per_age(15, side, vv.clamp_min(1e-12), mu)); rec.update(pushes(15, cd, vv.clamp_min(1e-12)))
        D.setdefault(15, {}).update(rec); return k3
    cd.hooks['k3_15'] = k3_15
    k4h0 = cd.hooks.get('k4_15')

    def k4_15(l, k4, ctx):
        k4 = k4 if k4h0 is None else k4h0(l, k4, ctx)
        D.setdefault(15, {})['k4'] = (k4.detach().to(f64) if t.is_tensor(k4) and k4.ndim else float(k4) * t.ones(n, dtype=f64)).clone(); return k4
    cd.hooks['k4_15'] = k4_15

    gnn_m = cd.env['gnn_step']; cols_fn = cell(gnn_m, 'columns') if 'columns' in gnn_m.__code__.co_freevars else None

    def gnn_step(l, x, sigma, W, state, P):
        if 1 <= l <= 15:
            xx = x.detach().to(t.float32); xn = t.clamp((xx - P['centers'][l]) / P['scales'][l], -8, 8)
            if l == 15: xn = xn * t.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=t.float32)[None, :]
            g = xn[:, 5:6]; src = state.detach() @ P['msg_w'].T + P['msg_b']; cm = int(P['msg_w'].shape[0]) // 3
            w = W.detach().to(t.float32); w2 = w * w; w3 = w2 * w
            m1 = w.T @ src[:, :cm]; m2 = w2.T @ src[:, cm:2 * cm]; m3 = w3.T @ src[:, 2 * cm:]
            e1 = w2.T @ (src[:, :cm] ** 2); e2 = (w2 * w2).T @ (src[:, cm:2 * cm] ** 2); e3 = (w3 * w3).T @ (src[:, 2 * cm:] ** 2)
            msgs = t.clamp(t.cat([g * m1, g * m2, g * m3, t.log1p(t.clamp(e1, min=0)), t.log1p(t.clamp(e2, min=0)), t.log1p(t.clamp(e3, min=0))], 1), -8, 8)
            r = D.setdefault(l, {}); r.update(x=xx.to(f64), xn=xn.to(f64), msgs=msgs.to(f64), state_in=state.detach().to(f64))
            if cols_fn is not None: r['cols'] = cols_fn(l).to(f64)
        out = gnn_m(l, x, sigma, W, state, P)
        if 1 <= l <= 15: D[l].update(delta=out[0].detach().to(f64), state_out=out[1].detach().to(f64))
        return out
    cd.env['gnn_step'] = gnn_step

    out0 = cd.hooks['state_out']

    def state_out(l, out, ctx):
        out = out0(l, out, ctx)
        m = out[0].detach().to(f64); r = D.setdefault(l, {}); r['m'] = m
        if 'host' in ctx: r['host'] = ctx['host'].detach().to(f64)
        if l <= 14:
            aux = ctx['aux']; Cp = out[1].detach().to(f64); dl = ctx.get('delta')
            z_ = t.zeros(n, dtype=f64); pk3_ = aux.get('post_k3'); pk4_ = aux.get('post_k4')
            ST[l] = dict(pv=t.diagonal(Cp).clone(), pk3=(pk3_.detach().to(f64) if t.is_tensor(pk3_) else z_), pk4=(pk4_.detach().to(f64) * t.ones(n, dtype=f64) if t.is_tensor(pk4_) else z_),
                         d=(aux['pdf'] / aux['sigma']).detach().to(f64), dd=(dl.detach().to(f64) ** 2 if dl is not None else t.zeros(n, dtype=f64)))
        return out
    cd.hooks['state_out'] = state_out


install0 = E.M.install


sys.path.append(str(HERE)); import b6cols as B6      # appended: nothing of the card resolves to this lane


def install(cd):
    install0(cd); hooks(cd); B6.install(cd, torch)
E.M.install = install
B6CAP = {}
_inst2 = E.M.install


def install_cap(cd):
    _inst2(cd); B6CAP.clear(); cd.b6cols = B6CAP
    pin_b6 = cd.hooks['pair_in']
E.M.install = install_cap
rec = E.run_network(rank, 'eval')
ref = O / f'trainer/run_M/results/a_0hS41_u2_r{rank:04d}.json'
refv = json.loads(ref.read_text())['mse'] * 1e9 if ref.exists() else None
out = {'final_x1e9': np.array(rec['mse'] * 1e9), 'per_layer_x1e9': np.array(rec['mse_layers']) * 1e9}
for l, r in B6CAP.items():
    for k, x in r.items(): out[f'b6_{k}_{l}'] = x.cpu().numpy().astype(np.float64)
for l, r in D.items():
    for k, x in r.items(): out[f'{k}_{l}'] = x.cpu().numpy().astype(np.float32 if x.ndim == 2 else np.float64)
np.savez(HERE / (f's42dump_rank{rank:04d}.npz' if not CERT else f's42dump_rank{rank:04d}_cert{CL}.npz'), **out)
print(json.dumps(dict(TAG='physicist T60 S42 dump', rank=rank, final_x1e9=round(rec['mse'] * 1e9, 6), trainer_stored_x1e9=refv, layers=sorted(D), keys=sorted({k for r in D.values() for k in r}), wall=round(time.time() - t0, 1))), flush=True)
