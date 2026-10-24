#!/usr/bin/env python3
"""TAG opus-physicist T71: V15 DUMP (lead 10-10 04:40Z): what in the card's post covariance of layer 14 (13) makes its per-unit VARIANCE error at layer 15 (14)?
v_(l+1) = diag(W' Cpost_l W), so the variance error is diag(W' dC W). The S42 torch card (as T60 / T66: trainer's request a_0hS41_u2_r0287, frozen otr_m f2dfccba), eval mode,
with a wrapper on the pair map (_nonlinear_pair_batch) at layers 13 and 14 that splits its post covariance (channel 0 of the Hermite series, times s_i s_j, off the diagonal)
into the terms the card computes, plus terms it drops, each pushed through the next layer's weights as p_X = diag(W' X W) (per unit of the next layer):
  gauss   the Gaussian part, Mehler orders r = 1..6            trunc  the Gaussian orders r = 7..12 the card drops (A_r = Hermite coefficients, card's derivatives())
  e21     the K21 term (T21 tri(2,1) + T21' tri(1,2)) / 2        e22    the K22 term / 4
  e31k    the K31 term from t4 rho / 6                            e31r   the K31 remainder the modules add (the law's / cross form's remainder31) / 6
  e3      the per-unit skew terms / 6                             e4     the per-unit kurtosis terms / 24
  r21     the K21 term's next order in rho (rho^3 / 6), dropped  r34    the skew / kurtosis terms' next order in rho, dropped
  s2121   second order T21 o T21' F3 G3 / 4, dropped             s33    second order rho t3_i t3_j F4 G4 / 36, dropped
  diag    the post variances on the diagonal                       after  what the card changes in the post covariance after the pair map (handed-on C minus the map's)
  edge    the learned edge's correction (layers 8-14)               headadj  after minus edge = the head's mean correction moved into C (- host d' - d host' - d d')
Certificates: the terms add up to the map's covariance (off the diagonal); the sum of all pushes = the card's variance at the next layer.
Also dumped: the card's post variances pv_13 / pv_14 (diagonal of the handed-on C), its mu / v at layers 14 and 15, the head's 126 inputs at 14 and 15 (for regressions after the head's own inputs), the kept rows at 15.
-> v15dump_rank<rank>.npz (version 3: + edge / headadj) ; usage: t71_v15_dump.py <rank,rank,...> [threads = 4]   (self-gated between networks like T66; stop: physicist/STOP_T71)"""
import hashlib, json, math, os, sys, time, subprocess, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
RANKS = [int(x) for x in sys.argv[1].split(',')]; threads = str(int(sys.argv[2])) if len(sys.argv) > 2 else '4'
FROZEN = HERE / 'otr_frozen_f2dfccba'; SHA = 'f2dfccbadf7d90d727d9283cf6454bddcb5ef8912ebe020f7b3bf2c9e1a1cb6e'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]
sys.path.insert(0, str(O / 'trainer')); sys.path.insert(0, str(FROZEN))
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == FROZEN and hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest() == SHA, E.M.__file__
from otr_common import load_params, load_pl
torch = E.torch; t = torch; f64 = t.float64; n = 1024
REQ = json.loads((O / 'trainer/run_M/claimed/a_0hS41_u2_r0287.w0.json').read_text())
E.set_parameters(*load_params(torch, REQ['params'])); E.PL.set_values({f: [float(x) for x in v] for f, v in load_pl(torch, REQ['params']).items()})
E.STAR_ATTACH = False; E.KEEP_FINAL = True; E.STAR_SS = REQ['ss']; E.M.F4C = bool(REQ['f4c']); E.M.NCAR = REQ['ncar']; E.M.HEADF = REQ['hf']
if hasattr(E.M, 'RG'): E.M.RG = False
hu = np.array(json.loads(REQ['hu']) if isinstance(REQ['hu'], str) else REQ['hu'], dtype=np.float64); HU = np.zeros((len(E.M.HEAD_COLUMNS), 128)); HU[:hu.shape[0]] = hu
E.M.head_weights(HU.tolist())
R = {}; X = {}; INFO = {}
LAYERS = (13, 14)


def terms(env, args, kwargs, cov_card):
    """the card's pair-map covariance split into terms (n x n, times s_i s_j, diagonal 0), recomputed from the map's own inputs"""
    mu, C, K21, k4 = args[:4]; k22o = args[4] if len(args) > 4 else kwargs.get('k22_override'); rem = args[6] if len(args) > 6 else kwargs.get('remainder31')
    mu, C, K21 = mu.detach().to(f64), C.detach().to(f64), K21.detach().to(f64)
    k4 = (k4.detach().to(f64) if t.is_tensor(k4) else t.tensor(float(k4), dtype=f64)) * t.ones(n, dtype=f64)
    v = t.diagonal(C).clone(); s = v.sqrt(); a = mu / s; pdf = t.exp(-a * a / 2) / math.sqrt(2 * math.pi); g = t.erfc(-a / math.sqrt(2)) / 2; mg = s * (pdf + a * g)
    eye = t.eye(n, dtype=t.bool)
    rho = (C / t.outer(s, s)).masked_fill(eye, 0.)
    T21 = (K21 / (v[:, None] * s[None, :])).masked_fill(eye, 0.)
    K22 = k4 / 3 if k22o is None else (k22o.detach().to(f64) if t.is_tensor(k22o) else t.tensor(float(k22o), dtype=f64))
    T22 = (K22[None, :] if K22.ndim == 1 else K22) / t.outer(v, v)      # numpy broadcasting of the source: a per-unit K22 runs along the columns
    t3 = t.diagonal(K21) / s ** 3; t4 = k4 / v ** 2
    T31k = t4[:, None] * rho; T31r = rem.detach().to(f64) if (rem is not None and t.is_tensor(rem)) else t.zeros(n, n, dtype=f64)
    A = [x.to(f64) for x in env['derivatives'](a.to(t.float64), pdf, g, mg / s, 12)[0]]      # channel 0 (ReLU) Hermite coefficients, orders 0..12
    f = lambda r: A[r][:, None]; gg = lambda r: A[r][None, :]
    P = [rho]
    for k in range(2, 13): P.append(P[-1] * (rho / float(k)))     # P[r-1] = rho^r / r!
    gauss = sum(P[r - 1] * (f(r) * gg(r)) for r in range(1, 7)); trunc = sum(P[r - 1] * (f(r) * gg(r)) for r in range(7, 13))
    tri = lambda fo, go, Rw: sum((f(fo + j) * gg(go + j)) * (1.0 if w is None else w) for j, w in enumerate(Rw))
    RR = (None, P[0], P[1]); P12 = (P[0], P[1])
    e21 = (T21 * tri(2, 1, RR) + T21.T * tri(1, 2, RR)) / 2
    e22 = T22 * tri(2, 2, RR) / 4
    e31k = (T31k * tri(3, 1, RR) + T31k.T * tri(1, 3, RR)) / 6; e31r = (T31r * tri(3, 1, RR) + T31r.T * tri(1, 3, RR)) / 6
    e3 = (t3[:, None] * tri(4, 1, P12) + t3[None, :] * tri(1, 4, P12)) / 6
    e4 = (t4[:, None] * tri(5, 1, P12) + t4[None, :] * tri(1, 5, P12)) / 24
    r21 = (T21 * (P[2] * f(5) * gg(4)) + T21.T * (P[2] * f(4) * gg(5))) / 2
    r34 = (t3[:, None] * P[2] * f(7) * gg(3) + t3[None, :] * P[2] * f(3) * gg(7)) / 6 + (t4[:, None] * P[2] * f(8) * gg(3) + t4[None, :] * P[2] * f(3) * gg(8)) / 24
    s2121 = T21 * T21.T * f(3) * gg(3) / 4; s33 = rho * t.outer(t3, t3) * f(4) * gg(4) / 36
    ss = t.outer(s, s); out = {}
    for name, M in (('gauss', gauss), ('trunc', trunc), ('e21', e21), ('e22', e22), ('e31k', e31k), ('e31r', e31r), ('e3', e3), ('e4', e4), ('r21', r21), ('r34', r34),
                    ('s2121', s2121), ('s33', s33)):
        out[name] = (M * ss).masked_fill(eye, 0.)
    card_terms = sum(out[k] for k in ('gauss', 'e21', 'e22', 'e31k', 'e31r', 'e3', 'e4'))
    off = cov_card.detach().to(f64).masked_fill(eye, 0.)
    cert = float((card_terms - off).abs().max() / off.abs().max())
    return out, cert, float(A[2].abs().max())


def hooks(cd):
    side = cd.m_side; env = cd.env
    pmap0 = env['_nonlinear_pair_batch']

    def pmap(*args, **kwargs):
        res = pmap0(*args, **kwargs); l = side.get('layer')
        if l in LAYERS and not side.get('again') and args[1].shape[0] == n:
            with t.no_grad():
                X[l], INFO[f'cert_terms_{l}'], _ = terms(env, args, kwargs, res[1])
                X[l]['diag_var'] = t.diag(t.diagonal(res[1]).detach().to(f64)); X[l]['_cov_map'] = res[1].detach().to(f64)
        return res
    env['_nonlinear_pair_batch'] = pmap
    edge0 = env['_message_edge_source']

    def edge(*args, **kwargs):
        res = edge0(*args, **kwargs); l = side.get('layer')
        if l in LAYERS and l in X and not side.get('again'):
            X[l]['edge'] = res.detach().to(f64).clone()
        return res
    env['_message_edge_source'] = edge

    out0 = cd.hooks['state_out']

    def state_out(l, out, ctx):
        out = out0(l, out, ctx)
        if l in LAYERS and l in X and not side.get('again'):
            with t.no_grad():
                Cp = out[1].detach().to(f64); X[l]['after'] = Cp - X[l].pop('_cov_map'); X[l]['_Cp'] = Cp; X[l]['headadj'] = X[l]['after'] - X[l].get('edge', 0.0) if 'edge' in X[l] else X[l]['after'].clone(); R[f'pv_{l}'] = t.diagonal(Cp).clone()      # the card's post variances (diag of the handed-on C)
        return out
    cd.hooks['state_out'] = state_out

    pin0 = cd.hooks['pair_in']

    def pin_hook(l, pin, ctx):
        pin = pin0(l, pin, ctx)
        if l == 14 and not side.get('again'):
            with t.no_grad():
                C = pin[1].detach().to(f64); R['v_14'] = t.diagonal(C).clone(); R['mu_14'] = pin[0].detach().to(f64)
                push(cd, 13, 14, C)
        return pin
    cd.hooks['pair_in'] = pin_hook
    k3h0 = cd.hooks.get('k3_15')

    def k3_15(l, k3, ctx):
        k3 = k3 if k3h0 is None else k3h0(l, k3, ctx)
        if not side.get('again'):
            with t.no_grad():
                vv = ctx['vv'].detach().to(f64); R['v_15'] = vv; R['mu_15'] = ctx['mu'].detach().to(f64)
                R['kept_15'] = ((R['mu_15'] / vv.clamp_min(1e-12).sqrt()).abs() <= 3.5).to(f64); push(cd, 14, 15, None)
        return k3
    cd.hooks['k3_15'] = k3_15

    gnn_m = cd.env['gnn_step']

    def gnn_step(l, x, sigma, W, state, P):
        if l in (14, 15) and not side.get('again'):
            with t.no_grad():
                xx = x.detach().to(t.float32); xn = t.clamp((xx - P['centers'][l]) / P['scales'][l], -8, 8)
                if l == 15: xn = xn * t.tensor([1] * 13 + [0] * 3 + [1] * 6 + [0] * 7 + [1], dtype=t.float32)[None, :]
                g = xn[:, 5:6]; src = state.detach() @ P['msg_w'].T + P['msg_b']; cm = int(P['msg_w'].shape[0]) // 3
                w = W.detach().to(t.float32); w2 = w * w; w3 = w2 * w
                m1 = w.T @ src[:, :cm]; m2 = w2.T @ src[:, cm:2 * cm]; m3 = w3.T @ src[:, 2 * cm:]
                e1 = w2.T @ (src[:, :cm] ** 2); e2 = (w2 * w2).T @ (src[:, cm:2 * cm] ** 2); e3 = (w3 * w3).T @ (src[:, 2 * cm:] ** 2)
                msgs = t.clamp(t.cat([g * m1, g * m2, g * m3, t.log1p(t.clamp(e1, min=0)), t.log1p(t.clamp(e2, min=0)), t.log1p(t.clamp(e3, min=0))], 1), -8, 8)
                R[f'H_{l}'] = t.cat([xn, msgs], 1).to(f64)
        return gnn_m(l, x, sigma, W, state, P)
    cd.env['gnn_step'] = gnn_step


def push(cd, lsrc, ldst, Cpre):
    """p_X = diag(W' X W) for every term X of layer lsrc, into layer ldst; certificate against the card's v at ldst"""
    if lsrc not in X or '_Cp' not in X[lsrc]: return
    W = cd.ws[ldst].to(f64)
    for name, M in X[lsrc].items():
        if name.startswith('_'): continue
        R[f'p_{name}_{ldst}'] = ((M @ W) * W).sum(0)
    tot = ((X[lsrc]['_Cp'] @ W) * W).sum(0); vcard = R[f'v_{ldst}'] if f'v_{ldst}' in R else (t.diagonal(Cpre) if Cpre is not None else None)
    if vcard is not None: INFO[f'cert_push_{ldst}'] = float(((tot - vcard).abs() / vcard.abs()).max())
    R[f'p_total_{ldst}'] = tot
    del X[lsrc]


install0 = E.M.install


def install(cd):
    install0(cd); hooks(cd)
E.M.install = install


def gate():
    while True:
        if (HERE / 'STOP_T71').exists(): return False
        L = [x for x in (O / 'TEAM_LOG.md').read_text().splitlines() if x.startswith('[lead] ')]
        o = max([i for i, x in enumerate(L) if 'QUIET WINDOW OPEN' in x], default=-1); c = max([i for i, x in enumerate(L) if 'QUIET WINDOW CLOSED' in x], default=-1)
        pr = [x for x in L if 'MACHINE PRIORITY UNTIL' in x]; busy = o > c
        if pr:
            mm = re.search(r'MACHINE PRIORITY UNTIL (\d\d):(\d\d)Z', pr[-1]); now = time.gmtime()
            if mm and now.tm_hour * 60 + now.tm_min < int(mm.group(1)) * 60 + int(mm.group(2)): busy = True
        if not busy and subprocess.run(['bash', str(O / 'idle_gate.sh'), '30'], capture_output=True).returncode == 0: return True
        time.sleep(20)


for rank in RANKS:
    if (HERE / f'v15dump_rank{rank:04d}.npz').exists(): continue
    if not gate(): print(json.dumps(dict(TAG='physicist T71 stop', rank=rank)), flush=True); break
    t0 = time.time(); R.clear(); X.clear(); INFO.clear()
    rec = E.run_network(rank, 'eval')
    out = {k: x.cpu().numpy().astype(np.float64) for k, x in R.items()}; out['final_x1e9'] = np.array(rec['mse'] * 1e9)
    np.savez(HERE / f'v15dump_rank{rank:04d}.npz', **out)
    print(json.dumps(dict(TAG='physicist T71 V15 dump', rank=rank, final_x1e9=round(rec['mse'] * 1e9, 6), info={k: float('%.3g' % v) for k, v in INFO.items()}, keys=len(out),
                          card_s=round(rec['seconds'], 1), wall=round(time.time() - t0, 1))), flush=True)
