#!/usr/bin/env python3
"""TAG opus-reexam E11: a hook-level GAIN-NORMALISED ("frame B") version of the current torch card, physics only.
Frame B (see SESSION_NOTES section 11): after every layer the post state (mean m, covariance C, third-order bank columns p, the
Sym(C x b) carrier b) is re-normalised by the size of the activation vector along its mean, G = m.x/|m|^2, with weights G:
   l = m/|m|^2, cbar = C l, s2 = l.cbar, c' = cbar - m s2, Pi = I - m l'
   C  <- Pi [(1 + s2) C - T.l] Pi' + 2 c' c''       (T.l: third cumulant contracted with l; only its Sym(C x b) part is used)
   b  <- Pi b - 2 c'
   p  <- Pi p (bank columns), Q <- Pi Q (old-tier basis), radial <- 0 (no fourth-order machinery), mean unchanged.
Arms (same parameters, no refit):
   lab_on        the card as deployed                         lab_off       learned mean correction off (layers 1-15)
   B_off         frame B, learned mean correction off, neutral carry dials (bscale = s_carry = s_new = 1)
   radial (fourth-order isotropic coefficient): radial_B = radial_post + 12 var(t) - 12 eps, eps = rigid part of Pi (T.l) Pi on Pi C Pi
   B_off_<opt>   variants: nolegs (birth columns without the fitted leg corrections; valid for lab too), rmc (radial = Monte Carlo mean kurtosis of the next frame-B pre-activation: an oracle of 15 scalars), r0 (radial 0), rkeep (radial_post kept), rigid (T.l replaced by its rigid part), noT (drop T.l), nobank / nostar (drop that part of T.l), lead (leading order only: C <- Pi C Pi, b <- Pi b - 2 c'), noedge (learned covariance edge off),
                 nob (b <- Pi b only), keepdials (trained bscale/s_carry/s_new)
Reported: per-layer MSE of the post mean vs the organizers' 1e9 truth; relative Frobenius error of the carried covariance
(off-diagonal) against Monte Carlo truth (lab: fable_covhunter_v1/mcall 32M; frame B: e10 file, if present).
usage: e11_frameB_card.py <rank> <threads> [arms comma list]  -> e11_rank{rank}.json"""
import os, sys, ast, json, time, inspect, importlib.util
from pathlib import Path
os.environ['JOIN_SCHED'] = '384-320-384'
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; H = P2 / 'codex_resume_20260926_v2'
TR = H / 'hill_climb_rrpi_detached_star_v1'
for rel in ('fable_analyst_v1', 'fable_covhunter_v1', 'fable_engineer_v1', 'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    sys.path.insert(0, str(P2 / rel))
sys.path.insert(0, str(TR))
import numpy as np, torch
from native_parameters import load_s33_arrays
rank = int(sys.argv[1]); threads = int(sys.argv[2])
ARMS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['lab_on', 'lab_off', 'B_off', 'B_off_lead', 'B_off_noT', 'B_off_noedge']
arrays, dials = load_s33_arrays(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
spec = importlib.util.spec_from_file_location('e11_current_card', P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py')
LC = importlib.util.module_from_spec(spec); spec.loader.exec_module(LC)
assert LC.SCHED == '384-320-384'
bd_source = inspect.getsource(LC.JT.make_bd); assert bd_source.count('for j in range(4):') == 1
exec(compile(ast.parse(bd_source.replace('for j in range(4):', 'for j in range(3):')), str(__file__) + ':j3cut', 'exec'), LC.JT.__dict__)
nodes = [n for n in ast.parse((TR / 'train_segment.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'install_star']
ns = dict(torch=torch); exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TR / 'train_segment.py') + ':install_star', 'exec'), ns)
torch.set_num_threads(threads); torch.set_grad_enabled(False)
w = LC.JT.weights(rank)
tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
mcp = P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'; mc = np.load(mcp) if mcp.exists() else None
e10 = sorted(HERE.glob(f'e10_rank{rank:04d}_N*M.npz')); e10 = np.load(e10[-1]) if e10 else None


RMC = {}
if e10 is not None:
    for l in range(1, 16):
        g = sum(float(e10[f'Z{l}_h{h}_g']) for h in (0, 1)); m1, m2, m3, m4 = sum(e10[f'Z{l}_h{h}_p'] for h in (0, 1)) / g
        v = m2 - m1 ** 2; k4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * v ** 2
        RMC[l] = float((k4 / v ** 2).mean())


def truthB(l):
    """frame-B covariance of the post state of layer l from the e10 Monte Carlo (both halves pooled)."""
    g = sum(float(e10[f'B{l}_h{h}_g']) for h in (0, 1)); s1 = sum(e10[f'B{l}_h{h}_s1'] for h in (0, 1)) / g
    M = sum(e10[f'B{l}_h{h}_M'].astype(np.float64) for h in (0, 1)) / g
    return M - np.outer(s1, s1)


def offrel(A, B):
    D = A - B; D = D - np.diag(np.diag(D)); R = B - np.diag(np.diag(B))
    return float(np.linalg.norm(D) / np.linalg.norm(R))


def run(arm):
    frame, *opts = arm.split('_'); learned = 'on' in opts
    D = dict(dials)
    if frame == 'B' and 'keepdials' not in opts: D.update(bscale=1.0, s_carry=1.0, s_new=1.0)
    LC.JT.DIALS_T = {k: torch.tensor(v, dtype=torch.float64) for k, v in D.items()}
    cd = LC.card(torch, w, p, T['a'], T['c'], e, fine_ckpt=False); ns['install_star'](cd)
    if not learned: cd.hooks['delta'] = lambda layer, d, ctx: d if layer == 0 else torch.zeros_like(d)
    if 'noedge' in opts: cd.hooks['edge'] = lambda layer, inc, ctx: torch.zeros_like(inc)
    if 'nolegs' in opts:      # births without the learned / fitted leg corrections (pure tree-level birth columns)
        def _bd_plain(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
            return (bp * alive[:, None]).to(torch.float32), torch.zeros((bp.shape[0], bp.shape[0]), dtype=torch.float64)
        cd.env['_bd_update'] = _bd_plain
    covs = {}; diag = {}
    prev_out = cd.hooks.get('state_out')
    def state_out(l, st, ctx):
        st = prev_out(l, st, ctx) if prev_out is not None else st
        if l == 15: return st
        m, C, pcol, lam, star, b, alive, radial, hs, Q, F, core, stops, idx = st
        if frame == 'B':
            mm = m @ m; ell = m / mm; mh = m / mm.sqrt(); mh32 = mh.to(torch.float32)
            cbar = C @ ell; s2 = ell @ cbar; cpr = cbar - m * s2
            proj = lambda X: (lambda Y: Y - torch.outer(Y @ mh, mh))(X - torch.outer(mh, mh @ X))
            offd = lambda X: X - torch.diag(torch.diagonal(X))
            C0 = proj(C); eps = torch.zeros((), dtype=torch.float64)
            if 'lead' in opts: Cn = C0
            else:
                Tl = torch.zeros_like(C)
                if 'noT' not in opts:
                    Tl = Tl + (C * (b @ ell) + torch.outer(cbar, b) + torch.outer(b, cbar))      # Sym(C x b) . l
                    if 'nobank' not in opts:                                                      # bank . l = P diag(lam (P'l)) P'
                        p64 = pcol.to(torch.float64); wgt = lam.to(torch.float64) * (ell @ p64)
                        Tl = Tl + (p64 * wgt[None, :]) @ p64.T
                    if 'nostar' not in opts:                                                      # plane-only star . l
                        S = star.to(torch.float64); rest = S.T / 3; dg = torch.diagonal(S)
                        Rod = rest - torch.diag(torch.diagonal(rest)); X = Rod * ell[:, None]
                        Tl = Tl + (X + X.T + torch.diag(Rod @ ell + dg * ell))
                PT = proj(Tl); eps = (offd(PT) * offd(C0)).sum() / (offd(C0) ** 2).sum()            # rigid part of the gain response
                if 'rigid' in opts: PT = eps * C0
                Cn = (1 + s2) * C0 - PT + 2 * torch.outer(cpr, cpr)
            b = b - mh * (mh @ b)
            if 'nob' not in opts: b = b - 2 * cpr
            pcol = pcol - torch.outer(mh32, mh32 @ pcol)
            if Q is not None and torch.is_tensor(Q) and Q.numel(): Q = Q - torch.outer(mh32, mh32 @ Q)
            rad_in = float(radial)
            if 'r0' in opts: radial = torch.zeros_like(radial)
            elif 'rmc' in opts: radial = torch.zeros_like(radial) + RMC[l + 1]                    # oracle scalar: Monte Carlo mean kurtosis of the next frame-B pre-activation
            elif 'rkeep' not in opts: radial = radial + 12 * s2 - 12 * eps
            diag[l] = dict(var_t=float(s2), eps=float(eps), radial_post=rad_in, radial_B=float(radial))
            C = Cn
            st = (m, C, pcol, lam, star, b, alive, radial, hs, Q, F, core, stops, idx)
        covs[l] = C.numpy().copy()
        return st
    cd.hooks['state_out'] = state_out
    pred = cd.run(False).numpy().astype(np.float64)
    mse = ((pred - tp) ** 2).mean(1)
    crel = {}
    for l in (2, 4, 6, 8, 10, 12, 13, 14):
        if frame == 'lab' and mc is not None:
            Tc = mc[f'cov_post_{l}'].astype(np.float64); m = mc[f'mean_post_{l}'].astype(np.float64); mh = m / np.linalg.norm(m)
            Pm = np.eye(1024) - np.outer(mh, mh)
            crel[l] = (offrel(covs[l], Tc), offrel(Pm @ covs[l] @ Pm, Pm @ Tc @ Pm))
        elif frame == 'B' and e10 is not None:
            crel[l] = (offrel(covs[l], truthB(l)),)
    return mse, crel, diag


out = dict(TAG='opus-reexam E11 frame-B card (hook level), physics-only comparison', rank=rank, arms={}); t0 = time.time()
for arm in ARMS:
    mse, crel, diag = run(arm)
    out['arms'][arm] = dict(final=float(mse[15]), per_layer=[float(x) for x in mse], cov_rel_err={str(k): v for k, v in crel.items()}, diag={str(k): v for k, v in diag.items()})
    print(json.dumps(dict(rank=rank, arm=arm, final_x1e9=round(float(mse[15]) * 1e9, 3), all_layer_x1e9=round(float(mse.mean()) * 1e9, 3),
                          per_layer_x1e9=[round(float(x) * 1e9, 2) for x in mse],
                          cov_offdiag_rel_err_pct={k: [round(100 * x, 3) for x in v] for k, v in crel.items()}, radial_B={k: round(v['radial_B'], 4) for k, v in diag.items()}, wall=round(time.time() - t0, 1))), flush=True)
json.dump(out, open(HERE / f'e11_rank{rank:04d}.json', 'w'), indent=1)
print(json.dumps(dict(done=True, rank=rank)), flush=True)
