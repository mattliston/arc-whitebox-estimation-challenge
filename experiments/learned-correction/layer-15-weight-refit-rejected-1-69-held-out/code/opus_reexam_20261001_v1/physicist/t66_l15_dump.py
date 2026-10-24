#!/usr/bin/env python3
"""TAG opus-physicist T66: LAYER-15 DUMP OF THE S42 TORCH CARD for the OFFLINE fit of new final-layer head columns (lead 10-10 03:05Z / 03:22Z: the final layer's correction
touches nothing downstream, so final prediction = raw mean at 15 + delta15(its inputs; weights) can be fitted as a plain regression on dumped layer-15 inputs).
Card, request and engine as T60 (the trainer's S42 card, request a_0hS41_u2_r0287, frozen otr_m f2dfccba), forward only (eval mode). Dumped per unit of layer 15:
  H = the head's 126 inputs as gnn_step forms them (30 normalised features after the layer-15 mask + 96 message columns), its sigma input, the state in / out, delta,
  the raw mean before delta (host), the final prediction m, the 1e9 truth; mu, v, k3, k4 of the pre-activation; the module's 21 columns (17-19 standardised over
  the units alive at 14, as frozen);
  CANDIDATES, raw (not standardised): c17 = the law matrix's diagonal at 15 (module column 17's numerator dB: sum_r pp_ir (w_r pp_ir + x_r pp_i,partner(r)), its least-
  squares residual on (v, 1, mu^2) over the KEPT rows |mu / sigma| <= 3.5, over (sum v) v_i), c18 = the young pairs' fourth-chaos plane diagonal over v^2, c19 = K4L's value
  over v^2; b6 f15, qold, eold (b6cols.py); per-age local quartic q / chaos f / k3 c / energies e, s of the bank's cohorts; (W o W) pushes of layer 14's post v, k3, k4,
  kink density, delta^2; the top-8 modes of layer 14's post covariance: eigenvalues and loadings (W15' u_k)_i (sign: sum(u_k) > 0).
-> l15dump_rank<rank>.npz ; usage: t66_l15_dump.py <rank,rank,...> [threads = 4]   (one process for several networks; between networks it waits while a quiet
   window or a lead's machine-priority line is in force or the idle gate fails, and stops at physicist/STOP_T66)"""
import hashlib, json, math, os, sys, time
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
from otr_common import load_params, load_pl, data_paths
torch = E.torch; t = torch; f64 = t.float64; n = 1024; t0 = time.time(); NMODE = 8
REQ = json.loads((O / 'trainer/run_M/claimed/a_0hS41_u2_r0287.w0.json').read_text())
E.set_parameters(*load_params(torch, REQ['params'])); E.PL.set_values({f: [float(x) for x in v] for f, v in load_pl(torch, REQ['params']).items()})
E.STAR_ATTACH = False; E.KEEP_FINAL = True; E.STAR_SS = REQ['ss']; E.M.F4C = bool(REQ['f4c']); E.M.NCAR = REQ['ncar']; E.M.HEADF = REQ['hf']
if hasattr(E.M, 'RG'): E.M.RG = False
hu = np.array(json.loads(REQ['hu']) if isinstance(REQ['hu'], str) else REQ['hu'], dtype=np.float64); HU = np.zeros((len(E.M.HEAD_COLUMNS), 128)); HU[:hu.shape[0]] = hu
E.M.head_weights(HU.tolist())
R = {}; ST = {}; MODES = {}; TM = dict(eigh=0.0, c18=0.0, age=0.0)
cell = lambda fn, name: fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents
BANDS = (('1', 1, 1), ('2', 2, 2), ('3', 3, 3), ('4', 4, 4), ('56', 5, 6), ('7', 7, 99))
sys.path.append(str(HERE)); import b6cols as B6
from t60_s42_dump_parts import per_age      # the per-age sums of T60, unchanged


def hooks(cd):
    side = cd.m_side
    out0 = cd.hooks['state_out']

    def state_out(l, out, ctx):
        out = out0(l, out, ctx)
        if l == 14:
            with t.no_grad():
                aux = ctx['aux']; Cp = out[1].detach().to(f64); dl = ctx.get('delta'); z_ = t.zeros(n, dtype=f64); pk3_ = aux.get('post_k3'); pk4_ = aux.get('post_k4')
                ST[14] = dict(pv=t.diagonal(Cp).clone(), pk3=(pk3_.detach().to(f64) if t.is_tensor(pk3_) else z_), pk4=(pk4_.detach().to(f64) * t.ones(n, dtype=f64) if t.is_tensor(pk4_) else z_),
                              d=(aux['pdf'] / aux['sigma']).detach().to(f64), dd=(dl.detach().to(f64) ** 2 if dl is not None else z_))
                t1 = time.time(); ev, U = t.linalg.eigh(0.5 * (Cp + Cp.T)); TM['eigh'] += time.time() - t1; ev, U = ev.flip(0)[:NMODE], U.flip(1)[:, :NMODE]
                MODES[14] = (ev, U * t.where(U.sum(0) < 0, -1.0, 1.0).to(f64)[None, :]); R['post_modes14'] = ev; R['post_trace14'] = t.trace(Cp).reshape(1)
        if l == 15:
            R['m'] = out[0].detach().to(f64)
            if 'host' in ctx: R['host'] = ctx['host'].detach().to(f64)
        return out
    cd.hooks['state_out'] = state_out

    k3h0 = cd.hooks.get('k3_15')

    def k3_15(l, k3, ctx):
        k3 = k3 if k3h0 is None else k3h0(l, k3, ctx)
        with t.no_grad():
            vv, mu = ctx['vv'].detach().to(f64).clamp_min(1e-12), ctx['mu'].detach().to(f64)
            t1 = time.time(); R.update(mu=mu, v=vv, k3=k3.detach().to(f64).clone()); R.update(per_age(15, side, vv, mu, t, f64, n, BANDS)); TM['age'] += time.time() - t1
            W = cd.ws[15].to(f64); W2 = W * W; p = ST[14]
            R.update(pv=(W2.T @ p['pv']) / vv, pk3=W2.T @ p['pk3'], pk4=W2.T @ p['pk4'], pd=W2.T @ p['d'], pdd=(W2.T @ p['dd']) / vv)
            lam, U = MODES[14]; L = W.T @ U; R['mload'] = L / vv.sqrt()[:, None]; R['mshare'] = (L * L) * lam[None, :] / vv[:, None]
        return k3
    cd.hooks['k3_15'] = k3_15
    k4h0 = cd.hooks.get('k4_15')

    def k4_15(l, k4, ctx):
        k4 = k4 if k4h0 is None else k4h0(l, k4, ctx)
        R['k4'] = (k4.detach().to(f64) if t.is_tensor(k4) and k4.ndim else float(k4) * t.ones(n, dtype=f64)).clone()
        with t.no_grad():      # batch-6 columns of layer 15, as b6cols.py's k3_15 / k4_15 hooks form them (b6cols is not installed: its layers 1-14 are not needed here)
            vv, mu = ctx['vv'].detach().to(f64).clamp_min(1e-12), ctx['mu'].detach().to(f64)
            R['b6_f15'] = B6._so2_parts(t, mu, vv, R['k3'], R['k4'])[0]; R['b6_qold'], R['b6_eold'] = B6._old_sums(t, side, 15, vv)
        return k4
    cd.hooks['k4_15'] = k4_15

    gnn_m = cd.env['gnn_step']; cols_fn = cell(gnn_m, 'columns')

    def gnn_step(l, x, sigma, W, state, P):
        if l == 15:
            with t.no_grad():
                xx = x.detach().to(t.float32); xn = t.clamp((xx - P['centers'][l]) / P['scales'][l], -8, 8)
                xn = xn * t.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=t.float32)[None, :]
                g = xn[:, 5:6]; src = state.detach() @ P['msg_w'].T + P['msg_b']; cm = int(P['msg_w'].shape[0]) // 3
                w = W.detach().to(t.float32); w2 = w * w; w3 = w2 * w
                m1 = w.T @ src[:, :cm]; m2 = w2.T @ src[:, cm:2 * cm]; m3 = w3.T @ src[:, 2 * cm:]
                e1 = w2.T @ (src[:, :cm] ** 2); e2 = (w2 * w2).T @ (src[:, cm:2 * cm] ** 2); e3 = (w3 * w3).T @ (src[:, 2 * cm:] ** 2)
                msgs = t.clamp(t.cat([g * m1, g * m2, g * m3, t.log1p(t.clamp(e1, min=0)), t.log1p(t.clamp(e2, min=0)), t.log1p(t.clamp(e3, min=0))], 1), -8, 8)
                R.update(H=t.cat([xn, msgs], 1).to(f64), x=xx.to(f64), sigma=sigma.detach().to(f64).reshape(-1), state_in=state.detach().to(f64), cols=cols_fn(l).to(f64))
                # raw final-layer columns 17-19 (module's arithmetic, without its standardisation; 17's regression over the kept rows)
                F15 = side['f15']; vv = F15['vv'].to(f64).clamp_min(1e-12); mu15 = F15['mu'].to(f64); kept = (mu15 / vv.sqrt()).abs() <= 3.5
                pp = side['pp'].detach().to(f64); ncol = pp.shape[1]; S6 = cd.ncar6_state
                if S6.get('w') is not None and ncol:
                    wl, xl, partner = S6['w'][:ncol].to(f64), S6['x'][:ncol].to(f64), S6['partner'][:ncol]
                    dB = (pp * (pp * wl[None, :] + pp[:, partner] * xl[None, :])).sum(1)
                    X = t.stack([vv, t.ones_like(vv), mu15 * mu15], 1)
                    co = t.linalg.lstsq(X[kept], dB[kept][:, None]).solution[:, 0]
                    R['c17'] = (dB - X @ co) / (vv.sum() * vv); R['c17raw'] = dB
                for age, Sk in getattr(cd, 'k31l_states', []):
                    if age == 2 and 'plane_of' in Sk and ncol: t1 = time.time(); R['c18'] = t.diagonal(Sk['plane_of'](15, side['pp'])).to(f64) / (vv * vv); TM['c18'] += time.time() - t1
                R['c19'] = F15['dk'].to(f64) / (vv * vv); R['kept'] = kept.to(f64)
        out = gnn_m(l, x, sigma, W, state, P)
        if l == 15: R.update(delta=out[0].detach().to(f64).reshape(-1), state_out=out[1].detach().to(f64))
        return out
    cd.env['gnn_step'] = gnn_step


install0 = E.M.install
B6CAP = {}


def install(cd):
    install0(cd); hooks(cd)
E.M.install = install
import subprocess


def gate():
    """wait while a quiet window is open, a lead's MACHINE PRIORITY line has not run out, or the idle gate fails; False if STOP_T66 exists"""
    while True:
        if (HERE / 'STOP_T66').exists(): return False
        L = [x for x in (O / 'TEAM_LOG.md').read_text().splitlines() if x.startswith('[lead] ')]
        o = max([i for i, x in enumerate(L) if 'QUIET WINDOW OPEN' in x], default=-1); c = max([i for i, x in enumerate(L) if 'QUIET WINDOW CLOSED' in x], default=-1)
        pr = [x for x in L if 'MACHINE PRIORITY UNTIL' in x]; busy = o > c
        if pr:
            import re as _re
            mm = _re.search(r'MACHINE PRIORITY UNTIL (\d\d):(\d\d)Z', pr[-1]); now = time.gmtime()
            if mm and now.tm_hour * 60 + now.tm_min < int(mm.group(1)) * 60 + int(mm.group(2)): busy = True
        if not busy and subprocess.run(['bash', str(O / 'idle_gate.sh'), '30'], capture_output=True).returncode == 0: return True
        time.sleep(20)


for rank in RANKS:
    if (HERE / f'l15dump_rank{rank:04d}.npz').exists(): continue
    if not gate(): print(json.dumps(dict(TAG='physicist T66 stop', rank=rank)), flush=True); break
    t0 = time.time(); R.clear(); ST.clear(); MODES.clear(); TM.update(eigh=0.0, c18=0.0, age=0.0)
    rec = E.run_network(rank, 'eval')
    truth = np.load(data_paths(rank)[1])['mc_post'].astype(np.float64)[15]
    out = {'final_x1e9': np.array(rec['mse'] * 1e9), 'truth': truth, 'pred': np.array(rec['final_prediction'], dtype=np.float64)}
    for k, x in R.items(): out[k] = x.cpu().numpy().astype(np.float64)
    np.savez(HERE / f'l15dump_rank{rank:04d}.npz', **out)
    chk = float(np.abs(out['pred'] - out['m']).max()) if 'm' in out else None
    print(json.dumps(dict(TAG='physicist T66 L15 dump', rank=rank, final_x1e9=round(rec['mse'] * 1e9, 6), mse_check_x1e9=round(float(((out['pred'] - truth) ** 2).mean()) * 1e9, 6),
                          pred_vs_m=chk, keys=len(out), card_s=round(rec['seconds'], 1), hooks_s={k: round(v, 1) for k, v in TM.items()}, wall=round(time.time() - t0, 1))), flush=True)
