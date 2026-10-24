#!/usr/bin/env python3
"""TAG opus-physicist T78: CLOSED-LOOP ORACLE of the pair map's PLANES at the last pair layers (13-14), on the trainer's current card (jmD point, replayed as T76),
learned parts on. An outermost pair_in hook replaces, at the chosen layers, the off-diagonal part of the planes the pair map receives by card + eps (truth - card):
  K21  the third-order plane (pin[2]; its diagonal, the per-unit k3, stays the card's)
  K22  the (2,2) plane (pin[4], the k22 override; the card's is radial (v v' + 2 C o C) / 3)
  K31  the (3,1) plane, through the map's remainder31: T31_true = K31_true / (v_i s_i s_j), remainder = T31_target - t4_i r_i rho_ij (r_i = the card's K31 row factor
       0.85 + 0.5 mu_i k3_i / v_i^2 scaled as gn3 configures it is NOT re-derived here: the remainder is taken against the card's own T31, recomputed from its inputs)
Truth planes: public_g19_attribution_v1/mps_planes_rank<r>.npz (8M samples; K21_l, K22_l, K31_l of the pre-activations of layer l). Networks with truth planes: 35, 100.
usage: t78_plane_oracle.py <ranks> <arm|arm|...>  ; arm = base or '<planes>:<layers>:<eps>', planes in {K21,K22,K31,all}, layers 'a-b', e.g. 'all:13-14:1'"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
RANKS = [int(x) for x in sys.argv[1].split(',')]; ARMS = sys.argv[2].split('|'); threads = sys.argv[3] if len(sys.argv) > 3 else '2'
FROZEN = HERE / 'otr_frozen_482e996f'; SHA = '482e996f479fbb8d0b237c2a5f04982f604fad3e3994f9772db46569b332a226'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]
sys.path.insert(0, str(O / 'trainer')); sys.path.insert(0, str(FROZEN))
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == FROZEN and hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest() == SHA, E.M.__file__
from otr_common import load_params, load_pl
torch = E.torch; t = torch; f64 = t.float64
REQ = json.loads((O / 'trainer/run_M/claimed/a_0mG1_base_r0005.w1.json').read_text())
E.set_parameters(*load_params(torch, REQ['params'])); shifts = {f: list(v) for f, v in load_pl(torch, REQ['params']).items()}
E.PL.set_values(shifts)
E.STAR_ATTACH = bool(REQ.get('attach', E.STAR_ATTACH)); E.KEEP_FINAL = False; E.KEEP_ALL = False; E.STAR_SS = REQ.get('ss')
E.M.F4C = bool(REQ.get('f4c', E.M.F4C)); E.M.NCAR = REQ.get('ncar'); E.M.HEADF = int(REQ.get('hf') or 0); E.M.ROWS.clear(); E.M.RG = bool(REQ.get('rg'))
E.M.DUMP15 = REQ.get('dump15'); E.M.B6 = bool(REQ.get('b6')); E.M.HSPLIT = {int(k): tuple(v) for k, v in (REQ.get('hsplit') or {}).items()}
E.M.HBANDS = bool(REQ.get('hbands')); E.M.B7 = bool(REQ.get('b7'))
if E.M.RG or E.M.RG_W: E.M.readout_weights(REQ.get('rgw'))
if E.M.HEADF or E.M.HEADU is not None: E.M.head_weights(REQ.get('hu'))
ARM = [None]; TRUTH = {}; INFO = {}
install0 = E.M.install


def install(cd):
    install0(cd); pin0 = cd.hooks['pair_in']; side = getattr(cd, 'm_side', {})

    def pin_hook(l, pin, ctx):
        pin = pin0(l, pin, ctx)
        a = ARM[0]
        if a is None or l not in a['layers'] or side.get('again'): return pin
        with t.no_grad():
            mu, C, K21, k4, k22, rem = pin
            n = mu.shape[0]; eye = t.eye(n, dtype=t.bool); eps = a['eps']; T = TRUTH
            v = t.diagonal(C).detach().to(f64).clamp_min(1e-12); s = v.sqrt()
            K21n, k22n, remn = K21, k22, rem
            if a['planes'] in ('K21', 'all'):
                Kc = K21.detach().to(f64); Kt = t.tensor(T[f'K21_{l}'], dtype=f64)
                K21n = t.where(eye, Kc, Kc + eps * (Kt - Kc)).to(K21.dtype)
            if a['planes'] in ('K22', 'all', 'allc'):
                Kc = (k22.detach().to(f64) if t.is_tensor(k22) and k22.ndim == 2 else None)
                if Kc is not None:
                    Kt = t.tensor(T[f'K22_{l}'], dtype=f64); k22n = t.where(eye, Kc, Kc + eps * (Kt - Kc)).to(k22.dtype)
            if a['planes'] in ('K31', 'all', 'allc'):
                # the card's T31 = t4 r rho (+ rem); the target T31 = card + eps (truth - card), as a new remainder
                k3 = t.diagonal(K21.detach().to(f64)); k4v = (k4.detach().to(f64) if t.is_tensor(k4) and k4.ndim else float(k4) * t.ones(n, dtype=f64))
                rho = (C.detach().to(f64) / t.outer(s, s)).masked_fill(eye, 0.); t4 = k4v / v ** 2
                rowf = cd.env['_ROWF'][0](mu.detach().to(f64), k3, v) if cd.env.get('_ROWF') and cd.env['_ROWF'][0] is not None else 1.0
                T31c = t4[:, None] * (rowf[:, None] if t.is_tensor(rowf) else rowf) * rho + (rem.detach().to(f64) if t.is_tensor(rem) else 0.0)
                T31t = (t.tensor(T[f'K31_{l}'], dtype=f64) / (v[:, None] * s[:, None] * s[None, :])).masked_fill(eye, 0.)
                base_r = (rem.detach().to(f64) if t.is_tensor(rem) else t.zeros(n, n, dtype=f64))
                remn = (base_r + eps * (T31t - T31c)).masked_fill(eye, 0.).to(C.dtype)
            INFO[l] = 1
            return (mu, C, K21n, k4, k22n, remn)
    cd.hooks['pair_in'] = pin_hook
    pmap0 = cd.env['_nonlinear_pair_batch']

    def pmap(*args, **kwargs):      # arm K21c: the TRUE K21 off the diagonal in the map's COVARIANCE output only (target / post planes / mean from the card's K21)
        res = pmap0(*args, **kwargs); a = ARM[0]; l = cd.call_layer
        if a is None or a['planes'] not in ('K21c', 'allc') or l not in a['layers'] or side.get('again'): return res
        with t.no_grad():
            K21 = args[2]; n = K21.shape[0]; eye = t.eye(n, dtype=t.bool); Kc = K21.detach().to(f64); Kt = t.tensor(TRUTH[f'K21_{l}'], dtype=f64)
            Kn = t.where(eye, Kc, Kc + a['eps'] * (Kt - Kc)).to(K21.dtype)
            res2 = pmap0(args[0], args[1], Kn, *args[3:], **kwargs); INFO[l] = 2
        return (res[0], res2[1]) + tuple(res[2:])
    cd.env['_nonlinear_pair_batch'] = pmap
E.M.install = install
out = HERE / 't78_plane_oracle.jsonl'
for r in RANKS:
    D = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{r:04d}.npz'); TRUTH.clear(); TRUTH.update({k: D[k] for k in D.files if k.startswith('K')})
    base = None
    for arm in ARMS:
        if arm == 'base': ARM[0] = None
        else:
            pl, ly, ep = arm.split(':'); lo, hi = (int(x) for x in ly.split('-'))
            if any(f'K21_{l}' not in TRUTH for l in range(lo, hi + 1)): print('no truth planes for', r, ly); continue
            ARM[0] = dict(planes=pl, layers=set(range(lo, hi + 1)), eps=float(ep))
        INFO.clear(); t0 = time.time(); rec = E.run_network(r, 'eval')
        if arm == 'base': base = rec['mse']
        line = dict(rank=r, arm=arm, mse=rec['mse'], change=(rec['mse'] / base - 1) if base else None, layers_hit=sorted(INFO), wall=round(time.time() - t0, 1))
        with open(out, 'a') as fh: fh.write(json.dumps(line) + '\n')
        print(json.dumps(dict(TAG='physicist T78', **{k: line[k] for k in ('rank', 'arm', 'change', 'layers_hit')}, final_x1e9=round(rec['mse'] * 1e9, 6))), flush=True)
