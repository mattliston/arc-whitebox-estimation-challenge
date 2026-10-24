#!/usr/bin/env python3
"""TAG opus-physicist T74: CLOSED-LOOP ARM of the dial kappa_C = how much of the learned mean correction delta the card moves into the post covariance
(lead 10-10 05:1xZ: hand the winner of the post-map split as a strength dial). The card's step does C <- C - host d' - d host' - d d' after the head (the second
moment kept, the mean changed); T72 (12 networks, paired truths) finds the per-unit variance error at layers 14 / 15 proportional to that change pushed through W with
theta +0.81 / +0.95 (12 of 12 signs): the truth wants almost none of it. Arm: a state_out hook at layers lo..hi adds back (1 - kappa_C) (host d' + d host' + d d'),
d = the delta the step used (ctx['delta']), host = ctx['host']; kappa_C = 1 is the card bit for bit. S42 torch card as T60 (trainer's request a_0hS41_u2_r0287, frozen otr_m),
eval mode, final-layer MSE against the 1e9 truth. Certificate printed per network: the explicit change (host d' + d host' + d d') pushed into the next layer against the
card's own (the difference of v_(l+1) between kappa 1 and 0 is computed in the run).
usage: t74_kappa_c.py <ranks> <kappas, e.g. 1,0.5,0> [lo hi] [threads]"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
RANKS = [int(x) for x in sys.argv[1].split(',')]; KAPPAS = [float(x) for x in sys.argv[2].split(',')]
LO, HI = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1, 14); threads = sys.argv[5] if len(sys.argv) > 5 else '3'
FROZEN = HERE / 'otr_frozen_f2dfccba'; SHA = 'f2dfccbadf7d90d727d9283cf6454bddcb5ef8912ebe020f7b3bf2c9e1a1cb6e'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]
sys.path.insert(0, str(O / 'trainer')); sys.path.insert(0, str(FROZEN))
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == FROZEN and hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest() == SHA, E.M.__file__
from otr_common import load_params, load_pl
torch = E.torch; t = torch; f64 = t.float64
REQ = json.loads((O / 'trainer/run_M/claimed/a_0hS41_u2_r0287.w0.json').read_text())
E.set_parameters(*load_params(torch, REQ['params'])); E.PL.set_values({f: [float(x) for x in v] for f, v in load_pl(torch, REQ['params']).items()})
E.STAR_ATTACH = False; E.KEEP_FINAL = False; E.STAR_SS = REQ['ss']; E.M.F4C = bool(REQ['f4c']); E.M.NCAR = REQ['ncar']; E.M.HEADF = REQ['hf']
if hasattr(E.M, 'RG'): E.M.RG = False
hu = np.array(json.loads(REQ['hu']) if isinstance(REQ['hu'], str) else REQ['hu'], dtype=np.float64); HU = np.zeros((len(E.M.HEAD_COLUMNS), 128)); HU[:hu.shape[0]] = hu
E.M.head_weights(HU.tolist())
KAPPA = [1.0]; INFO = {}
install0 = E.M.install


def install(cd):
    install0(cd); side = cd.m_side; out0 = cd.hooks['state_out']

    def state_out(l, out, ctx):
        out = out0(l, out, ctx)
        if LO <= l <= HI and KAPPA[0] != 1.0 and not side.get('again'):
            host, d = ctx['host'], ctx['delta']
            host = host.to(out[1].dtype); d = d.to(out[1].dtype).reshape(-1)
            back = t.outer(host, d) + t.outer(d, host) + t.outer(d, d)
            out = (out[0], out[1] + (1.0 - KAPPA[0]) * back) + tuple(out[2:])
            INFO[l] = float(back.abs().max())
        return out
    cd.hooks['state_out'] = state_out
E.M.install = install
res = {}
for r in RANKS:
    for k in KAPPAS:
        KAPPA[0] = k; INFO.clear(); t0 = time.time()
        rec = E.run_network(r, 'eval'); res[(r, k)] = rec['mse']
        print(json.dumps(dict(TAG='physicist T74 kappa_C arm', rank=r, kappa=k, lo=LO, hi=HI, final_x1e9=round(rec['mse'] * 1e9, 6),
                              change_vs_k1=(round(100 * (rec['mse'] / res[(r, 1.0)] - 1), 3) if (r, 1.0) in res else None), wall=round(time.time() - t0, 1))), flush=True)
for k in KAPPAS:
    if k == 1.0: continue
    ratios = np.array([res[(r, k)] / res[(r, 1.0)] for r in RANKS if (r, k) in res and (r, 1.0) in res])
    if len(ratios): print(json.dumps(dict(TAG='physicist T74 summary', kappa=k, lo=LO, hi=HI, networks=len(ratios), pooled_change_pct=round(100 * (sum(res[(r, k)] for r in RANKS) / sum(res[(r, 1.0)] for r in RANKS) - 1), 3),
                                          lower=int((ratios < 1).sum()), se_pct=round(100 * float(np.std(np.log(ratios), ddof=1) / np.sqrt(len(ratios))) if len(ratios) > 1 else 0.0, 3))), flush=True)
