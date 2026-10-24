#!/usr/bin/env python3
"""TAG fable-analyst (AN-60): COMPONENT ABLATION AUDIT on today's card (lead 23:2x, "find more s1s"). Card: the physicist's
s6bcard_loop3 (LOOP_DIALMOD = s6bcard_jfitR1: round-1 dials; S18 low-rank star with s1 = 0; gn_linear_hook; E-67 pair-series loop form,
fine_ckpt off), params loaded exactly as refit_mps_v3 does (payload tensors copied from the .pt; default the loopR1 params), learned ON,
no refit, float64 CPU, final MSE against the organizer truth. Each variant sets ONE optional term to 0 (or its simplest form), WRAPPING
the card's own hooks (the dial module's pair_in / k4_15 / b_post / state_out stay underneath):
  star_s2     star push off-diagonal (the s2 term; s1 is already 0) -> 0, L1-14     star_diag   star push diagonal -> 0, L1-14
  star_l15    the L15 star k3 part (star_diagonal) -> 0
  leg0..leg3  birth-update K21 leg j -> 0 (its learned c_j*orth(dir_j) AND its physical dial add X_j; leg1 = the X1 physical leg)
  b           carried b -> 0 (b_post, every layer)          bfix   gain b-fix gamma .1 -> 0
  carry       CP-weight dials s_carry / s_new -> 1 (the dial module's state_out hook removed)
  k4          radial k4 -> 0 everywhere: pair map input (also zeroes T31, which is built from t4), the GN dcov's ctx k4, L15 k4
  k22         K22 input to the pair map -> 0                k31    K31 row factor in the pair map -> 0 (T31 = 0; the GN dcov keeps its own)
  gndcov      GN v3 covariance increment (gn_linear_hook) off
  omrad       omitted-radial source off                     recenter  recentering (knob .92) -> 0
  head15      learned mean correction at L15 -> 0           deltaA-B  learned mean correction -> 0 at layers A-B
  edge        message edge -> 0 (L8-14)                     edgeA-B   message edge -> 0 at layers A-B
  bdcopy      parity row: my exec'd copy of the dial module's _bd_update with no leg removed (must equal base)
usage: ablate_loop3.py <ranks|valid16> <threads> <out.jsonl> <variants> [params.pt]"""
import os, sys, json, time, inspect
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
os.environ.setdefault('LOOP_DIALMOD', 's6bcard_jfitR1')
for d in ('fable_engineer_v1', 'fable_covhunter_v1', 'fable_physicist_v1/refit6b', 'fable_analyst_v1'):
    sys.path.insert(0, str(P2 / d))
import numpy as np, torch
import s6bcard_loop3 as LC
import s6bcard_joint2 as SJ
arg, THREADS, OUT, VARIANTS = sys.argv[1], int(sys.argv[2]), HERE / sys.argv[3], sys.argv[4].split(',')
PARAMS = Path(sys.argv[5]) if len(sys.argv) > 5 else P2 / 'fable_physicist_v1/refit6b/s6b_loopR1/params_epoch0.pt'
torch.set_num_threads(THREADS)
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
RANKS = inputs['valid'][:16] if arg == 'valid16' else [int(x) for x in arg.split(',')]
p, a, c, e = LC.payload(torch)
sv = torch.load(PARAMS)
with torch.no_grad():
    for k in p: p[k].copy_(sv['p'][k])
    a.copy_(sv['a']); c.copy_(sv['c'])
    for k in e: e[k].copy_(sv['e'][k])
_src = inspect.getsource(SJ.make_bd)
assert _src.count("def make_bd(torch, D):") == 1 and _src.count("        for j in range(4):\n") == 1
_src = _src.replace("def make_bd(torch, D):", "def make_bd_ablate(torch, D, zero=None):").replace(
    "        for j in range(4):\n", "        for j in range(4):\n            if j == zero: continue\n")
_ns = {}; exec(_src, SJ.__dict__, _ns); make_bd_ablate = _ns['make_bd_ablate']


def wrap(cd, name, fn):
    old = cd.hooks.get(name)
    cd.hooks[name] = (lambda l, v, ctx, old=old: fn(l, old(l, v, ctx), ctx)) if old is not None else fn


def band(v, head):
    a_, b_ = (int(x) for x in v[len(head):].split('-')); return a_, b_


def apply(cd, v):
    t = torch
    if v == 'base': return
    if v == 'bdcopy': cd.env['_bd_update'] = make_bd_ablate(t, dict(SJ.DIALS), None)
    elif v.startswith('leg'): cd.env['_bd_update'] = make_bd_ablate(t, dict(SJ.DIALS), int(v[3:]))
    elif v == 'star_s2':
        f0 = cd.env['_flms_star']; cd.env['_flms_star'] = lambda w, A, f0=f0: t.diag(t.diagonal(f0(w, A)))
    elif v == 'star_diag':
        f0 = cd.env['_flms_star']
        def star(w, A, f0=f0):
            out = f0(w, A).clone(); out.diagonal().zero_(); return out
        cd.env['_flms_star'] = star
    elif v == 'star_l15': cd.env['_flms_star_diagonal'] = lambda w, A: t.zeros(w.shape[1], dtype=t.float64)
    elif v == 'b': wrap(cd, 'b_post', lambda l, b, ctx: b * 0)
    elif v == 'bfix': cd.knobs['bfix_gamma'] = 0.0
    elif v == 'carry': assert cd.hooks.pop('state_out', None) is not None
    elif v == 'k4':
        def pin(l, q, ctx):
            ctx['k4'] = ctx['k4'] * 0; return (q[0], q[1], q[2], q[3] * 0, q[4], q[5])
        wrap(cd, 'pair_in', pin); wrap(cd, 'k4_15', lambda l, k4, ctx: k4 * 0)
    elif v == 'k22': wrap(cd, 'pair_in', lambda l, q, ctx: (q[0], q[1], q[2], q[3], q[4] * 0, q[5]))
    elif v == 'k31': cd.env['_ROWF'][0] = lambda mu, k3, vv: 0.0 * mu
    elif v == 'gndcov': assert cd.hooks.pop('cov_pm', None) is not None
    elif v == 'omrad': cd.knobs['omitted_radial'] = False
    elif v == 'recenter': cd.knobs['recenter'] = 0.0
    elif v == 'head15': wrap(cd, 'delta', lambda l, d, ctx: d * 0 if l == 15 else d)
    elif v.startswith('delta'):
        a_, b_ = band(v, 'delta'); wrap(cd, 'delta', lambda l, d, ctx: d * 0 if a_ <= l <= b_ else d)
    elif v == 'edge': wrap(cd, 'edge', lambda l, x, ctx: x * 0)
    elif v.startswith('edge'):
        a_, b_ = band(v, 'edge'); wrap(cd, 'edge', lambda l, x, ctx: x * 0 if a_ <= l <= b_ else x)
    else: raise ValueError(v)


STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
with OUT.open('a') as fo:
    for r in RANKS:
        w = np.load(STORE / Path(man[r]['path']).parent / 'weights.npy')
        tp = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
        for v in VARIANTS:
            t0 = time.time(); cd = LC.card(torch, w, p, a, c, e, fine_ckpt=False); apply(cd, v)
            with torch.no_grad(): pred = cd.run(False)[15].to(torch.float64).numpy()
            rec = dict(rank=r, variant=v, mse=float(((pred - tp) ** 2).mean()), wall=round(time.time() - t0, 1), params=PARAMS.name,
                       params_dir=PARAMS.parent.name, card='s6bcard_loop3 (' + os.environ['LOOP_DIALMOD'] + ')')
            fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps({k: rec[k] for k in ('rank', 'variant', 'mse', 'wall')}), flush=True)
