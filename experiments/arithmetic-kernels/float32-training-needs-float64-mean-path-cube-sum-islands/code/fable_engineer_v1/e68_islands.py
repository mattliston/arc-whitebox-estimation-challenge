#!/usr/bin/env python3
"""TAG fable-engineer (E-68a islands): which parts of the loop card need float64? Card = the physicist's s6bcard_loop (LOOP_DIALMOD,
default s6bcard_jfitR1) + the E-67 drop-in (fine_ckpt=False + pair_series loop form), CPU. Reference = the float64 card. The float32
card is the same code under F32Mode (e68_f32_cpu.py). An ISLAND runs one function / hook in float64 inside the float32 card: its tensor
arguments are cast to float64, it runs with torch function handling disabled (its own float32 casts stay native), and its float64
outputs are cast back to float32. Per arm: L15 prediction relative diff and final-MSE change vs the float64 card (no grad), then one
grad step: per-tensor relative gradient diff (median, max). The error an island removes is the error made inside that part.
usage (python with torch): e68_islands.py <threads> <rank> [arm,arm,...]"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
os.environ.setdefault('LOOP_DIALMOD', 's6bcard_jfitR1')
for d in ('fable_analyst_v1', 'fable_physicist_v1/refit6b', 'fable_covhunter_v1'): sys.path.insert(0, str(P2 / d))
sys.path.insert(0, str(HERE))
import numpy as np, torch
from torch.overrides import TorchFunctionMode
import importlib.util
_spec = importlib.util.spec_from_file_location('s6bcard_loop_physicist', P2 / 'fable_physicist_v1/refit6b/s6bcard_loop.py')
LC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(LC)
import pair_series_torch as ps
torch.set_num_threads(int(sys.argv[1])); RANK = int(sys.argv[2])
F64, F32 = torch.float64, torch.float32
def conv(x, src, dst):
    if isinstance(x, torch.Tensor): return x.to(dst) if x.dtype == src else x
    if isinstance(x, tuple): return tuple(conv(y, src, dst) for y in x)
    if isinstance(x, list): return [conv(y, src, dst) for y in x]
    if isinstance(x, dict): return {k: conv(v, src, dst) for k, v in x.items()}
    return x
F64_SEEN = []
class F32Mode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if kwargs.get('dtype') is F64: kwargs['dtype'] = F32
        args = tuple(F32 if a is F64 else a for a in args)
        if len(F64_SEEN) < 12 and any(isinstance(a, torch.Tensor) and a.dtype == F64 for a in list(args) + list(kwargs.values())):
            F64_SEEN.append(getattr(func, '__name__', str(func)))
        return conv(func(*args, **kwargs), F64, F32)
def island(fn):
    def wrapped(*args, **kw):
        a64, k64 = conv(args, F32, F64), conv(kw, F32, F64)
        with torch._C.DisableTorchFunction():
            out = fn(*a64, **k64)
        return conv(out, F64, F32)
    return wrapped
ENV = ['_nonlinear_pair_batch', '_nonlinear_first_gaussian', '_bd_update', '_message_edge_source', 'features', 'final_scalar_features', 'gnn_step',
       'nonlinear_scalar', '_bdnh_coefficients', '_flms_star', '_flms_star_diagonal']
SHARED = ['push', 'read']
p64, a64, c64, e64 = LC.payload(torch)
def leaves(p, a, c, e):
    return [p['head'], p['head_b'], e['output'], e['bias'], a, c] + [e[k] for k in e if k not in ('output', 'bias')] + [p[k] for k in ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')]
names = ['head', 'head_b', 'e.output', 'e.bias', 'a', 'c'] + ['e.' + k for k in e64 if k not in ('output', 'bias')] + ['msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b']
f32c = lambda v: v.detach().to(F32).clone()
p32, a32, c32, e32 = {k: f32c(v) for k, v in p64.items()}, f32c(a64), f32c(c64), {k: f32c(v) for k, v in e64.items()}
L64, L32 = leaves(p64, a64, c64, e64), leaves(p32, a32, c32, e32)
for x in L64 + L32: x.requires_grad_(True)
w = LC.DM.weights(RANK)
tp = torch.tensor(np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{RANK:04d}.npz')['mc_post'][15].astype(np.float64))
def build(arm):
    if arm == 'f64':
        return ps.install(LC.card(torch, w, p64, a64, c64, e64, fine_ckpt=False), torch)
    cd = ps.install(LC.card(torch, w, p32, a32, c32, e32, fine_ckpt=False), torch)
    isl = [] if arm == 'f32' else (ENV + ['shared.' + k for k in SHARED] + ['hook.' + k for k in cd.hooks] if arm == 'all_parts' else arm.split('+'))
    for k in isl:
        if k.startswith('shared.'): cd.shared[k[7:]] = island(cd.shared[k[7:]])
        elif k.startswith('hook.'): cd.hooks[k[5:]] = island(cd.hooks[k[5:]])
        elif k == 'hooks':
            for h in list(cd.hooks): cd.hooks[h] = island(cd.hooks[h])
        else: cd.env[k] = island(cd.env[k])
    return cd
def run(arm, grad):
    L = L64 if arm == 'f64' else L32
    if not grad:
        with torch.no_grad():
            if arm == 'f64': return build(arm).run(False).to(F64)
            with F32Mode(): out = build(arm).run(False)
            return out.to(F64)
    for x in L: x.grad = None
    if arm == 'f64':
        pred = build(arm).run(True); loss = ((pred[15].to(F64) - tp) ** 2).mean(); loss.backward()
    else:
        with F32Mode(): pred = build(arm).run(True)
        loss = ((pred[15].to(F64) - tp) ** 2).mean()
        with F32Mode(): torch.autograd.backward(loss)
    return [x.grad.detach().to(F64).clone() if x.grad is not None else None for x in L]
q64 = run('f64', False); g64 = run('f64', True)
mse = lambda q: float(((q[15] - tp) ** 2).mean())
arms = sys.argv[3].split(',') if len(sys.argv) > 3 else (['f32', 'all_parts'] + ENV + ['shared.push', 'shared.read', 'hooks', 'hook.cov_pm'])
for arm in arms:
    t0 = time.time()
    try:
        q = run(arm, False); g = run(arm, True)
    except Exception as ex:
        print(json.dumps(dict(rank=RANK, arm=arm, error=repr(ex)[:300])), flush=True); continue
    rel = [float((b - a).norm() / a.norm()) for a, b in zip(g64, g) if a is not None and b is not None and float(a.norm()) > 0]
    relx = sorted([(n, float((b - a).norm() / a.norm())) for n, a, b in zip(names, g64, g) if a is not None and b is not None and float(a.norm()) > 0 and n != 'e.rho_scale'], key=lambda t: -t[1])
    print(json.dumps(dict(rank=RANK, arm=arm, L15_rel=float('%.3g' % float((q[15] - q64[15]).norm() / q64[15].norm())), mse_rel=float('%.3g' % (mse(q) / mse(q64) - 1)),
                          grad_median=float('%.3g' % float(np.median(rel))), grad_max_excl_rho=float('%.3g' % relx[0][1]), worst=relx[0][0], s=round(time.time() - t0, 1),
                          f64_inputs_seen=F64_SEEN[:12])), flush=True)
