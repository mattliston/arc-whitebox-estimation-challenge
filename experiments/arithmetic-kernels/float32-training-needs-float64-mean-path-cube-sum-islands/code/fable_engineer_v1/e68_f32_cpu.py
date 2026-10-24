#!/usr/bin/env python3
"""TAG fable-engineer (E-68a): float32 vs float64 on the loop card, CPU. Card = the physicist's s6bcard_loop (LOOP_DIALMOD, default
s6bcard_jfitR1 = the live round-1 joint2 dials) with the E-67 drop-in (fine_ckpt=False + pair_series loop form), run(True) for grads.
The float32 arm runs the SAME code under F32Mode, a TorchFunctionMode that rewrites every float64 dtype request to float32 and casts any
float64 result to float32 (it counts float64 tensors arriving as inputs, which must stay 0), with float32 copies of the payload as leaves.
Per net: no-grad predictions (per-layer relative diff of the post means, L15), final MSE of each arm vs truth (loss in float64 outside the
mode), then one grad step per arm: per-tensor relative gradient diff (norm), median, max. Saves both arms' predictions to e68_preds_r<rank>.npz.
usage (python with torch): e68_f32_cpu.py <threads> <rank>[,<rank>...]"""
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
torch.set_num_threads(int(sys.argv[1])); RANKS = [int(r) for r in sys.argv[2].split(',')]
F64, F32 = torch.float64, torch.float32
def cast(x):
    if isinstance(x, torch.Tensor): return x.to(F32) if x.dtype == F64 else x
    if isinstance(x, tuple): return tuple(cast(y) for y in x)
    if isinstance(x, list): return [cast(y) for y in x]
    if isinstance(x, dict): return {k: cast(v) for k, v in x.items()}
    return x
def has64(x):
    if isinstance(x, torch.Tensor): return x.dtype == F64
    if isinstance(x, (tuple, list)): return any(has64(y) for y in x)
    if isinstance(x, dict): return any(has64(v) for v in x.values())
    return False
class F32Mode(TorchFunctionMode):
    f64_inputs = 0
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if kwargs.get('dtype') is F64: kwargs['dtype'] = F32
        args = tuple(F32 if a is F64 else a for a in args)
        if has64(args) or has64(kwargs): F32Mode.f64_inputs += 1
        return cast(func(*args, **kwargs))
p64, a64, c64, e64 = LC.payload(torch)
def leaves(p, a, c, e):
    return [p['head'], p['head_b'], e['output'], e['bias'], a, c] + [e[k] for k in e if k not in ('output', 'bias')] + [p[k] for k in ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')]
names = ['head', 'head_b', 'e.output', 'e.bias', 'a', 'c'] + ['e.' + k for k in e64 if k not in ('output', 'bias')] + ['msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b']
p32 = {k: v.detach().to(F32).clone() for k, v in p64.items()}; a32, c32 = a64.detach().to(F32).clone(), c64.detach().to(F32).clone()
e32 = {k: v.detach().to(F32).clone() for k, v in e64.items()}
L64, L32 = leaves(p64, a64, c64, e64), leaves(p32, a32, c32, e32)
for x in L64 + L32: x.requires_grad_(True)
def build(f32, w):
    cd = LC.card(torch, w, *( (p32, a32, c32, e32) if f32 else (p64, a64, c64, e64) ), fine_ckpt=False)
    return ps.install(cd, torch)
def predict(f32, w):
    with torch.no_grad():
        if f32:
            with F32Mode(): out = build(True, w).run(False)
        else: out = build(False, w).run(False)
    return out.to(F64)
def grads(f32, w, tp):
    L = L32 if f32 else L64
    for x in L: x.grad = None
    t0 = time.time()
    if f32:
        with F32Mode(): pred = build(True, w).run(True)
        loss = ((pred[15].to(F64) - tp) ** 2).mean()          # float64 loss formed OUTSIDE the mode
        with F32Mode(): torch.autograd.backward(loss)          # recompute runs under the mode (Tensor.backward is intercepted: the
                                                                # mode would pop itself and the recompute would come back float64)
    else:
        pred = build(False, w).run(True); loss = ((pred[15].to(F64) - tp) ** 2).mean(); loss.backward()
    return float(loss), [x.grad.detach().to(F64).clone() if x.grad is not None else None for x in L], time.time() - t0
for rank in RANKS:
    w = LC.DM.weights(rank)
    tp = torch.tensor(np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64))
    q64, q32 = predict(False, w), predict(True, w)
    np.savez(HERE / f'e68_preds_r{rank:04d}.npz', f64=q64.numpy(), f32=q32.numpy())
    per_layer = [float((q32[l] - q64[l]).norm() / q64[l].norm()) for l in range(16)]
    m64, m32 = float(((q64[15] - tp) ** 2).mean()), float(((q32[15] - tp) ** 2).mean())
    print(json.dumps(dict(rank=rank, dialmod=os.environ['LOOP_DIALMOD'], f64_inputs_in_mode=F32Mode.f64_inputs, mse_f64=m64, mse_f32=m32, mse_rel_change=m32 / m64 - 1,
                          pred_rel_diff_L15=per_layer[15], pred_rel_diff_by_layer=[float('%.3g' % x) for x in per_layer])), flush=True)
    l64, g64, t64 = grads(False, w, tp); l32, g32, t32 = grads(True, w, tp)
    rows = []
    for n, x64, x32 in zip(names, g64, g32):
        if x64 is None or x32 is None: continue
        rows.append((n, float((x32 - x64).norm() / x64.norm()) if float(x64.norm()) > 0 else float('nan'), float(x64.norm())))
    rel = [r[1] for r in rows]
    print(json.dumps(dict(rank=rank, loss_rel_change=l32 / l64 - 1, grad_rel_median=float(np.nanmedian(rel)), grad_rel_max=float(np.nanmax(rel)),
                          worst=sorted(rows, key=lambda r: -r[1])[:4], f64_inputs_in_mode=F32Mode.f64_inputs, step_s_f64=round(t64, 1), step_s_f32=round(t32, 1))), flush=True)
