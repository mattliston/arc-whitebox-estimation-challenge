#!/usr/bin/env python3
"""TAG fable-engineer (E-68a islands v3 = v2 + the k3 PAIR FORM arm 'k3pair' (float32: sum over +/- column pairs of c (p+ - p-)(p+^2 + p+ p- + p-^2),
no float64) + a log of every unit selection (select/select_floor: layer, slot, count, ids) compared with the float64 card):
which parts of the loop card need float64? Card = the physicist's s6bcard_loop (LOOP_DIALMOD,
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
        with torch._C.DisableTorchFunction():             # the casts too: under the mode .to(float64) is rewritten to float32
            out = fn(*conv(args, F32, F64), **conv(kw, F32, F64))
        return conv(out, F64, F32)
    return wrapped
import ast, inspect, textwrap, types, covstep
def _isl_mm(wt, x):                                         # inline island: w64.T @ x in float64
    with torch._C.DisableTorchFunction():
        return (wt.to(F64) @ x.to(F64)).to(F32)
def _isl_sum1(x):                                           # inline island: ((p64*p64*p64) * lam64).sum(1) in float64
    return x
class _Inline(ast.NodeTransformer):
    def __init__(self, which): self.which = which
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if 'mu' in self.which and isinstance(node.op, ast.MatMult) and isinstance(node.left, ast.Attribute) and node.left.attr == 'T' \
                and isinstance(node.left.value, ast.Name) and node.left.value.id == 'w64' and isinstance(node.right, ast.Name) and node.right.id in ('m', 'b'):
            return ast.copy_location(ast.Call(func=ast.Name('_isl_mm', ast.Load()), args=[node.left, node.right], keywords=[]), node)
        return node
    def visit_Call(self, node):
        self.generic_visit(node)
        if ('k3sum' in self.which or 'k3pair' in self.which) and isinstance(node.func, ast.Attribute) and node.func.attr == 'sum' and ast.unparse(node.func.value) == ast.unparse(ast.parse('(p64 * p64 * p64) * lam.to(t.float64)').body[0].value):
            fn = '_isl_k3' if 'k3sum' in self.which else '_k3_pair'
            return ast.copy_location(ast.Call(func=ast.Name(fn, ast.Load()), args=[ast.Name('pp', ast.Load()), ast.Name('lam', ast.Load()), ast.Name('stops', ast.Load())], keywords=[]), node)
        return node
def _isl_k3(pp, lam, stops=None):
    with torch._C.DisableTorchFunction():
        p = pp.to(F64); return ((p * p * p) * lam.to(F64)).sum(1).to(F32)
_PAIRS = {}
def _k3_pair(pp, lam, stops):                               # float32 pair form (runs under the mode; all float32)
    key = (tuple(stops), int(pp.shape[1]))
    if key not in _PAIRS:
        pos, neg, lo = [], [], 0
        for hi in stops:
            n = (hi - lo) // 2; pos += range(lo, lo + n); neg += range(lo + n, hi); lo = hi
        assert lo == pp.shape[1]
        _PAIRS[key] = (torch.tensor(pos), torch.tensor(neg))
    pos, neg = _PAIRS[key]
    P, M = pp[:, pos], pp[:, neg]
    return (((P - M) * (P * P + P * M + M * M)) * lam[pos]).sum(1)
def patched_step(which):
    src = textwrap.dedent(inspect.getsource(covstep.HookCard.step))
    tree = _Inline(which).visit(ast.parse(src)); ast.fix_missing_locations(tree)
    g = dict(covstep.__dict__); g.update(_isl_mm=_isl_mm, _isl_k3=_isl_k3, _k3_pair=_k3_pair)
    exec(compile(tree, 'covstep_step_islands', 'exec'), g)
    n_mm, n_k3 = ast.unparse(tree).count('_isl_mm('), ast.unparse(tree).count('_isl_k3(') + ast.unparse(tree).count('_k3_pair(')
    return g['step'], n_mm, n_k3
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
SEL = []
def logged(fn, cd):
    def wrapped(*a, **k):
        ids, alive = fn(*a, **k)
        if not torch.is_grad_enabled(): SEL.append((cd.call_layer, len(SEL), int(ids.numel()), tuple(ids.tolist())))
        return ids, alive
    return wrapped
def build(arm):
    if arm == 'f64':
        cd = ps.install(LC.card(torch, w, p64, a64, c64, e64, fine_ckpt=False), torch)
        cd.select, cd.select_floor = logged(cd.select, cd), logged(cd.select_floor, cd)
        return cd
    cd = ps.install(LC.card(torch, w, p32, a32, c32, e32, fine_ckpt=False), torch)
    isl = [] if arm == 'f32' else (ENV + ['shared.' + k for k in SHARED] + ['hook.' + k for k in cd.hooks] if arm.startswith('all_parts') else arm.split('+'))
    inline = [k for k in ('mu', 'k3sum', 'k3pair') if k in arm.split('+')] + (['mu', 'k3sum'] if arm == 'all_parts+inline' else [])
    if inline:
        fn, n_mm, n_k3 = patched_step(inline); cd.step = types.MethodType(fn, cd); INLINE_COUNTS[arm] = (n_mm, n_k3)
    isl = [k for k in isl if k not in ('mu', 'k3sum', 'k3pair', 'all_parts', 'inline')]
    cd.select, cd.select_floor = logged(cd.select, cd), logged(cd.select_floor, cd)
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
INLINE_COUNTS = {}
SEL.clear(); q64 = run('f64', False); SEL64 = list(SEL); g64 = run('f64', True)
mse = lambda q: float(((q[15] - tp) ** 2).mean())
arms = sys.argv[3].split(',') if len(sys.argv) > 3 else ['f32', 'mu', 'k3sum', 'mu+k3sum', '_nonlinear_pair_batch', '_bd_update', '_flms_star', 'shared.push', 'shared.read',
       'gnn_step', 'features', 'hook.cov_pm', 'hooks', 'all_parts', 'all_parts+inline']
for arm in arms:
    t0 = time.time()
    try:
        SEL.clear(); q = run(arm, False); sel = list(SEL); g = run(arm, True)
        seldiff = [(a[0], a[2], b[2], len(set(a[3]) ^ set(b[3]))) for a, b in zip(SEL64, sel) if a[3] != b[3]]
    except Exception as ex:
        print(json.dumps(dict(rank=RANK, arm=arm, error=repr(ex)[:300])), flush=True); continue
    rel = [float((b - a).norm() / a.norm()) for a, b in zip(g64, g) if a is not None and b is not None and float(a.norm()) > 0]
    relx = sorted([(n, float((b - a).norm() / a.norm())) for n, a, b in zip(names, g64, g) if a is not None and b is not None and float(a.norm()) > 0 and n != 'e.rho_scale'], key=lambda t: -t[1])
    print(json.dumps(dict(rank=RANK, arm=arm, L15_rel=float('%.3g' % float((q[15] - q64[15]).norm() / q64[15].norm())), mse_rel=float('%.3g' % (mse(q) / mse(q64) - 1)),
                          grad_median=float('%.3g' % float(np.median(rel))), grad_max_excl_rho=float('%.3g' % relx[0][1]), worst=relx[0][0], s=round(time.time() - t0, 1), inline_sites=INLINE_COUNTS.get(arm), selections=len(sel), selection_diffs=seldiff[:6], n_selection_diffs=len(seldiff))), flush=True)
