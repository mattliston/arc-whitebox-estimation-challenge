"""TAG fable-engineer (E-68): the loop card in FLOAT32 on MPS, for refits. Same card code as the CPU loop card (the physicist's
s6bcard_loop with the E-67 drop-in: fine_ckpt=False + pair_series loop form), run under two torch function modes:
  - torch.device('mps'): factory functions allocate on MPS;
  - MpsF32Mode: every float64 dtype request becomes float32, float64 results become float32, numpy float64 data becomes float32; in a
    call that mixes devices, CPU tensors with dims move to MPS (MPS has no float64); pure-CPU calls (the card's sha checks) stay on CPU.
Precision fixes certified in E-68a (float32 vs float64: final MSE <= 0.05%, gradients ~3e-3 median):
  - the step's two mean-path products w64.T @ m and w64.T @ b run in FLOAT64 ON CPU (islands; 2 matvecs per layer, ~0.2 s per step);
  - the bank k3 sums sum_j lam_j p_j^3 use the PAIR FORM sum_pairs c (p+ - p-)(p+^2 + p+ p- + p-^2) over the +/- column pairs each
    layer appends (exact in real arithmetic; float32 error 1e-7 vs 2.5e-7 naive), on MPS.
covstep.HookCard.step is patched by AST (2 product sites, 2 k3 sites; asserted) and bound to the card instance.
usage:
    import mps_card as mc
    leaves = mc.to_mps(p), mc.to_mps(a), ...          # float32 MPS copies of the payload (make them the trained leaves)
    cd = mc.card(LC, torch, w, p_m, a_m, c_m, e_m)      # LC = the loop card module; w = numpy weights of the net
    with mc.on_mps(): pred = cd.run(True)               # forward (per-layer checkpoints as the harness)
    loss = ((pred[15].to('cpu').to(torch.float64) - truth) ** 2).mean()     # float64 loss on CPU, OUTSIDE the modes
    with mc.on_mps(): torch.autograd.backward(loss)     # NOT loss.backward(): Tensor.backward is intercepted and the recompute would
                                                        # run without the modes
EVAL MODE (E-70) for research scripts that build their own cards (any covstep.HookCard: s6bcard, s6bcard_joint2, dial modules, hooks):
    import mps_card as mc, mps_lock
    with mps_lock.hold('my grid'):                     # the box's one MPS job (FIFO); wrap the WHOLE MPS part (v1's one-liner
        ...                                            # .__enter__() on a discarded object released at once; mps_lock v2 fixes it)
    pl = mc.to_mps((p, a, c, e))                        # float32 MPS payload
    with mc.on_mps():
        cd = SJ.card(torch, w, *pl); lh.install(cd, torch); ...   # build + installs + your hooks, exactly as on CPU
    mc.bind(cd, w)                                      # the E-68 fixes, with THIS net's weights
    pred = mc.run(cd)                                   # (16, 1024) float64 on CPU; form the MSE on CPU in float64
Grad modes: form the loss outside the modes in float64 on CPU; with per-layer checkpoints call torch.autograd.backward(loss) inside
mc.on_mps() (without checkpoints loss.backward() is fine: the backward then runs no Python).
Gotchas: torch._dynamo must be imported before the modes are entered (this module does it); do not call x.to('cpu', torch.float64)
on an MPS tensor (move first, then cast)."""
import ast, contextlib, inspect, textwrap, types, sys
from pathlib import Path as _Path
for _d in ('fable_engineer_v1', 'fable_covhunter_v1', 'fable_analyst_v1'):   # covstep (the HookCard) + pair_series_torch, whatever the caller's path
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in sys.path: sys.path.append(_p)
import numpy as np
import torch, torch._dynamo   # dynamo first: checkpoint imports it lazily, which fails under torch.device('mps')
from torch.overrides import TorchFunctionMode
import pair_series_torch as ps
import covstep

F64, F32, DEV = torch.float64, torch.float32, torch.device('mps')
FACTORY = {torch.tensor, torch.as_tensor}
ISLAND = [0]          # > 0 while an island runs: the mode passes calls through (also on the autograd worker thread)
STATS = dict(moves=0)


def _devs(x, acc):
    if isinstance(x, torch.Tensor): acc.add(x.device.type)
    elif type(x) in (tuple, list):
        for y in x: _devs(y, acc)
    return acc


def _fix(x, mixed):   # only CPU->MPS moves cast; other float64 inputs pass (casting them would also rewrite x.dtype reads)
    if isinstance(x, torch.Tensor):
        if mixed and x.device.type == 'cpu' and x.dim() > 0:
            STATS['moves'] += 1
            return x.to(DEV, F32 if x.dtype == F64 else x.dtype)
        return x
    if type(x) in (tuple, list): return type(x)(_fix(y, mixed) for y in x)
    return x


def _out(x):          # exact tuple/list/dict only: torch.Size stays a Size
    if isinstance(x, torch.Tensor): return x.to(F32) if x.dtype == F64 else x
    if type(x) is tuple: return tuple(_out(y) for y in x)
    if type(x) is list: return [_out(y) for y in x]
    if type(x) is dict: return {k: _out(v) for k, v in x.items()}
    return x


class MpsF32Mode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if ISLAND[0]: return func(*args, **(kwargs or {}))
        kwargs = dict(kwargs or {})
        if func is torch.Tensor.numpy and args and isinstance(args[0], torch.Tensor) and args[0].device.type == 'mps':
            return args[0].detach().cpu().numpy()        # E-70: research code calls .numpy() on card tensors
        if func is torch.Tensor.double:                  # E-70: x.double() (e.g. gn3_knobs' kink mask) -> float32
            return args[0].float()
        if func is torch.Tensor.new_tensor and kwargs.get('dtype') is None and len(args) > 1 and isinstance(args[1], np.ndarray) and args[1].dtype == np.float64:
            kwargs['dtype'] = F32
        gen = kwargs.get('generator')
        if isinstance(gen, torch.Generator) and gen.device.type == 'cpu':
            kwargs.setdefault('device', torch.device('cpu'))   # E-70: seeded CPU draws stay on CPU in the REQUESTED dtype (the same
            return _out(func(*args, **kwargs))                 # numbers as the CPU run), then become float32 (and move on mixed use)
        if kwargs.get('dtype') is F64: kwargs['dtype'] = F32
        if func in FACTORY and kwargs.get('dtype') is None and args and isinstance(args[0], np.ndarray) and args[0].dtype == np.float64:
            kwargs['dtype'] = F32
        args = tuple(F32 if a is F64 else a for a in args)
        mixed = len(_devs(args, _devs(list(kwargs.values()), set()))) > 1
        if mixed: args, kwargs = _fix(args, True), {k: _fix(v, True) for k, v in kwargs.items()}
        return _out(func(*args, **kwargs))


@contextlib.contextmanager
def on_mps():
    with torch.device(DEV), MpsF32Mode():
        yield


N2 = 1024 * 1024


def _make_isl_mm(W64):
    """w64.T @ x in float64 on CPU with THIS card's weights (W64 = the net's numpy weights as CPU float64, bound per card; v1 cached
    by MPS storage address and served a previous net's weights when the allocator reused the address: mps_card_v1_W64cache_bug.py.bak)."""
    def _isl_mm(wt, x):   # wt is a view of the card's (16,n,n) weights: its storage offset gives the layer
        ISLAND[0] += 1
        try:
            return (W64[wt.storage_offset() // N2].T @ x.to('cpu').to(F64)).to(F32).to(DEV)
        finally:
            ISLAND[0] -= 1
    return _isl_mm


_PAIRS = {}


def _k3_pair(pp, lam, stops):   # sum_j lam_j pp_ij^3 over the +/- column pairs (runs under the mode: all float32 on MPS)
    key = (tuple(stops), int(pp.shape[1]))
    if key not in _PAIRS:
        pos, neg, lo = [], [], 0
        for hi in stops:
            n = (hi - lo) // 2
            assert hi - lo == 2 * n
            pos += range(lo, lo + n); neg += range(lo + n, hi); lo = hi
        assert lo == pp.shape[1], (lo, pp.shape)
        if len(_PAIRS) > 64: _PAIRS.clear()
        _PAIRS[key] = (torch.tensor(pos, device=pp.device), torch.tensor(neg, device=pp.device))
    pos, neg = _PAIRS[key]
    P, M = pp[:, pos], pp[:, neg]
    return (((P - M) * (P * P + P * M + M * M)) * lam[pos]).sum(1)


class _Inline(ast.NodeTransformer):
    K3 = ast.unparse(ast.parse('(p64 * p64 * p64) * lam.to(t.float64)').body[0].value)

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult) and isinstance(node.left, ast.Attribute) and node.left.attr == 'T' \
                and isinstance(node.left.value, ast.Name) and node.left.value.id == 'w64' \
                and isinstance(node.right, ast.Name) and node.right.id in ('m', 'b'):
            return ast.copy_location(ast.Call(func=ast.Name('_isl_mm', ast.Load()), args=[node.left, node.right], keywords=[]), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'sum' and ast.unparse(node.func.value) == self.K3:
            return ast.copy_location(ast.Call(func=ast.Name('_k3_pair', ast.Load()),
                                              args=[ast.Name(n, ast.Load()) for n in ('pp', 'lam', 'stops')], keywords=[]), node)
        return node


_tree = _Inline().visit(ast.parse(textwrap.dedent(inspect.getsource(covstep.HookCard.step))))
ast.fix_missing_locations(_tree)
_u = ast.unparse(_tree)
assert _u.count('_isl_mm(') == 2 and _u.count('_k3_pair(') == 2, 'covstep.HookCard.step changed: re-check the island sites'
_CODE = compile(_tree, 'covstep_step_mps', 'exec')


def _step_for(W64):   # the patched step with this card's island (a fresh globals dict per card)
    g = dict(covstep.__dict__); g.update(_isl_mm=_make_isl_mm(W64), _k3_pair=_k3_pair)
    exec(_CODE, g)
    return g['step']


def to_mps(x):
    """float32 MPS copies of a payload piece: a tensor, a dict of tensors, or a tuple/list of those (e.g. the (p, a, c, e) payload)."""
    if isinstance(x, dict): return {k: to_mps(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return type(x)(to_mps(v) for v in x)
    return x.detach().to(DEV, F32).clone()


def bind(cd, w):
    """E-70 eval drop-in: give an ALREADY-BUILT card (a covstep.HookCard built under on_mps() with MPS float32 params, hooks installed
    before or after) the E-68 precision fixes: the patched step with this net's float64 CPU mean-path islands and the pair-form k3."""
    ISLAND[0] += 1                    # bind() may be called inside on_mps(): the mode must not turn these float64 CPU weights into MPS float32
    try:
        W64 = torch.as_tensor(np.asarray(w), dtype=F64, device='cpu')
    finally:
        ISLAND[0] -= 1
    assert W64.device.type == 'cpu' and W64.dtype == F64
    assert tuple(W64.shape) == (16, 1024, 1024) and cd.w.device.type == 'mps', 'build the card under mps_card.on_mps() with to_mps params'
    assert cd.w.is_contiguous() and cd.ws.untyped_storage().data_ptr() == cd.w.untyped_storage().data_ptr()
    assert type(cd).step is covstep.HookCard.step, 'the card overrides HookCard.step: the island patch would not apply'
    cd.step = types.MethodType(_step_for(W64), cd)
    return cd


def run(cd):
    """E-70 eval drop-in: no-grad forward of a bound MPS card; returns the (16, 1024) post means as float64 on CPU."""
    with torch.no_grad(), on_mps():
        out = cd.run(False)
    return out.to('cpu').to(F64)


def card(LC, torch_, w, p, a, c, e):
    """the loop card (LC.card) on MPS in float32 with the E-67 drop-in and the E-68 precision fixes; p/a/c/e must be MPS float32."""
    with on_mps():
        cd = ps.install(LC.card(torch_, w, p, a, c, e, fine_ckpt=False), torch_)
    return bind(cd, w)
