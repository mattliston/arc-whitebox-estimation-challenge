#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-DF, lever 1): identity-tracking DATAFLOW census of a candidate's products on one net, in process.
Products = the outermost call of any product entry (_mm, _mm_batch pair, _sym_upper, the tree entries, fnp.matmul/dot/einsum/tensordot,
array @). Every non-view result outside products (elementwise, concatenate, astype, fancy getitem, copyto destination) is tagged with its
op and parent tags; views resolve to their root through .base. Per product: entry, caller site, operand shapes and view kind (T or not),
operand provenance (root tag: product / op / input), bill. Per tag: consumers (products and other ops).
Analysis (printed + JSON): (1) association: single-use product outputs consumed by exactly one product through a view; dense volume of
the current vs re-associated order, bill change at 0.50 and 0.56 x 2mkn; (2) shared operands: roots feeding >= 2 products, by side;
(3) long-k provenance (k >= 2048): the producing ops of both operands up to 3 levels (Khatri-Rao check: broadcast multiply / reshape).
usage: dataflow_census.py <cand.py> <rank> <tag>  -> altbasis/dataflow_<tag>.json"""
import sys, json, types, time, inspect, collections
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
from flopscope._ndarray import FlopscopeArray as FA
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank, tag = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("df_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
ON = [False]; INSIDE = [0]; GEN = [0]; TAGS = []; PRODUCTS = []; USES = collections.defaultdict(list); CTX = [None]; RID = [0]
def root(x):
    y = x
    while isinstance(getattr(y, "base", None), np.ndarray):
        if "_df" in getattr(y, "__dict__", {}): return y
        y = y.base
    return y
def tag_of(x):
    """provenance tag of array x (tagging an untagged root as an input)"""
    if not isinstance(x, np.ndarray): return None
    if "_df" in getattr(x, "__dict__", {}): return x._df
    r = root(x)
    if "_df" not in getattr(r, "__dict__", {}):
        try: r._df = len(TAGS); TAGS.append(dict(op="input", shape=list(r.shape), parents=[]))
        except Exception: return None
    return r._df
def view_kind(x):
    if not isinstance(x, np.ndarray) or x.ndim != 2: return "-"
    r = root(x); t = "T" if (x.strides[0] < x.strides[1] and x.shape[0] > 1 and x.shape[1] > 1) else "N"
    return t + ("" if tuple(x.shape) in (tuple(r.shape), tuple(r.shape[::-1])) else "s")
def new_tag(res, op, parents, extra=None):
    outs = res if isinstance(res, (list, tuple)) else [res]
    for o in outs:
        if isinstance(o, np.ndarray):
            try: o._df = len(TAGS); TAGS.append(dict(op=op, shape=list(o.shape), parents=parents, **(extra or {})))
            except Exception: pass
def arrays_in(args, kwargs):
    out = []
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, np.ndarray): out.append(a)
        elif isinstance(a, (list, tuple)): out += [x for x in a if isinstance(x, np.ndarray)] + [y for x in a if isinstance(x, (list, tuple)) for y in x if isinstance(y, np.ndarray)]
    return out
def is_view_of_inputs(res, ins):
    if isinstance(res, (list, tuple)):
        arrs = [r for r in res if isinstance(r, np.ndarray)]
        return bool(arrs) and all(any(root(r) is root(a) for a in ins) for r in arrs)
    if not isinstance(res, np.ndarray): return False
    rr = root(res); return any(rr is root(a) for a in ins)
def site(depth=2):
    f = sys._getframe(depth)
    while f is not None and f.f_code.co_filename == __file__: f = f.f_back
    return f"{f.f_code.co_name}:{f.f_lineno}" if f else "-"
def generic(name, fn):
    def w(*args, **kwargs):
        if not ON[0] or INSIDE[0] or GEN[0]: return fn(*args, **kwargs)
        ins = arrays_in(args, kwargs); GEN[0] += 1
        try: res = fn(*args, **kwargs)
        finally: GEN[0] -= 1
        if name.startswith(".__i") and args and isinstance(args[0], np.ndarray):
            ps = [tag_of(args[0])]
            for a in ins[1:]:
                t = tag_of(a)
                if t is not None: ps.append(t); USES[t].append((name, site()))
            new_tag(args[0], name, ps); return res
        if name in ("copyto",) and len(args) >= 2:
            src = tag_of(args[1]) if isinstance(args[1], np.ndarray) else None
            if src is not None: USES[src].append(("copyto", site()))
            new_tag(args[0], "copyto", [src]); return res
        if is_view_of_inputs(res, ins) and "out" not in kwargs: return res
        ps = []
        for a in ins:
            t = tag_of(a)
            if t is not None: ps.append(t); USES[t].append((name, site()))
        new_tag(res, name, ps, dict(in_shapes=[list(a.shape) for a in ins][:4]))
        return res
    return w
class UProxy:
    def __init__(self, name, orig): self._n, self._o, self._w = name, orig, generic(name, orig)
    def __call__(self, *a, **k): return self._w(*a, **k)
    def __getattr__(self, attr): return getattr(self._o, attr)
PRODUCT_FNP = {"matmul", "dot", "einsum", "tensordot", "inner", "vdot"}
def vol_of(name, args):
    try:
        if name == "einsum":
            subs = args[0].replace(" ", ""); ins, out = subs.split("->") if "->" in subs else (subs, "")
            sizes = {}
            for s, a in zip(ins.split(","), args[1:]):
                for c, d in zip(s, a.shape): sizes[c] = int(d)
            v = 1
            for d in sizes.values(): v *= d
            return 2 * v, None
        a, b = args[0], args[1]
        m, k = (int(a.shape[-2]), int(a.shape[-1])) if a.ndim >= 2 else (1, int(a.shape[-1]))
        n = int(b.shape[-1]) if b.ndim >= 2 else 1
        batch = int(np.prod(a.shape[:-2])) if a.ndim > 2 else 1
        return 2 * batch * m * k * n, (m, k, n)
    except Exception:
        return None, None
def product(name, fn, kind="pair"):
    def w(*args, **kwargs):
        if not ON[0] or INSIDE[0]: return fn(*args, **kwargs)
        where = site(); c = CTX[0]; f0 = c._flops_used
        if kind == "batch":
            pairs = args[0]; ops = [(tag_of(a), tag_of(b), view_kind(a), view_kind(b), list(a.shape), list(b.shape)) for a, b in pairs]
        elif name == "einsum":
            ops = [(None, None, "-", "-", [list(x.shape) for x in args[1:] if isinstance(x, np.ndarray)], args[0])]
            tl = [tag_of(x) for x in args[1:] if isinstance(x, np.ndarray)]
        else:
            a, b = args[0], args[1]; ops = [(tag_of(a), tag_of(b), view_kind(a), view_kind(b), list(a.shape), list(b.shape))]
        INSIDE[0] += 1
        try: res = fn(*args, **kwargs)
        finally: INSIDE[0] -= 1
        bill = int(c._flops_used - f0)
        outs = res if kind == "batch" else [res]
        for i, (ta, tb, va, vb, sa, sb) in enumerate(ops):
            if name == "einsum":
                v, mkn = vol_of(name, args); pid = len(PRODUCTS)
                PRODUCTS.append(dict(id=pid, entry=name, site=where, subs=sb, shapes=sa, tags=tl, bill=bill, vol=v))
                for t in tl:
                    if t is not None: USES[t].append(("P%d" % pid, where))
            else:
                a_ = pairs[i][0] if kind == "batch" else args[0]; b_ = pairs[i][1] if kind == "batch" else args[1]
                v, mkn = vol_of("pair", (a_, b_)); pid = len(PRODUCTS)
                PRODUCTS.append(dict(id=pid, entry=name, site=where, a=ta, b=tb, va=va, vb=vb, sa=sa, sb=sb, mkn=mkn, vol=v,
                                     bill=bill // len(ops), batch=len(ops) if kind == "batch" else 1))
                for t in (ta, tb):
                    if t is not None: USES[t].append(("P%d" % pid, where))
            o = outs[i] if i < len(outs) else None
            if isinstance(o, np.ndarray):
                try: o._df = len(TAGS); TAGS.append(dict(op="product", pid=pid, shape=list(o.shape), parents=[]))
                except Exception: pass
        return res
    return w
# product entries in the candidate (outermost call wins)
for n in ("_mm", "_sym_upper", "_packed_product_cached", "_packed_read5", "_w5_port_product_body", "_cp_d6_shared", "_packed_product_4",
          "_push5_product", "_push6_padded", "_square_product", "_w5_port_product"):
    if callable(getattr(mod, n, None)): setattr(mod, n, product(n, getattr(mod, n)))
if callable(getattr(mod, "_mm_batch", None)): mod._mm_batch = product("_mm_batch", mod._mm_batch, kind="batch")
# flopscope.numpy namespace
for n in dir(fnp):
    if n.startswith("_"): continue
    o = getattr(fnp, n)
    if inspect.isclass(o) or inspect.ismodule(o) or not callable(o): continue
    if n in PRODUCT_FNP: setattr(fnp, n, product(n, o))
    elif inspect.isfunction(o): setattr(fnp, n, generic(n, o))
    else:
        try: setattr(fnp, n, UProxy(n, o))
        except Exception: pass
# array methods / operators
for n in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__neg__", "__pow__",
          "__iadd__", "__isub__", "__imul__", "__itruediv__", "__getitem__", "astype", "copy", "sum", "mean", "max", "min", "dot", "clip"):
    if hasattr(FA, n): setattr(FA, n, generic("." + n, getattr(FA, n)))
for n in ("__matmul__",): setattr(FA, n, product("@", getattr(FA, n)))
for n in ("__rmatmul__",):
    if hasattr(FA, n): setattr(FA, n, product("@r", getattr(FA, n)))
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); est = mod.Estimator(); t0 = time.time()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    CTX[0] = ctx; ON[0] = True
    pred = est.predict(MLP([W[l] for l in range(16)], f"df-{rank}"), 2**41)
    ON[0] = False
total = int(ctx.summary_dict()["flops_used"])
print(f"net r{rank}: bill {total/1e9:.2f} G, {len(PRODUCTS)} products ({sum(p['bill'] for p in PRODUCTS)/1e9:.1f} G), {len(TAGS)} tags, wall {time.time()-t0:.0f} s", flush=True)
json.dump(dict(total=total, products=PRODUCTS, tags=TAGS, uses={str(k): v for k, v in USES.items()}), open(HERE / f"dataflow_{tag}.json", "w"))
print("DONE", tag)
