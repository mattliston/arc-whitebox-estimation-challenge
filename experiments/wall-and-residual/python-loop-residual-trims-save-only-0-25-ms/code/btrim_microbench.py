#!/usr/bin/env python3
"""TAG fable-engineer2 E2-BT micro-bench: Python-side cost of the tree-kernel loop bodies, S23 vs --btrim, with the flopscope ops,
side lookup and loaders replaced by identical stubs in both modules (so the difference is only the loop bytecode). Also times the
_rule_depth / _ipt_fits call cost (memo hit vs S23 chain) on the same stubs. usage: btrim_microbench.py <S23.py> <btrim.py>"""
import sys, time, types, importlib.util, statistics


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class V:   # stand-in view (weakref-able like the client proxy)
    __slots__ = ('__weakref__',)


def op(x, y, out=None):
    return out


def entry(L):
    T = V()
    chunks = [None] + [tuple(V() for _ in range(7)) for _ in range(L)]
    basis = [(V(), V()) for _ in range(L - 1)]
    return (T, chunks, V(), None, basis)


def patch(mod, entries):
    mod.fnp = types.SimpleNamespace(add=op, subtract=op, matmul=op)
    mod._ipt_side = lambda kind, L, r, c: entries[(kind, L)]
    mod._rs_load_a = lambda *a: None
    mod._rs_load_b = lambda *a: None
    mod._rs_out_quadrants = lambda *a: (V(), V(), V(), V())


def product(mod, L):
    TA = mod._ipt_forward_a_ab(None, L, 1024, 1024)
    TB = mod._ipt_forward_b_ab(None, L, 1024, 1024)
    mod._ipt_leaf_unwind_ab(TA, TB, None, L, 1024, 1024)
    mod._BH_HOLD.clear()


def bench(fn, reps=2000):
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        fn()
    return (time.perf_counter_ns() - t0) / reps / 1e3   # us per call


old, new = load(sys.argv[1], 's23'), load(sys.argv[2], 'btrim')
entries = {(kind, L): entry(L) for kind in range(3) for L in (2, 3, 4, 5, 6)}
patch(old, entries); patch(new, entries)
for L in (4, 5, 6):
    ops = (L - 1) + 3 * L + (L - 1) + 3 * L + 1 + 6 * (L - 1) + (L - 1) + 7
    o, n = [], []
    for _ in range(15):
        o.append(bench(lambda: product(old, L))); n.append(bench(lambda: product(new, L)))
    mo, mn = statistics.median(o), statistics.median(n)
    print(f"L={L}: {ops} ops/product | S23 {mo:6.2f} us, btrim {mn:6.2f} us per product | saving {mo - mn:5.2f} us ({1e3 * (mo - mn) / ops:4.0f} ns/op)")
shapes = [('push', 1024, 1024, 4096), ('read', 1024, 1024, 1024), ('push', 256, 1024, 2048), ('read', 512, 1024, 1024)]
for kind, m, k, n in shapes:
    old._rule_depth(kind, m, k, n); new._rule_depth(kind, m, k, n)
    o = statistics.median(bench(lambda: old._rule_depth(kind, m, k, n), 20000) for _ in range(9))
    w = statistics.median(bench(lambda: new._rule_depth(kind, m, k, n), 20000) for _ in range(9))
    assert old._rule_depth(kind, m, k, n) == new._rule_depth(kind, m, k, n)
    print(f"_rule_depth{(kind, m, k, n)}: S23 {o:5.2f} us, btrim {w:5.2f} us per call")
arena = types.SimpleNamespace(shape=(old._PACKED_ELEMENTS,))
old._SCRATCH = [arena]; new._SCRATCH = [arena]
for L, m, k, n in ((5, 1024, 1024, 1024), (6, 1024, 1024, 4096), (4, 256, 1024, 1024)):
    assert old._ipt_fits(L, m, k, n) == new._ipt_fits(L, m, k, n)
    o = statistics.median(bench(lambda: old._ipt_fits(L, m, k, n), 20000) for _ in range(9))
    w = statistics.median(bench(lambda: new._ipt_fits(L, m, k, n), 20000) for _ in range(9))
    print(f"_ipt_fits{(L, m, k, n)} = {new._ipt_fits(L, m, k, n)}: S23 {o:5.2f} us, btrim {w:5.2f} us per call")
