#!/usr/bin/env python3
"""TAG fable-engineer2 (E-71 group 5, physics-body request diet, PAPER census; v2 rows also carry the server's cumulative flops_used
(per-request bill = consecutive differences) and the full wrapped-function stack, outer/inner): the engineer's request_census_e71.py with (1) full
by-function / by-site lists (no top-N cut) and (2) a per-request TRACE: seq, innermost candidate function, call site, physics layer
(_predict_binary's loop variable l, if on the stack), op, operand summary (handles as h<id>, scalars, shapes of index keys), result
ids / shapes / dtypes. The trace lets body_diet_price.py find re-derived views, repeated identical ops and no-op casts, and the
independent same-op groups. Nothing is timed. usage: body_census.py <cand.py> <ranks> <tag> <outdir>   (client python, server URL set)"""
import sys, json, types, importlib.util, time, collections
from pathlib import Path
import msgpack
import flopscope as flops, flopscope.numpy as fnp
from flopscope import _connection
cand, ranks, tag, outdir = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(",")], sys.argv[3], Path(sys.argv[4])
outdir.mkdir(parents=True, exist_ok=True)
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("bc5_candidate", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
CF = str(cand)
stack = []
def wrap(name, f):
    def w(*a, **k):
        stack.append(name)
        try: return f(*a, **k)
        finally: stack.pop()
    w.__name__ = getattr(f, "__name__", name); w.__wrapped__ = f
    return w
for name, f in list(mod.__dict__.items()):
    if isinstance(f, types.FunctionType) and f.__code__.co_filename == CF and name == f.__name__:
        mod.__dict__[name] = wrap(name, f)
by_op = collections.Counter(); by_fn_op = collections.Counter(); by_site_op = collections.Counter()
frees = dict(trips=0, handles=0); active = [False]; trace = []
def where():
    f = sys._getframe(2); st = None; layer = None
    while f is not None:
        if f.f_code.co_filename == CF:
            if st is None: st = f"{f.f_code.co_name}:{f.f_lineno}"
            if f.f_code.co_name == "_predict_binary" and layer is None:
                v = f.f_locals.get("l")
                layer = v if isinstance(v, int) else None
        f = f.f_back
    return st or "<outside>", layer
def summ(x, depth=0):
    if isinstance(x, dict):
        if "__handle__" in x: return "h" + str(x["__handle__"])
        return {str(k): summ(v, depth + 1) for k, v in x.items()} if depth < 5 else "{..}"
    if isinstance(x, (list, tuple)):
        return [summ(v, depth + 1) for v in x[:64]] if depth < 5 else "[..]"
    if isinstance(x, (bytes, bytearray)): return f"<{len(x)}B>"
    if isinstance(x, float): return float(f"{x:.9g}")
    return x
def res(resp):
    r = resp.get("result") if isinstance(resp, dict) else None
    if isinstance(r, dict) and "id" in r: return [[str(r["id"]), list(r.get("shape", ())), r.get("dtype")]]
    if isinstance(r, dict) and "multi" in r: return [[str(i.get("id")), list(i.get("shape", ())), i.get("dtype")] for i in r["multi"]]
    if isinstance(r, dict) and "value" in r: return [["value", [], r.get("dtype")]]
    return None
_orig_send = _connection.Connection.send_recv
def counted(self, raw):
    resp = _orig_send(self, raw)
    if active[0]:
        try: msg = msgpack.unpackb(raw, raw=False)
        except Exception: msg = {}
        op = msg.get("op", "?")
        fn = stack[-1] if stack else None; st, layer = where()
        by_op[op] += 1; by_fn_op[(fn, op)] += 1; by_site_op[(st, op)] += 1
        b = resp.get("budget") if isinstance(resp, dict) else None
        fu = b.get("flops_used") if isinstance(b, dict) else None
        trace.append([len(trace), fn, st, layer, op, summ(msg.get("args")), summ(msg.get("kwargs")), res(resp), fu, "/".join(stack)])
    return resp
_connection.Connection.send_recv = counted
_orig_free = _connection.encode_free
def free_counted(handles):
    if active[0]: frees["trips"] += 1; frees["handles"] += len(handles)
    return _orig_free(handles)
_connection.encode_free = free_counted
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator()
if any("setup" in k.__dict__ for k in type(est).__mro__[:-1] if k.__name__ != "BaseEstimator"):
    t0 = time.time()
    est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
    print(json.dumps(dict(tag=tag, setup_s=round(time.time() - t0, 2))), flush=True)
results = []
for rank in ranks:
    for c in (by_op, by_fn_op, by_site_op): c.clear()
    frees.update(trips=0, handles=0); trace.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f"{tag}-bc5-{rank}") as ctx:
        active[0] = True
        blob = fnp.load(str(STORE / Path(man[rank]["path"]).parent / "weights.npy")); weights = [blob[l] for l in range(16)]
        out = est.predict(MLP(weights, f"bc5-{rank}"), 2**41)
        out._fetch_data()
        active[0] = False
    s = ctx.summary_dict()
    tot = sum(by_op.values())
    rec = dict(tag=tag, rank=rank, requests=tot, free_trips=frees["trips"], freed_handles=frees["handles"], flops=s.get("flops_used"),
               by_op=by_op.most_common(), by_fn_op=[[f, o, n] for (f, o), n in by_fn_op.most_common()],
               by_site_op=[[st, o, n] for (st, o), n in by_site_op.most_common()])
    results.append(rec)
    with open(outdir / f"body_trace_{tag}_r{rank}.jsonl", "w") as fh:
        for row in trace: fh.write(json.dumps(row) + "\n")
    print(json.dumps(dict(tag=tag, rank=rank, requests=tot, free_trips=frees["trips"], flops=rec["flops"], traced=len(trace))), flush=True)
json.dump(results, open(outdir / f"body_census_{tag}.json", "w"), indent=1)
print("BC5_DONE", tag, flush=True)
