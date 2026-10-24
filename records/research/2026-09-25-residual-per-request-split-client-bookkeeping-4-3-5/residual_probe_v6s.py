#!/usr/bin/env python3
"""TAG fable-engineer2 copy (E2-RF2 = engineer's residual_probe_v6.py + Estimator.setup(context) called once before the first net, as the platform worker and census_setup.py do; outputs to <outdir>). Original: TAG fable-engineer (E-61 v6): v5 + each GC pass is labelled by WHERE it started: inside a flopscope dispatch span (flopscope's
timed-dispatch 'wrapped' frame is on the stack: the pass is billed to dispatch, not residual) or outside one (the pass is residual).
Per net: gc_ms_in_span, gc_ms_outside. v5 docstring follows.
(E-60 v5): v4 + the objects each GC pass actually collects (cyclic garbage), per generation, and the count of
live tracked objects and frozen objects after each net, to decide whether gc.freeze can strand garbage. v4 docstring follows.
(E-60 v4): v3 with an optional GC policy applied by the HARNESS (the candidate file is untouched), to price
GC tuning before building it: env RP_GC_THRESHOLD=N sets gc.set_threshold(N, 10, 10) before the first net; RP_GC_FREEZE=1 calls
gc.freeze() after each net (so later nets' full collections skip everything alive at that point). v3 docstring follows.
(E-60 v3): v2 + the time Python spends in the cyclic garbage collector (gc.callbacks, per generation) and in the
client's handle finalizers (flopscope._handles.enqueue_free, run when a proxy dies), both part of the platform residual.
v2 docstring: RESIDUAL by function at the production boundary, fine-grained. residual_probe.py wrapped a fixed list;
v2 wraps EVERY function the candidate file defines (module-level, including functions captured under other names are left as they are)
and also counts the flopscope requests (Connection.send_recv) issued while each function is the innermost wrapped frame, so a function's
residual can be split into (requests x per-request client time outside the dispatch span) and its own Python. Exclusive residual per
function = its wall - dispatch accrued inside it - its wrapped callees' inclusive residual - calls x the wrapper's own measured
overhead (calibrated in-process on a wrapped no-op before the nets). Per net: the context residual (as the census reports it), the
wrapped total, and the functions ranked by corrected exclusive residual with calls and requests.
usage: residual_probe_v2.py <cand.py> <ranks> <tag> <outdir>   (client python, FLOPSCOPE_SERVER_URL set) -> <outdir>/residual2_<tag>.json"""
import sys, json, types, importlib.util, time, collections, gc, os
from pathlib import Path
import flopscope as flops, flopscope.numpy as fnp
from flopscope import _dispatch, _connection, _handles
cand, ranks, tag, outdir = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(",")], sys.argv[3], Path(sys.argv[4])
outdir.mkdir(parents=True, exist_ok=True)
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("rp2_candidate", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
incl = collections.Counter(); excl = collections.Counter(); calls = collections.Counter(); reqs = collections.Counter()
stack = []            # [child inclusive residual accumulated, name]
now, disp = time.perf_counter_ns, _dispatch.total_dispatch_ns
def wrap(name, f):
    def w(*a, **k):
        t0, d0 = now(), disp()
        frame = [0, name]
        stack.append(frame)
        try:
            return f(*a, **k)
        finally:
            r = (now() - t0) - (disp() - d0)
            stack.pop()
            incl[name] += r; excl[name] += r - frame[0]; calls[name] += 1
            if stack:
                stack[-1][0] += r
    w.__name__ = getattr(f, "__name__", name); w.__wrapped__ = f
    return w
_orig_send = _connection.Connection.send_recv
def counted(self, raw):
    reqs[stack[-1][1] if stack else None] += 1
    return _orig_send(self, raw)
_connection.Connection.send_recv = counted
gcs = {"n": collections.Counter(), "ns": collections.Counter(), "t0": 0, "collected": collections.Counter(), "uncollectable": collections.Counter(),
       "where": None, "in_ns": 0, "out_ns": 0, "in_n": 0, "out_n": 0}
DISP_FILE = os.path.realpath(_dispatch.__file__)
def _in_span():
    f = sys._getframe(2)
    while f is not None:
        c = f.f_code
        if c.co_name == "wrapped" and os.path.realpath(c.co_filename) == DISP_FILE:
            return True
        f = f.f_back
    return False
def gc_cb(phase, info):
    if phase == "start":
        gcs["where"] = _in_span()
        gcs["t0"] = time.perf_counter_ns()
    else:
        g = info.get("generation", -1)
        dt = time.perf_counter_ns() - gcs["t0"]
        gcs["n"][g] += 1; gcs["ns"][g] += dt
        if gcs["where"]:
            gcs["in_ns"] += dt; gcs["in_n"] += 1
        else:
            gcs["out_ns"] += dt; gcs["out_n"] += 1
        gcs["collected"][g] += int(info.get("collected", 0)); gcs["uncollectable"][g] += int(info.get("uncollectable", 0))
gc.callbacks.append(gc_cb)
fin = {"n": 0, "ns": 0}
_orig_enqueue = _handles.enqueue_free
def enqueue_timed(handle_id):
    t0 = time.perf_counter_ns()
    try:
        return _orig_enqueue(handle_id)
    finally:
        fin["n"] += 1; fin["ns"] += time.perf_counter_ns() - t0
_handles.enqueue_free = enqueue_timed
fname = str(cand)
wrapped = 0
for name, f in list(mod.__dict__.items()):
    if isinstance(f, types.FunctionType) and f.__code__.co_filename == fname and name == f.__name__:
        mod.__dict__[name] = wrap(name, f); wrapped += 1
# calibrate the wrapper's own cost: a wrapped no-op called many times, no requests
def _noop():
    return None
wn = wrap("_calibration_noop", _noop)
t0 = now()
for _ in range(200000):
    wn()
per_call_ns = (now() - t0) / 200000
incl.clear(); excl.clear(); calls.clear(); reqs.clear()
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
GC_T = int(os.environ.get("RP_GC_THRESHOLD", "0")); GC_F = os.environ.get("RP_GC_FREEZE", "0") == "1"
if GC_T:
    gc.set_threshold(GC_T, 10, 10)
est = mod.Estimator(); results = []
if "setup" in type(est).__dict__ or any("setup" in c.__dict__ for c in type(est).__mro__[:-1] if c.__name__ != "BaseEstimator"):
    sctx = types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0)
    _ts = time.time(); est.setup(sctx)
    print(json.dumps(dict(tag=tag, setup_wall_s=round(time.time() - _ts, 3), setup_log=[list(x) for x in getattr(mod, "_PB_LOG", [])])), flush=True)
    incl.clear(); excl.clear(); calls.clear(); reqs.clear()
for rank in ranks:
    incl.clear(); excl.clear(); calls.clear(); reqs.clear(); gcs['n'].clear(); gcs['ns'].clear(); gcs['collected'].clear(); gcs['uncollectable'].clear(); fin['n'] = 0; fin['ns'] = 0
    gcs["in_ns"] = gcs["out_ns"] = gcs["in_n"] = gcs["out_n"] = 0
    with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f"{tag}-rp2-{rank}") as ctx:
        blob = fnp.load(str(STORE / Path(man[rank]["path"]).parent / "weights.npy")); weights = [blob[l] for l in range(16)]
        out = est.predict(MLP(weights, f"rp2-{rank}"), 2**41)
        out._fetch_data()
    s = ctx.summary_dict()
    corr = {k: (v - calls[k] * per_call_ns) / 1e6 for k, v in excl.items()}
    rows = sorted(corr.items(), key=lambda kv: -kv[1])
    rec = dict(rank=rank, residual_s=round(float(s.get("residual_wall_time_s") or -1), 4), wall_s=round(float(s.get("wall_time_s") or -1), 2),
               wrapper_ns_per_call=round(per_call_ns, 1), wrapped_functions=wrapped, total_calls=sum(calls.values()), total_requests=sum(reqs.values()),
               corrected_total_ms=round(sum(corr.values()), 1),
               gc_collections={str(k): v for k, v in gcs['n'].items()}, gc_ms={str(k): round(v / 1e6, 2) for k, v in gcs['ns'].items()},
               finalizers=fin['n'], finalizer_ms=round(fin['ns'] / 1e6, 2), gc_counts_thresholds=list(gc.get_threshold()),
               ranked=[dict(fn=k, ms=round(v, 2), calls=calls[k], requests=reqs.get(k, 0)) for k, v in rows])
    rec["gc_policy"] = dict(threshold=GC_T, freeze_after_net=GC_F)
    rec["gc_ms_in_span"] = round(gcs["in_ns"] / 1e6, 2); rec["gc_ms_outside"] = round(gcs["out_ns"] / 1e6, 2)
    rec["gc_n_in_span"] = gcs["in_n"]; rec["gc_n_outside"] = gcs["out_n"]
    rec["gc_collected"] = {str(k): v for k, v in gcs["collected"].items()}; rec["gc_uncollectable"] = {str(k): v for k, v in gcs["uncollectable"].items()}
    rec["tracked_objects"] = len(gc.get_objects()); rec["frozen_objects"] = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
    print(json.dumps(dict(tag=tag, rank=rank, gc_collected=rec["gc_collected"], gc_uncollectable=rec["gc_uncollectable"], tracked=rec["tracked_objects"], frozen=rec["frozen_objects"])), flush=True)
    results.append(rec)
    if GC_F:
        gc.freeze()
    print(json.dumps(dict(tag=tag, rank=rank, residual_s=rec["residual_s"], corrected_total_ms=rec["corrected_total_ms"], wrapper_ns=rec["wrapper_ns_per_call"],
                          gc_ms=rec["gc_ms"], gc_n=rec["gc_collections"], gc_ms_in_span=rec["gc_ms_in_span"], gc_ms_outside=rec["gc_ms_outside"],
                          gc_n_in_span=rec["gc_n_in_span"], gc_n_outside=rec["gc_n_outside"], finalizers=rec["finalizers"], finalizer_ms=rec["finalizer_ms"],
                          calls=rec["total_calls"], requests=rec["total_requests"], top=[(r["fn"], r["ms"], r["calls"], r["requests"]) for r in rec["ranked"][:15]])), flush=True)
json.dump(results, open(outdir / f"residual6_{tag}.json", "w"), indent=1)
