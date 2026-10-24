#!/usr/bin/env python3
"""TAG fable-porter v2 (PT-44, lead 22:2x "fail hard on non-finite outputs"): + per net n_out and nonfinite from the fetched prediction
(_fetch_data() = (bytes, shape, dtype); decoded with the array module, no numpy). Base: TAG fable-porter census_boundary.py + Estimator.setup(context) called once before the first net when the candidate defines one
(the engineer's census_setup.py block verbatim, E-61: as the platform worker, no budget context around it; wall, requests and _PB_LOG printed).
Arms without a setup method skip it, so one client is fair to both arms. Base: TAG fable-porter copy of fable_engineer2_v1/census_e2.py (unchanged logic; output lands in fable_porter_v1/). ORIGINAL: TAG fable-engineer2 (E2-C): request census at the production boundary (numpy-free flopscope client + OpenBLAS server), several nets
in ONE process (first = cold, later = warm). Per net: requests, per-function counts, client wall, residual (participant Python =
wall - dispatch, as the platform bills it), util.  usage (client python, FLOPSCOPE_SERVER_URL set): census_e2.py <cand.py> <ranks> <tag>
-> fable_engineer2_v1/<tag>_census.json"""
import collections, importlib.util, json, sys, types, time
from pathlib import Path
import flopscope as flops, flopscope.numpy as fnp
import os, subprocess
SERVER_PID = os.environ.get('E2_SERVER_PID')
def server_cpu_s():
    """Cumulative user+sys CPU seconds of the server process (contention-robust, unlike wall)."""
    if not SERVER_PID:
        return float('nan')
    out = subprocess.run(['ps', '-o', 'cputime=', '-p', SERVER_PID], capture_output=True, text=True).stdout.strip()
    parts = [float(x) for x in out.replace('-', ':').split(':')]
    sec = 0.0
    for x in parts:
        sec = sec * 60 + x
    return sec
from flopscope import _connection
STUDY = Path(__file__).resolve().parent; HERE = STUDY.parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
assert importlib.util.find_spec("numpy") is None, "must run numpy-free"
cand, ranks, tag = Path(sys.argv[1]).resolve(), ([] if sys.argv[2]=='-' else [int(x) for x in sys.argv[2].split(',')]), sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("census_candidate", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
mode = sys.argv[4]
assert mode == 'phase1_3s'
phase1_record = {}
def phase1_only(t0):
    saved_deadline = mod._PB_WALL_S
    assert saved_deadline == 4.0
    # Test-only3s: original t0/order, restored before outer fixed setup tails.
    mod._PB_WALL_S = 3.0
    mod._LVF_KEEP[0] = True
    try:
        mod._LVF_OLD_PB_BIND(t0)
    finally:
        mod._LVF_KEEP[0] = False
        mod._PB_WALL_S = saved_deadline
        phase1_record.update(requested_phase1_budget_s=3.0,restored_deadline_s=mod._PB_WALL_S,
            phase1_bound_roster=[list(k) for k in mod._PB_SINGLE
                if dict.get((mod._IPT_A,mod._IPT_B,mod._IPT_C)[k[0]],k[1:]) is not None],
            phase1_fused_roster=[list(k) for k in mod._PB_SINGLE
                if (dict.get((mod._IPT_A,mod._IPT_B,mod._IPT_C)[k[0]],k[1:]) or (None,None,None,None))[3] is not None])
        mod._LVF_PENDING.clear()
        mod._PB_LOG.append(('forced_phase2_unavailable',0,round(mod._pb_time.monotonic()-t0,3)))
        mod._BH_HOLD.clear()
mod._KSF_INNER = phase1_only
by_site = collections.Counter(); by_func = collections.Counter(); total = [0]; active = [False]
cand_name = str(cand)
_orig = _connection.Connection.send_recv
def counted(self, raw_request):
    if active[0]:
        total[0] += 1
        f = sys._getframe(1); site = None
        while f is not None:
            if f.f_code.co_filename == cand_name:
                site = (f.f_lineno, f.f_code.co_name); break
            f = f.f_back
        by_site[site] += 1; by_func[site[1] if site else None] += 1
    return _orig(self, raw_request)
_connection.Connection.send_recv = counted
# Deferred frees of GC'd server arrays ride a SEPARATE hidden round trip (a `free` op sent just before the next request, inside
# send_recv, bypassing it): count those trips too, since each is a platform ZMQ round trip like a request.
frees = [0, 0]
_orig_drain = _connection.drain_pending
free_by_func = collections.Counter()
def counted_drain():
    got = _orig_drain()
    if active[0] and got:
        frees[0] += 1; frees[1] += len(got)
        f = sys._getframe(1); site = None
        while f is not None:
            if f.f_code.co_filename == cand_name:
                site = f.f_code.co_name; break
            f = f.f_back
        free_by_func[site] += 1
        f = sys._getframe(1)
        while f is not None:
            if f.f_code.co_filename == cand_name:
                free_by_site[f"{f.f_lineno} {f.f_code.co_name}"] += 1; break
            f = f.f_back
    return got
free_by_site = collections.Counter()
_connection.drain_pending = counted_drain
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); results = []
if "setup" in type(est).__dict__ or any("setup" in c.__dict__ for c in type(est).__mro__[:-1] if c.__name__ != "BaseEstimator"):
    sctx = types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0)
    total[0] = 0; active[0] = True; t_s = time.time()
    est.setup(sctx)
    active[0] = False
    print(f"{tag}: " + json.dumps(dict(setup_wall_s=round(time.time() - t_s, 3), setup_requests=total[0], setup_log=[list(x) for x in getattr(mod, "_PB_LOG", [])])), flush=True)
import hashlib
_essential = ((0, 3, 1024, 1024), (1, 3, 1024, 32), (2, 3, 1024, 32), (0, 3, 32, 1024), (1, 3, 1024, 1024), (2, 3, 32, 1024), (0, 3, 1024, 32), (1, 3, 32, 1024), (2, 3, 1024, 1024))
_coverage = [dict(key=k, bound=dict.get((mod._IPT_A,mod._IPT_B,mod._IPT_C)[k[0]],tuple(k[1:])) is not None,
    fused=(dict.get((mod._IPT_A,mod._IPT_B,mod._IPT_C)[k[0]],tuple(k[1:])) or (None,None,None,None))[3] is not None) for k in _essential]
_setup = dict(candidate=str(cand),candidate_sha256=hashlib.sha256(cand.read_bytes()).hexdigest(),mode=mode,
    setup_wall_s=time.time()-t_s,setup_requests=total[0],setup_log=[list(x) for x in mod._PB_LOG],
    phase1_deadline=mod._PB_WALL_S,essential=_coverage,phase1_simulation=phase1_record,
    bound_entries=sum(len(cache) for cache in (mod._IPT_A,mod._IPT_B,mod._IPT_C)),
    fused_entries=sum(e[3] is not None for cache in (mod._IPT_A,mod._IPT_B,mod._IPT_C) for e in cache.values()))
with (HERE/f'{tag}_setup.json').open('x') as _fp: json.dump(_setup,_fp,indent=2)
assert mod._PB_WALL_S==4.0 and phase1_record['restored_deadline_s']==4.0
KERNEL = ("_packed_product_cached", "_packed_frame", "_exact_view", "_a_frame", "_b_frame", "_o_frame")
for rank in ranks:
    fixture = STORE / Path(man[rank]["path"]).parent / "weights.npy" if rank < 1000 else P2 / "public_g19_attribution_v1" / f"fresh_{rank}_weights.npy"
    by_site.clear(); by_func.clear(); free_by_func.clear(); free_by_site.clear(); total[0] = 0; frees[0] = frees[1] = 0
    with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f"{tag}-census-{rank}") as ctx:
        blob = fnp.load(str(fixture)); weights = [blob[l] for l in range(16)]
        f0 = int(ctx.summary_dict()["flops_used"]); t0 = time.time(); sc0 = server_cpu_s(); cc0 = time.process_time()
        active[0] = True
        out = est.predict(MLP(weights, f"census-{rank}"), 2**41)
        raw = out._fetch_data()
        active[0] = False
        wall = time.time() - t0; f1 = int(ctx.summary_dict()["flops_used"]); sc1 = server_cpu_s(); cc1 = time.process_time()
    s = ctx.summary_dict()
    kern = sum(v for k, v in by_func.items() if k in KERNEL)
    rec = dict(rank=rank, requests=total[0], free_trips=frees[0], freed_handles=frees[1], round_trips=total[0] + frees[0], server_cpu_s=round(sc1 - sc0, 2), client_cpu_s=round(cc1 - cc0, 2), kernel_and_frames=kern, outside=total[0] - kern, wall_s=round(wall, 2),
               residual_s=round(float(s.get("residual_wall_time_s") or -1), 4), util=round((f1 - f0) / 2**41, 6),
               flops=f1 - f0, flops_ctx=int(s["flops_used"]), by_func=dict(by_func.most_common()), free_by_func={str(k): v for k, v in free_by_func.most_common()}, free_by_site=dict(free_by_site.most_common()),
               by_site={f"{k[0]} {k[1]}" if k else "-": v for k, v in by_site.items()})
    import array as _arr, math as _math
    _data, _shape, _dt = raw
    _a = _arr.array('f' if '32' in str(_dt) else 'd'); _a.frombytes(_data)
    rec['n_out'] = len(_a); rec['nonfinite'] = sum(1 for _x in _a if not _math.isfinite(_x))
    rec['prediction_sha256']=hashlib.sha256(_data).hexdigest(); rec['prediction_shape']=list(_shape); rec['prediction_dtype']=str(_dt)
    with (HERE/f'{tag}_pred_{rank}.bin').open('xb') as _fp: _fp.write(_data)
    results.append(rec)
    print(f"{tag}: " + json.dumps({k: v for k, v in rec.items() if k not in ('by_func', 'by_site', 'free_by_func', 'free_by_site')}), flush=True)
json.dump(results, open(HERE / f"{tag}_census.json", "w"), indent=1)
