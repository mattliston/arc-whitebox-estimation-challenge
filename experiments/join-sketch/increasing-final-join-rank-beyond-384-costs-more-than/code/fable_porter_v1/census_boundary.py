#!/usr/bin/env python3
"""TAG fable-porter copy of fable_engineer2_v1/census_e2.py (unchanged logic; output lands in fable_porter_v1/). ORIGINAL: TAG fable-engineer2 (E2-C): request census at the production boundary (numpy-free flopscope client + OpenBLAS server), several nets
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
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
assert importlib.util.find_spec("numpy") is None, "must run numpy-free"
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("census_candidate", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
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
    results.append(rec)
    print(f"{tag}: " + json.dumps({k: v for k, v in rec.items() if k not in ('by_func', 'by_site', 'free_by_func', 'free_by_site')}), flush=True)
json.dump(results, open(HERE / f"{tag}_census.json", "w"), indent=1)
