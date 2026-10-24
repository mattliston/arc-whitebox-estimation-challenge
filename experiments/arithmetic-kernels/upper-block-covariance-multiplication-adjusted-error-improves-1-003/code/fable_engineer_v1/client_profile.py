#!/usr/bin/env python3
"""TAG fable-engineer (E-44): where does the CLIENT's own Python time go (the platform's residual = wall - dispatch)? Numpy-free
flopscope client + OpenBLAS flopscope_server; one cold predict under cProfile. Prints per-function self time (tottime) and the time
spent waiting on the server (send_recv), so the residual can be split by function.
usage (client python, FLOPSCOPE_SERVER_URL set): client_profile.py <candidate.py> <rank> <tag>   -> fable_engineer_v1/client_profile_<tag>.txt"""
import sys, time, types, cProfile, pstats, io, json, importlib.util
from pathlib import Path
import flopscope as flops, flopscope.numpy as fnp
cand, rank, tag = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3]
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("cp_candidate", cand); mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator()
fixture = STORE / Path(man[rank]["path"]).parent / "weights.npy"
with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f"cp-{tag}") as ctx:
    blob = fnp.load(str(fixture)); weights = [blob[l] for l in range(16)]
    pr = cProfile.Profile(); t0 = time.monotonic(); c0 = time.process_time()
    pr.enable(); est.predict(MLP(weights, f"cp-{rank}"), 2**41); pr.disable()
    wall = time.monotonic() - t0; cpu = time.process_time() - c0
st = pstats.Stats(pr); st.sort_stats("tottime")
buf = io.StringIO(); st.stream = buf; st.print_stats(45)
cand_name = str(cand); own = {}
for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
    key = name if fn == cand_name else f"[{Path(fn).name}] {name}"
    own[key] = own.get(key, 0.0) + tt
wait = sum(tt for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items() if name in ("recv", "recv_multipart", "poll", "send", "send_multipart"))
head = f"{tag}: wall {wall:.2f} s, client process CPU {cpu:.2f} s, profiled self time {sum(own.values()):.2f} s, zmq send/recv/poll {wait:.2f} s"
lines = [head] + [f"   {t:8.3f} s  {k}" for k, t in sorted(own.items(), key=lambda kv: -kv[1])[:40]]
(HERE / f"client_profile_{tag}.txt").write_text("\n".join(lines) + "\n\n" + buf.getvalue())
print("\n".join(lines[:30]))
