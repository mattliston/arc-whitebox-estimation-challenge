#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-CP): where the participant RESIDUAL goes. One prediction under cProfile in the numpy-free client (server
separate); reports flopscope's own residual for the run and the candidate-file functions ranked by their own Python time (tottime,
callees excluded), plus the flopscope client functions' own time. usage (client python, FLOPSCOPE_SERVER_URL set):
client_prof.py <cand.py> <rank> <tag>  -> fable_engineer2_v1/client_prof_<tag>.txt"""
import cProfile, pstats, io, json, sys, types, importlib.util
from pathlib import Path
import flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank, tag = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("prof_candidate", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator()
fixture = STORE / Path(man[rank]["path"]).parent / "weights.npy"
prof = cProfile.Profile()
with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f"{tag}-prof") as ctx:
    blob = fnp.load(str(fixture)); weights = [blob[l] for l in range(16)]
    prof.enable(); out = est.predict(MLP(weights, f"prof-{rank}"), 2**41); raw = out._fetch_data(); prof.disable()
s = ctx.summary_dict()
st = pstats.Stats(prof)
rows_c, rows_f = [], []
for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
    if fn == str(cand): rows_c.append((tt, nc, f"{name}:{ln}"))
    elif "flopscope" in fn: rows_f.append((tt, nc, f"{fn.split('flopscope/')[-1]}:{name}"))
rows_c.sort(reverse=True); rows_f.sort(reverse=True)
lines = [f"flopscope residual_wall_time_s {s.get('residual_wall_time_s')}  (profiling inflates Python time)",
         f"candidate own Python time {sum(r[0] for r in rows_c):.3f} s; flopscope client own Python time {sum(r[0] for r in rows_f):.3f} s",
         "top candidate functions (own time s, calls, function:line):"] + [f"  {t:7.4f} {n:7d} {f}" for t, n, f in rows_c[:30]] + \
        ["top flopscope client functions (own time s, calls):"] + [f"  {t:7.4f} {n:7d} {f}" for t, n, f in rows_f[:20]]
(HERE / f"client_prof_{tag}.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
