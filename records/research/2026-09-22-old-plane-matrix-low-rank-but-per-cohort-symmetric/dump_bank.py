#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-2): dump the deployed 331824 card's signed cube-column bank at every layer of one native
prediction: pp (n x K dense columns after the push), lam (K signed weights), the old/young split (shared_old['columns']), the cohort
table (start,width per birth layer), response/live indices, and the (2,1) plane cp_pre it reads. Offline analysis of the old tier's
compressibility runs on these dumps.  usage: dump_bank.py <candidate.py> <rank>  -> bank_r<rank>/layer_LL.npz"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); rank = int(sys.argv[2]); OUT = HERE / f"bank_r{rank}"; OUT.mkdir(exist_ok=True)
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
src0 = cand.read_text()
H1 = "            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n"
H2 = "                fnp.fill_diagonal(cp_pre, diagonal)\n"
i = src0.rfind(H1); src = src0[:i] + H1 + "            _dump_bank(l, pp, lam, shared_old, response_indices, live_indices, alive)\n" + src0[i + len(H1):]
j = src.rfind(H2); src = src[:j] + H2 + "                _dump_plane(l, cp_pre)\n" + src[j + len(H2):]
H3 = "            K21 = cp_pre + star_pre + fnp.diag(Bpre)[:, None] * bpre[None, :] + 2 * Bpre * bpre[:, None]\n"
k = src.rfind(H3); assert k > 0
src = src[:k] + H3 + "            _dump_state(l, mu, Cpre, star_pre, bpre, Bpre, response_gate, alive, K21, radial)\n" + src[k + len(H3):]
src += '''

def _tonp(a):
    return np.asarray(a.tolist() if hasattr(a, "tolist") else a)
def _dump_bank(l, pp, lam, state, response_indices, live_indices, alive):
    np.savez(str(_OUT / f"layer_{l:02d}.npz"), pp=_tonp(pp).astype(np.float32), lam=_tonp(lam).astype(np.float32),
             old_columns=int(state["columns"]), cohorts=np.asarray(state["cohorts"], dtype=np.int64),
             response=_tonp(response_indices), live=_tonp(live_indices), alive=_tonp(alive))
def _dump_state(l, mu, Cpre, star_pre, bpre, Bpre, gate, alive, K21, radial):
    np.savez(str(_OUT / f"state_{l:02d}.npz"), mu=_tonp(mu).astype(np.float32), Cpre=_tonp(Cpre).astype(np.float32),
             star_pre=_tonp(star_pre).astype(np.float32), bpre=_tonp(bpre).astype(np.float32), Bpre_diag=np.diag(_tonp(Bpre)).astype(np.float32),
             gate=_tonp(gate).astype(np.float32), alive=_tonp(alive), K21=_tonp(K21).astype(np.float32), radial=float(radial))
def _dump_plane(l, cp_pre):
    np.save(str(_OUT / f"plane_{l:02d}.npy"), _tonp(cp_pre).astype(np.float32))
'''
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mod = types.ModuleType("dump_bank_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod; mod.np = np; mod._OUT = OUT
exec(compile(src, str(cand), "exec"), mod.__dict__)
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
t0 = time.time()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    pred = np.asarray(mod.Estimator().predict(MLP([W[l] for l in range(16)], f"dump-{rank}"), 2**41), dtype=np.float64)
np.save(str(OUT / "pred.npy"), pred); np.save(str(OUT / "truth.npy"), truth)
print(json.dumps(dict(rank=rank, util=ctx.summary_dict()["flops_used"] / 2**41, mse=float(((pred[15] - truth[15]) ** 2).mean()), wall=round(time.time() - t0, 1))))
