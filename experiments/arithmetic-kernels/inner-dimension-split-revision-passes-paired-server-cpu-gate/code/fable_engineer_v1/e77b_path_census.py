#!/usr/bin/env python3
"""TAG fable-engineer (E-77b, size the read6 class for the v2-split extension): in process (.venv), setup, then the net predicted TWICE in
one session; the second (warm) pass is measured so first-time binds do not inflate bills. Each outermost _mm call is logged with the
kernel that actually ran underneath (read6 / ks_split / read5 at depth 5 / square schedule / other), its billed flops and ops.
usage: e77b_path_census.py <cand.py> <rank> <tag>"""
import sys, json, types, collections
from pathlib import Path
import numpy as np, flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand, rank, tag = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3]
CF = str(cand)
mod = types.ModuleType("pc_mod"); mod.__file__ = CF; sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), CF, "exec"), mod.__dict__)
class MLP:
    def __init__(self, w, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, w, 0, nm
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
for c in (mod._IPT_A, mod._IPT_B, mod._IPT_C):
    mod._CAP_BY_ID[id(c)] = 100000                     # warm pass must not rebind evicted keys
CTX = [None]; DEPTH = [0]; PATH = [None]; LOG = []
def layer_now():
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename == CF and f.f_code.co_name == '_predict_binary' and 'l' in f.f_locals:
            return f.f_locals['l']
        f = f.f_back
    return None
def tagger(name, label):
    orig = getattr(mod, name, None)
    if orig is None: return
    def w(*a, **k):
        if DEPTH[0] and PATH[0] is None: PATH[0] = label
        return orig(*a, **k)
    setattr(mod, name, w)
for nm, lab in (('_packed_read6', 'read6'), ('_ks_product', 'ks_split'), ('_ipt_product_ab', 'tree'), ('_square_product', 'square')):
    tagger(nm, lab)
orig_mm = mod._mm
def mm(a, b):
    if DEPTH[0] or CTX[0] is None:
        return orig_mm(a, b)
    DEPTH[0] += 1; PATH[0] = None; f0 = CTX[0].flops_used; o0 = CTX[0]._op_log.__len__() if hasattr(CTX[0], '_op_log') else 0
    try: return orig_mm(a, b)
    finally:
        DEPTH[0] -= 1
        o1 = CTX[0]._op_log.__len__() if hasattr(CTX[0], '_op_log') else 0
        LOG.append(dict(layer=layer_now(), shapes=[list(a.shape), list(b.shape)], path=PATH[0] or 'other', flops=CTX[0].flops_used - f0, ops=o1 - o0))
mod._mm = mm
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
for p in range(2):
    LOG.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        CTX[0] = ctx
        est.predict(MLP([W[l] for l in range(16)], f"pc-{rank}-{p}"), 2**41)
        CTX[0] = None
    tot = ctx.summary_dict()["flops_used"]
json.dump(dict(tag=tag, rank=rank, flops_warm=tot, calls=LOG), open(HERE / f'e77b_paths_{tag}_r{rank}.json', 'w'))
by = collections.Counter(); n = collections.Counter(); ops = collections.Counter()
for c in LOG: by[c['path']] += c['flops']; n[c['path']] += 1; ops[c['path']] += c['ops']
print(json.dumps(dict(tag=tag, warm_net_flops=tot, mm_flops=sum(by.values()), by_path={k: [n[k], round(v / 1e9, 2), ops[k]] for k, v in by.items()})))
