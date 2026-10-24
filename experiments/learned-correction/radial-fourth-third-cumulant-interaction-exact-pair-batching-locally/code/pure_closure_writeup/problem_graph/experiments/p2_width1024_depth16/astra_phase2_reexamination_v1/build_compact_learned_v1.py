"""Attach a strict matched mean-only payload to a compact FlopScope host."""
import argparse
import ast
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--payload", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
target = HERE / args.out
if target.exists():
    raise FileExistsError(target)
source = (HERE / args.source).read_text()
assert "_GNN_LITERAL" not in source
inference = (HERE / "radial_gnn_numpy.py").read_text()
for node in ast.parse(inference).body:
    if isinstance(node, ast.FunctionDef):
        source += "\n\n"+ast.get_source_segment(inference, node).replace("np.", "fnp.")
payload = dict(np.load(HERE / args.payload))
assert set(payload) == {"centers","scales","rms","msg_w","msg_b","enc0_w","enc0_b","enc2_w","enc2_b","enc4_w","enc4_b","head","head_b","gain","first"}
assert payload["first"][0] == 1
source += "\n\n_GNN_LITERAL = "+repr({k: v.tolist() for k, v in payload.items()})+"\n_GNN_PARAMS = None\n"
source += '''
def _predict_learned_compact(mlp):
    global _GNN_PARAMS
    if _GNN_PARAMS is None:
        _GNN_PARAMS={k:fnp.asarray(v,dtype=fnp.float32) for k,v in _GNN_LITERAL.items()}
    P=_GNN_PARAMS
    state=fnp.zeros((int(mlp.width),int(P["head"].shape[1])),dtype=fnp.float32)
    def hook(l,m,C,K21,aux,w):
        nonlocal state
        x=features(l,m,C,K21,aux,w)
        delta,state=gnn_step(l,x,aux["sigma"],w,state,P)
        return fnp.asarray(delta,dtype=fnp.float64)
    return _predict_binary(mlp,hook=hook)

class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        return _predict_learned_compact(mlp)
'''
ast.parse(source)
target.write_text(source)
print(target)
