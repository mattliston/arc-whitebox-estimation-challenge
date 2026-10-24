"""Metered39-feature compact host, without any learned payload attached."""
import argparse
import ast
from pathlib import Path

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument("--out",required=True)
args=ap.parse_args()
target=HERE/args.out
if target.exists():
    raise FileExistsError(target)
parent=(HERE/"candidate_binary_compact_g01_learned_d6_v1.py").read_text()
parts=[]
for node in ast.parse(parent).body:
    if isinstance(node,ast.ClassDef):
        continue
    if isinstance(node,ast.FunctionDef) and node.name in ("gnn_step","silu","_predict_learned_compact"):
        continue
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith("_GNN") for t in node.targets):
        continue
    parts.append(ast.get_source_segment(parent,node))
source="\n\n".join(parts)
reference=(HERE/"compact_tensor_directions_v1.py").read_text()
for node in ast.parse(reference).body:
    if isinstance(node,ast.FunctionDef):
        source+="\n\n"+ast.get_source_segment(reference,node).replace("np.","fnp.")
old="        host = m.copy()"
assert source.count(old)==1
source=source.replace(old,'        if hook is not None:\n            aux["tensor_directions"]=direction_features(mu,Cpre,pp,lam,W[l],star)\n'+old)
source+='''

_features30=features

def features(l,m,C,K21,aux,W):
    return fnp.concatenate([_features30(l,m,C,K21,aux,W),aux["tensor_directions"]],axis=1)

class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        def null_hook(l,m,C,K21,aux,W):
            features(l,m,C,K21,aux,W)
            return fnp.zeros_like(m)
        return _predict_binary(mlp,hook=null_hook)
'''
ast.parse(source)
target.write_text(source)
print(target)
