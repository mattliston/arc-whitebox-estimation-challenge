"""Mechanical source-only FlopScope port; all outputs are new, versioned files."""
import ast
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
output = HERE/sys.argv[1]
if output.exists():
    raise FileExistsError(output)
parent = (HERE/"candidate_recovered_cp3_8k_v1.py").read_text()
tree = ast.parse(parent)
constants = {"_CP3_WINOGRAD_LEVELS","_CP3_WINOGRAD_ARENA_ELEMENTS","_CP3_PUSH_ARENA_VIEW_SPECS","_CP3_READ_ARENA_VIEW_SPECS"}
parts = []
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):
        parts.append(ast.get_source_segment(parent,node))
    if isinstance(node,ast.FunctionDef) and node.name in ("_cp3_bind_arena_views","_cp3_winograd_arena_depth4"):
        parts.append(ast.get_source_segment(parent,node))
source = '"""Phase2 radial K3 power-cumulant card. FlopScope numerical operations only."""\nimport flopscope as flops\nimport flopscope.numpy as fnp\nfrom whestbench import BaseEstimator\n\n'+"\n\n".join(parts)+"\n\n"
for filename,names in (("k3_radial_numpy.py",("gaussian_raw","nonlinear")),("radial_features.py",("features",))):
    text = (HERE/filename).read_text()
    for node in ast.parse(text).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            f = ast.get_source_segment(text,node).replace("np.","fnp.").replace("ndtr(a)","flops.stats.norm.cdf(a)")
            f = f.replace("fnp.pi","3.141592653589793")
            source += f+"\n\n"
source += (HERE/"radial_fnp_tail.py").read_text()
if len(sys.argv)>2:
    import numpy as np
    payload = dict(np.load(HERE/sys.argv[2]))
    assert payload["first"][0]==1
    text=(HERE/"radial_gnn_numpy.py").read_text()
    for node in ast.parse(text).body:
        if isinstance(node,ast.FunctionDef):
            source += "\n\n"+ast.get_source_segment(text,node).replace("np.","fnp.")
    source += "\n\n_GNN_LITERAL = "+repr({k:v.tolist() for k,v in payload.items()})+"\n_GNN_PARAMS = None\n"
    source += '''
def _predict_learned(mlp):
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
    return _predict(mlp,hook=hook)

class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        return _predict_learned(mlp)
'''
ast.parse(source)
output.write_text(source)
print(f"built {output} ({len(source)} bytes)")
