"""Metered compact K4-to-K3 host, reusing its active covariance push."""
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
edits={
    "    result=_mm(wa.T,_mm(ca,wa))":
        "    result=_mm(wa.T,_mm(ca,wa))\n    active_metric=result.astype(fnp.float64)",
    "    return result.astype(fnp.float64)":
        "    return result.astype(fnp.float64),active_metric",
    "    radial = 0.0":
        "    radial = 0.0\n    b = fnp.zeros(n,dtype=fnp.float64)",
    "        Cpre = _cov_compact(w32,C,previous,previous_alive) if cov32 else w.T @ C @ w":
        "        if cov32:\n            Cpre,Bpre = _cov_compact(w32,C,previous,previous_alive)\n        else:\n            Cpre=w.T@C@w\n            Bpre=w.T@(C*previous_alive[:,None]*previous_alive[None,:])@w\n        bpre=w.T@b",
    "        K21 = cp_pre+star_pre":
        "        K21 = cp_pre+star_pre+fnp.diag(Bpre)[:,None]*bpre[None,:]+2*Bpre*bpre[:,None]",
    '        birth = aux["pdf"]/aux["sigma"]':
        '        d = aux["pdf"]/aux["sigma"]\n        birth = d*(1+radial/3)\n        b = (g*bpre+radial*d*vv/6)*alive',
    "        residual = target-inherited":
        "        Bpost=C.copy()\n        fnp.fill_diagonal(Bpost,fnp.diag(C)*alive)\n        trace_post=fnp.diag(Bpost)[:,None]*b[None,:]+2*Bpost*b[:,None]\n        residual = target-inherited-trace_post",
    "        rest = target-cp_post":
        "        rest = target-cp_post-trace_post",
}
for old,new in edits.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
source+='''

class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        return _predict_binary(mlp)
'''
ast.parse(source)
target.write_text(source)
print(target)
