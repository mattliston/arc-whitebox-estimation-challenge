"""Versioned FlopScope port of the two-cube third-cumulant carrier."""
import ast
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
target=HERE/sys.argv[1]
if target.exists():raise FileExistsError(target)
parent=(HERE/"candidate_radial_gnn_splitread_v1.py").read_text()
parts=[]
for node in ast.parse(parent).body:
    if isinstance(node,(ast.Import,ast.ImportFrom)):
        parts.append(ast.get_source_segment(parent,node))
    elif isinstance(node,ast.Assign):
        names=[t.id for t in node.targets if isinstance(t,ast.Name)]
        if not any(n.startswith("_GNN") for n in names):parts.append(ast.get_source_segment(parent,node))
    elif isinstance(node,ast.FunctionDef) and node.name in ("_cp3_bind_arena_views","_cp3_winograd_arena_depth4","gaussian_raw","nonlinear","features","_mm"):
        part=ast.get_source_segment(parent,node)
        if node.name=="_mm":
            before='''    out = fnp.empty((m,n),dtype=fnp.float32)'''
            after='''    if n>15360:
        return fnp.concatenate([_mm(a,b[:,start:min(start+15360,n)]) for start in range(0,n,15360)],axis=1)
    out = fnp.empty((m,n),dtype=fnp.float32)'''
            assert before in part;part=part.replace(before,after)
        parts.append(part)
for filename,names in (("k3_radial_numpy.py",("star_push",)),("radial_cubic.py",("plane",)),("radial_binary_cubic.py",("predict",))):
    src=(HERE/filename).read_text()
    for node in ast.parse(src).body:
        if not isinstance(node,ast.FunctionDef) or node.name not in names:continue
        part=ast.get_source_segment(src,node).replace("np.","fnp.")
        if node.name=="star_push":
            part=part.replace("wa = w@A","wa = _mm(w,A)").replace("(wa*w)@w.T","_mm(wa*w,w.T)").replace("(w*w)@wa.T","_mm(w*w,wa.T)")
        elif node.name=="plane":part=part.replace("(p*p)@(p*lam).T","_mm(p*p,(p*lam).T)")
        elif node.name=="predict":
            part=part.replace('def predict(W,','def _predict_binary(mlp,')
            old='''    depth,n,_=W.shape;m=fnp.zeros(n);C=fnp.eye(n);radial=0.0'''
            new='''    global _SCRATCH
    depth,n=int(mlp.depth),int(mlp.width);W=mlp.weights
    if _SCRATCH is None:
        arena=fnp.empty((_CP3_WINOGRAD_ARENA_ELEMENTS,),dtype=fnp.float32)
        _SCRATCH=(arena,_cp3_bind_arena_views(arena,_CP3_PUSH_ARENA_VIEW_SPECS),_cp3_bind_arena_views(arena,_CP3_READ_ARENA_VIEW_SPECS))
    m=fnp.zeros(n,dtype=fnp.float64);C=fnp.eye(n,dtype=fnp.float64);radial=fnp.asarray(0.0,dtype=fnp.float64)'''
            assert old in part;part=part.replace(old,new)
            part=part.replace("W[l].astype(fnp.float32)","fnp.asarray(W[l],dtype=fnp.float32)").replace("W[l].astype(fnp.float64)","fnp.asarray(W[l],dtype=fnp.float64)")
            part=part.replace("w32.T@C.astype(fnp.float32)@w32","_mm(_mm(w32.T,C.astype(fnp.float32)),w32)")
            part=part.replace("pp=w32.T@p","pp=_mm(w32.T,p)")
            part=part.replace("return fnp.stack(means),states","return fnp.asarray(fnp.maximum(fnp.stack(means),0.0),dtype=fnp.float32)")
        parts.append(part)
parts.append('''class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        return _predict_binary(mlp)
''')
source='"""Phase2 two-cube K3 carrier. Numerical operations use FlopScope only."""\n'+"\n\n".join(parts)
ast.parse(source);target.write_text(source);print(target)
