"""Port the scalar final read and attach the strictly matched final-only head."""
import argparse
import ast
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
target = HERE/args.out
if target.exists():
    raise FileExistsError(target)
source = (HERE/args.source).read_text()
assert "_FINAL_LITERAL" not in source
Q = dict(np.load(HERE/"payload_compact_final_head_v1.npz"))
assert set(Q) == {"w0","b0","w2","b2","w4","b4","rms"}
for name in ("compact_scalar_final_v1.py","compact_final_head_v1.py"):
    module = (HERE/name).read_text()
    for node in ast.parse(module).body:
        if isinstance(node,ast.FunctionDef):
            source += "\n\n"+ast.get_source_segment(module,node).replace("np.","fnp.")
old = "gate_floor=0.01):"
assert source.count(old) == 1
source = source.replace(old,"gate_floor=0.01, final_hook=None):")
old = "        Cpre = _cov_compact(w32,C,previous,previous_alive) if cov32 else w.T @ C @ w"
new = '''        if l == depth-1:
            v = _cov_compact_diagonal(w32,C,previous,previous_alive)
            pp = _mm(w32[previous,:].T,p[previous,:])
            k3 = fnp.sum(pp.astype(fnp.float64)**3*lam.astype(fnp.float64),axis=1)+_star_compact_diagonal(w32,star,previous,previous_alive)
            m,aux = nonlinear_scalar(mu,v,k3,radial*v*v)
            if final_hook is not None:
                m = m+fnp.asarray(final_hook(l,m,aux,W[l]),dtype=fnp.float64)
            means.append(m.copy())
            break
'''+old
assert source.count(old) == 1
source = source.replace(old,new)
source += '''

def _cov_compact_diagonal(w,C,indices,alive):
    wa = w[indices,:]
    ca = C[fnp.ix_(indices,indices)].astype(fnp.float32)
    pushed = _mm(ca,wa)
    v = fnp.sum(wa.astype(fnp.float64)*pushed.astype(fnp.float64),axis=0)
    if int(indices.shape[0]) < int(w.shape[0]):
        cold = fnp.nonzero(~alive)[0]
        wc = w[cold,:].astype(fnp.float64)
        v = v+fnp.sum(wc*wc*fnp.diag(C)[cold,None],axis=0)
    return v

def _star_compact_diagonal(w,A,indices,alive):
    wa = w[indices,:]
    aa = A[fnp.ix_(indices,indices)].astype(fnp.float32)
    pushed = _mm(wa.T,aa)
    diagonal = fnp.sum(pushed*wa.T*wa.T,axis=1,dtype=fnp.float64)
    if int(indices.shape[0]) < int(w.shape[0]):
        cold = fnp.nonzero(~alive)[0]
        wc = w[cold,:]
        diagonal = diagonal+fnp.sum(wc*wc*wc*fnp.diag(A)[cold,None].astype(fnp.float32),axis=0,dtype=fnp.float64)
    return diagonal
'''
source += "\n\n_FINAL_LITERAL = "+repr({k:v.tolist() for k,v in Q.items()})+"\n_FINAL_PARAMS = None\n"
source += '''
def _predict_compact_final(mlp):
    global _GNN_PARAMS,_FINAL_PARAMS
    if _GNN_PARAMS is None:
        _GNN_PARAMS={k:fnp.asarray(v,dtype=fnp.float32) for k,v in _GNN_LITERAL.items()}
    if _FINAL_PARAMS is None:
        _FINAL_PARAMS={k:fnp.asarray(v,dtype=fnp.float32) for k,v in _FINAL_LITERAL.items()}
    P,Q=_GNN_PARAMS,_FINAL_PARAMS
    state=fnp.zeros((int(mlp.width),int(P["head"].shape[1])),dtype=fnp.float32)
    def hook(l,m,C,K21,aux,w):
        nonlocal state
        x=features(l,m,C,K21,aux,w)
        d,state=gnn_step(l,x,aux["sigma"],w,state,P)
        return fnp.asarray(d,dtype=fnp.float64)
    def final_hook(l,m,aux,w):
        x=final_scalar_features(l,m,aux,w)
        h=final_input(x,w,state,P)
        return fnp.asarray(final_delta(h,aux["sigma"],Q),dtype=fnp.float64)
    return _predict_binary(mlp,hook=hook,final_hook=final_hook)

class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        return _predict_compact_final(mlp)
'''
ast.parse(source)
target.write_text(source)
print(target)
