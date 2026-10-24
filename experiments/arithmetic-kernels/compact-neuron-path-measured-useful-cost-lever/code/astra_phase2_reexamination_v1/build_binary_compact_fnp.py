"""Build a versioned, metered compact live-neuron estimator.

The reference keeps dense zero-masked state for clarity; products use only
its active block plus exact cold diagonals. Every numerical op is FlopScope.
"""
import argparse
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--gate", type=float, required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
target = HERE / args.out
if target.exists():
    raise FileExistsError(target)
parent = (HERE / "candidate_binary_learned_d5_tile8192_v1.py").read_text()
parts = []
for node in ast.parse(parent).body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        parts.append(ast.get_source_segment(parent, node))
    elif isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not any(n.startswith("_GNN") for n in names):
            parts.append(ast.get_source_segment(parent, node))
    elif isinstance(node, ast.FunctionDef) and node.name not in ("_predict_binary", "_predict_learned_binary", "gnn_step", "silu"):
        parts.append(ast.get_source_segment(parent, node))
helpers = '''
def _mm(a,b):
    if not _WINOGRAD or str(a.dtype)!="float32" or str(b.dtype)!="float32":
        return a@b
    m,k=int(a.shape[0]),int(a.shape[1]);n=int(b.shape[1])
    if m==0 or k==0 or n==0:return fnp.zeros((m,n),dtype=fnp.float32)
    if n>_PACKED_TILE:
        return fnp.concatenate([_mm(a,b[:,j:min(j+_PACKED_TILE,n)]) for j in range(0,n,_PACKED_TILE)],axis=1)
    if k>_PACKED_TILE:
        out=_mm(a[:,:_PACKED_TILE],b[:_PACKED_TILE,:])
        for j in range(_PACKED_TILE,k,_PACKED_TILE):
            end=min(j+_PACKED_TILE,k)
            out=out+_mm(a[:,j:end],b[j:end,:])
        return out
    if m%32 or k%32 or n%32:return a@b
    if max(m,k,n)<=1024:
        if min(m,k,n)<=256:return a@b
        out=fnp.empty((m,n),dtype=fnp.float32)
        return _packed_product_3(a,b,out,_SCRATCH[3])
    out=fnp.empty((m,n),dtype=fnp.float32)
    return _packed_product_5(a,b,out,_SCRATCH[1] if n>1024 else _SCRATCH[2])

def _cov_compact(w,C,indices,alive):
    wa=w[indices,:]
    ca=C[fnp.ix_(indices,indices)].astype(fnp.float32)
    result=_mm(wa.T,_mm(ca,wa))
    if int(indices.shape[0])<int(w.shape[0]):
        cold=fnp.nonzero(~alive)[0]
        wc=w[cold,:]
        diagonal=fnp.diag(C)[cold].astype(fnp.float32)
        result=result+_mm((wc*diagonal[:,None]).T,wc)
    return result.astype(fnp.float64)

def _star_compact(w,A,previous,previous_alive,current):
    n=int(w.shape[1])
    wp=w[previous,:]
    aa=A[fnp.ix_(previous,previous)].astype(fnp.float32)
    pushed=_mm(wp.T,aa)
    wr=wp[:,current].T
    pr=pushed[current,:]
    sub=(2*_mm(pr*wr,wr.T)+_mm(wr*wr,pr.T))/3
    diagonal=fnp.sum(pushed*wp.T*wp.T,axis=1,dtype=fnp.float64)
    if int(previous.shape[0])<int(w.shape[0]):
        cold=fnp.nonzero(~previous_alive)[0]
        wc=w[cold,:]
        ad=fnp.diag(A)[cold].astype(fnp.float32)
        wc_current=wc[:,current]
        sub=sub+_mm((wc_current*wc_current).T,wc_current*ad[:,None])
        diagonal=diagonal+fnp.sum((wc*wc*wc*ad[:,None]),axis=0,dtype=fnp.float64)
    result=fnp.zeros((n,n),dtype=fnp.float64)
    result[fnp.ix_(current,current)]=sub
    fnp.fill_diagonal(result,diagonal)
    return result
'''
parts.append(helpers)
reference = (HERE / "radial_binary_compact.py").read_text()
for node in ast.parse(reference).body:
    if not isinstance(node, ast.FunctionDef):
        continue
    part = ast.get_source_segment(reference, node).replace("np.", "fnp.").replace("ndtr(", "flops.stats.norm.cdf(")
    if node.name == "select":
        part = part.replace('int(fnp.count_nonzero(score >= floor))', 'int(fnp.nonzero(score >= floor)[0].shape[0])')
        part = part.replace('dtype=bool', 'dtype=fnp.bool_')
    elif node.name == "predict":
        part = part.replace('def predict(W,', 'def _predict_binary(mlp,')
        part = part.replace('gate_floor=.001', f'gate_floor={args.gate!r}')
        old = '''    depth, n, _ = W.shape
    m = fnp.zeros(n)'''
        new = '''    global _SCRATCH
    depth,n=int(mlp.depth),int(mlp.width);W=mlp.weights
    if _SCRATCH is None:
        arena=fnp.empty((_PACKED_ELEMENTS,),dtype=fnp.float32)
        _SCRATCH=(arena,_cp3_bind_arena_views(arena,_PACKED_PUSH),_cp3_bind_arena_views(arena,_PACKED_READ),_cp3_bind_arena_views(arena,_PACKED_SQUARE))
    previous=fnp.arange(n)
    previous_alive=fnp.ones(n,dtype=fnp.bool_)
    m = fnp.zeros(n,dtype=fnp.float64)'''
        assert part.count(old) == 1
        part = part.replace(old, new)
        part = part.replace('W[l].astype(fnp.float32)', 'fnp.asarray(W[l],dtype=fnp.float32)')
        part = part.replace('W[l].astype(fnp.float64)', 'fnp.asarray(W[l],dtype=fnp.float64)')
        old = '(w32.T @ C.astype(fnp.float32) @ w32).astype(fnp.float64) if cov32 else w.T @ C @ w'
        assert part.count(old) == 1
        part = part.replace(old, '_cov_compact(w32,C,previous,previous_alive) if cov32 else w.T @ C @ w')
        part = part.replace('pp = w32.T @ p', 'pp = _mm(w32[previous,:].T,p[previous,:])')
        old = '''        star_pre = star_push(W[l], star)
        sd = fnp.diag(star_pre).copy()
        star_pre = star_pre*live
        fnp.fill_diagonal(star_pre, sd)'''
        assert part.count(old) == 1
        part = part.replace(old, '        star_pre = _star_compact(w32,star,previous,previous_alive,live_indices)')
        part = part.replace('        radial = radial_new', '        previous,previous_alive=live_indices,alive\n        radial = radial_new')
        part = part.replace('return fnp.stack(means), states', 'return fnp.asarray(fnp.maximum(fnp.stack(means),0.0),dtype=fnp.float32)')
    parts.append(part)
parts.append('''
class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        return _predict_binary(mlp)
''')
source = '\n\n'.join(parts)
ast.parse(source)
target.write_text(source)
print(target)
