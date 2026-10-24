"""Use the audited direct-write D6/4096 arena for compact rectangular products.

Only product padding and arithmetic schedule change. Live/birth masks and
all learned parameters stay frozen. Square products retain the D3 kernel.
"""
import argparse
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
target = HERE / args.out
if target.exists():
    raise FileExistsError(target)
source = (HERE / args.source).read_text()
d6 = (HERE / "candidate_binary_learned_d6_direct_v1.py").read_text()
fns = {n.name:ast.get_source_segment(d6,n) for n in ast.parse(d6).body if isinstance(n,ast.FunctionDef)}
assigns = {}
for node in ast.parse(d6).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        assigns[node.targets[0].id] = ast.get_source_segment(d6,node)
parts = []
replaced = []
for node in ast.parse(source).body:
    part = ast.get_source_segment(source,node)
    if isinstance(node,ast.FunctionDef):
        if node.name == "_mm":
            continue
        if node.name in ("_split_views_5", "_packed_product_5"):
            part = fns[node.name.replace("_5","_6")]
            replaced.append(node.name)
    elif isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        name = node.targets[0].id
        if name in ("_PACKED_ELEMENTS","_PACKED_PUSH","_PACKED_READ","_PACKED_SQUARE","_PACKED_TILE"):
            part = assigns[name]
            replaced.append(name)
    parts.append(part)
assert len(replaced) == 7, replaced
parts.append('''
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
    mp=((m+63)//64)*64;kp=((k+63)//64)*64;npad=((n+63)//64)*64
    if mp!=m or kp!=k:
        aa=fnp.zeros((mp,kp),dtype=fnp.float32)
        fnp.copyto(aa[:m,:k],a)
    else:aa=a
    if kp!=k or npad!=n:
        bb=fnp.zeros((kp,npad),dtype=fnp.float32)
        fnp.copyto(bb[:k,:n],b)
    else:bb=b
    out=fnp.empty((mp,npad),dtype=fnp.float32)
    out=_packed_product_6(aa,bb,out,_SCRATCH[1] if n>1024 else _SCRATCH[2])
    return out[:m,:n]
''')
result = "\n\n".join(parts)
ast.parse(result)
target.write_text(result)
print(target)
