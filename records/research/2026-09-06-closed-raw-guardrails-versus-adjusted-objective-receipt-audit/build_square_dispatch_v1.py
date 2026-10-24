"""Choose exact square-product schedules from independently measured costs.

No change to masks, moment formulas, learned parameters, or rectangular
products. All new views reuse the existing fixed arena. Dispatch inspects
array shapes only; all numerical model computation remains metered FNP.
"""
import argparse
import ast
import json
from pathlib import Path
from packed_arena_generic import layout

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument("--source",required=True)
ap.add_argument("--out",required=True)
args=ap.parse_args()
target=HERE/args.out
if target.exists():
    raise FileExistsError(target)
source=(HERE/args.source).read_text()
tree=ast.parse(source)
names=("_split_views_6","_packed_product_6","_mm")
functions={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
assert set(functions)==set(names)
prices=json.loads((HERE/"direct_kernel_schedule_prices_v1.json").read_text())
for depth in (4,5):
    split=functions["_split_views_6"].replace("_split_views_6",f"_split_views_{depth}")
    class DepthConstants(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value==6:
                return ast.copy_location(ast.Constant(depth),node)
            return node
    split=ast.unparse(ast.fix_missing_locations(DepthConstants().visit(ast.parse(split))))
    kernel=functions["_packed_product_6"].replace("_packed_product_6",f"_packed_product_{depth}").replace("_split_views_6",f"_split_views_{depth}").replace("levels = 6",f"levels = {depth}")
    source+="\n\n"+split+"\n\n"+kernel
arena_size=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_PACKED_ELEMENTS" for t in n.targets))
for depth in (4,5,6):
    size,specs,_=layout(1024,1024,1024,depth)
    assert size<=arena_size,(depth,size,arena_size)
    source+=f"\n\n_PACKED_SQUARE{depth} = "+repr(specs)
old="_SCRATCH = (arena, _cp3_bind_arena_views(arena, _PACKED_PUSH), _cp3_bind_arena_views(arena, _PACKED_READ), _cp3_bind_arena_views(arena, _PACKED_SQUARE))"
new=old[:-1]+", _cp3_bind_arena_views(arena, _PACKED_SQUARE4), _cp3_bind_arena_views(arena, _PACKED_SQUARE5), _cp3_bind_arena_views(arena, _PACKED_SQUARE6))"
assert source.count(old)==1
source=source.replace(old,new)
coefficients=tuple((int(d),tuple(prices["coefficients"][str(d)])) for d in (3,4,5,6))
source+="\n\n_SQUARE_COSTS = "+repr(coefficients)+"\n"
source+='''

def _square_schedule(m,k,n):
    chosen=0
    best=2*m*k*n-m*n
    dims=(m,k,n)
    for depth,coeff in _SQUARE_COSTS:
        unit=2**depth
        mp=(m+unit-1)//unit*unit
        kp=(k+unit-1)//unit*unit
        npad=(n+unit-1)//unit*unit
        cost=coeff[0]*mp*kp*npad+coeff[1]*mp*kp+coeff[2]*kp*npad+coeff[3]*mp*npad
        if mp!=m or kp!=k or npad!=n:
            cost=cost+2*(mp*kp+kp*npad)
        if cost<best:
            chosen=depth;best=cost;dims=(mp,kp,npad)
    return chosen,dims

def _square_product(a,b,m,k,n):
    depth,(mp,kp,npad)=_square_schedule(m,k,n)
    if depth==0:
        return a@b
    if mp!=m or kp!=k:
        aa=fnp.zeros((mp,kp),dtype=fnp.float32)
        fnp.copyto(aa[:m,:k],a)
    else:
        aa=a
    if kp!=k or npad!=n:
        bb=fnp.zeros((kp,npad),dtype=fnp.float32)
        fnp.copyto(bb[:k,:n],b)
    else:
        bb=b
    out=fnp.empty((mp,npad),dtype=fnp.float32)
    if depth==3:
        result=_packed_product_3(aa,bb,out,_SCRATCH[3])
    elif depth==4:
        result=_packed_product_4(aa,bb,out,_SCRATCH[4])
    elif depth==5:
        result=_packed_product_5(aa,bb,out,_SCRATCH[5])
    else:
        result=_packed_product_6(aa,bb,out,_SCRATCH[6])
    return result[:m,:n]
'''
dispatcher=functions["_mm"]
start=dispatcher.index("    if max(m, k, n) <= 1024:")
end=dispatcher.index("    mp =",start)
dispatcher=dispatcher[:start]+"    if max(m,k,n)<=1024:\n        return _square_product(a,b,m,k,n)\n"+dispatcher[end:]
source+="\n\n"+dispatcher+"\n"
ast.parse(source)
target.write_text(source)
print(target)
