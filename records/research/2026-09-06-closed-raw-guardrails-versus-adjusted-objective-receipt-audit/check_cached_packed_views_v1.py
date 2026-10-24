"""Same arithmetic and exact output on cold/warm, mixed-shape arena reuse."""
import ast
import json
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from packed_arena_generic import layout
from cached_packed_views_v1 import _packed_product_cached,_PACKED_FRAME_CACHE

HERE=Path(__file__).resolve().parent
out=HERE/"cached_packed_views_certificate_v1.json"
assert not out.exists()
source=(HERE/"candidate_fullpair_runtime_d5_t8192_v1.py").read_text()
lines=source.splitlines(keepends=True)
functions={node.name:"".join(lines[node.lineno-1:node.end_lineno])
           for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
env=dict(fnp=fnp)
for name in ("_cp3_bind_arena_views","_split_views_3","_split_views_4","_split_views_5",
             "_packed_product_3","_packed_product_4","_packed_product_5"):
    exec(functions[name],env)
rng=np.random.default_rng(385917)
records=[]
for depth in (3,4,5):
    size,specs,_=layout(128,128,128,depth)
    arena=fnp.empty(size,dtype=fnp.float32)
    views=env["_cp3_bind_arena_views"](arena,specs)
    for m,k,n in ((128,128,128),(96,128,64),(64,96,128),(128,128,128)):
        a=fnp.asarray(rng.normal(size=(m,k)).astype(np.float32))
        b=fnp.asarray(rng.normal(size=(k,n)).astype(np.float32))
        expected=fnp.empty((m,n),dtype=fnp.float32)
        actual=fnp.empty((m,n),dtype=fnp.float32)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ref:
            env[f"_packed_product_{depth}"](a,b,expected,views)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as got:
            _packed_product_cached(a,b,actual,views,depth)
        assert np.array_equal(np.asarray(expected),np.asarray(actual)),(depth,m,k,n)
        assert ref.flops_used==got.flops_used,(depth,ref.flops_used,got.flops_used)
        records.append(dict(depth=depth,shape=(m,k,n),byte_exact=True,flops=int(got.flops_used)))
    _PACKED_FRAME_CACHE.clear()
out.write_text(json.dumps(dict(records=records,scope="Native exact kernel parity and equal FLOPs; no whole-estimator runtime claim."),indent=2)+"\n")
print(out.read_text(),flush=True)
