"""Small exact, equal-cost certificate before a remote full-estimator test."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from packed_arena_generic import layout

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--tag",required=True)
args=ap.parse_args()
target=HERE/f"{args.tag}.json"
assert not target.exists()
def definitions(path):
    source=path.read_text();lines=source.splitlines(True)
    return {n.name:"".join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
parent=definitions(HERE/"candidate_fullpair_runtime_d5_t8192_v1.py")
candidate=definitions(HERE/args.candidate)
env=dict(fnp=fnp,_PACKED_FRAME_CACHE={})
for name in ("_cp3_bind_arena_views","_split_views_3","_split_views_4","_split_views_5","_packed_product_3","_packed_product_4","_packed_product_5"):
    exec(parent[name],env)
for name in ("_packed_frame","_packed_product_cached"):
    exec(candidate[name],env)
rng=np.random.default_rng(385917);records=[]
for depth in (3,4,5):
    size,specs,_=layout(128,128,128,depth)
    arena=fnp.empty(size,dtype=fnp.float32)
    views=env["_cp3_bind_arena_views"](arena,specs)
    for m,k,n in ((128,128,128),(96,128,64),(64,96,128),(128,128,128)):
        a=fnp.asarray(rng.normal(size=(m,k)).astype(np.float32));b=fnp.asarray(rng.normal(size=(k,n)).astype(np.float32))
        expected=fnp.empty((m,n),dtype=fnp.float32);actual=fnp.empty((m,n),dtype=fnp.float32)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ref:
            env[f"_packed_product_{depth}"](a,b,expected,views)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as got:
            env["_packed_product_cached"](a,b,actual,views,depth)
        assert np.array_equal(np.asarray(expected),np.asarray(actual)),(depth,m,k,n)
        assert ref.flops_used==got.flops_used,(depth,ref.flops_used,got.flops_used)
        records.append(dict(depth=depth,shape=(m,k,n),byte_exact=True,flops=int(got.flops_used)))
    env["_PACKED_FRAME_CACHE"].clear()
result=dict(candidate=args.candidate,sha256=hashlib.sha256((HERE/args.candidate).read_bytes()).hexdigest(),records=records)
target.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
