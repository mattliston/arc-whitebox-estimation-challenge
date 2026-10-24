"""Measure exact kernel cost polynomials before any larger schedule test.

Each fixed Winograd schedule has billed cost a*m*k*n+b*m*k+c*k*n+d*m*n.
Four independent shapes identify it; a fifth verifies it. Input creation,
arena allocation and output allocation are outside this kernel-only bill.
This is offline analysis, not an estimator or a metering shortcut.
"""
import ast
import json
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_limits
from packed_arena_generic import layout

HERE=Path(__file__).resolve().parent
outpath=HERE/"direct_kernel_schedule_prices_v1.json"
if outpath.exists():
    raise FileExistsError(outpath)
source=(HERE/"candidate_fullpair_scalar_final_batched_d6_v1.py").read_text()
functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
env=dict(fnp=fnp)
exec(functions["_cp3_bind_arena_views"],env)
for depth in (3,4,5,6,7):
    split=functions["_split_views_6"].replace("_split_views_6",f"_split_views_{depth}")
    split_tree=ast.parse(split)
    class DepthConstants(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value==6:
                return ast.copy_location(ast.Constant(depth),node)
            return node
    split=ast.unparse(ast.fix_missing_locations(DepthConstants().visit(split_tree)))
    kernel=functions["_packed_product_6"].replace("_packed_product_6",f"_packed_product_{depth}").replace("_split_views_6",f"_split_views_{depth}").replace("levels = 6",f"levels = {depth}")
    exec(split+"\n\n"+kernel,env)

rng=np.random.default_rng(210893)
shapes=((128,128,128),(256,128,128),(128,256,128),(128,128,256),(256,256,256))
records=[];coefficients={}
with threadpool_limits(limits=2):
    for depth in (3,4,5,6,7):
        costs=[];design=[]
        for m,k,n in shapes:
            a=rng.normal(size=(m,k)).astype(np.float32)/np.float32(np.sqrt(k))
            b=rng.normal(size=(k,n)).astype(np.float32)
            exact=a.astype(np.float64)@b.astype(np.float64)
            size,specs,_=layout(m,k,n,depth)
            arena=fnp.empty(size,dtype=fnp.float32)
            views=env["_cp3_bind_arena_views"](arena,specs)
            aa=fnp.asarray(a,dtype=fnp.float32);bb=fnp.asarray(b,dtype=fnp.float32)
            output=fnp.empty((m,n),dtype=fnp.float32)
            with flops.BudgetContext(flop_budget=2**38,quiet=True) as ctx:
                env[f"_packed_product_{depth}"](aa,bb,output,views)
            error=np.asarray(output)-exact
            rec=dict(depth=depth,shape=(m,k,n),flops=int(ctx.flops_used),
                     relative_frob=float(np.linalg.norm(error)/np.linalg.norm(exact)),
                     arena_bytes=size*4)
            records.append(rec);costs.append(rec["flops"]);design.append((m*k*n,m*k,k*n,m*n))
            print(json.dumps(rec),flush=True)
            del arena,views,aa,bb,output
        coef=np.linalg.solve(np.asarray(design[:4],float),np.asarray(costs[:4],float))
        prediction=float(np.asarray(design[4],float)@coef)
        assert abs(prediction-costs[4])<1e-4,(depth,prediction,costs[4])
        coefficients[depth]=coef.tolist()
    prices=[]
    for shape in ((512,512,512),(768,768,1024),(1024,1024,1024),(1024,768,4096),(768,4096,768),(1024,1024,4096),(1024,4096,1024)):
        m,k,n=shape
        prices.append(dict(shape=shape,kernel_flops={depth:float(np.dot(coef,(m*k*n,m*k,k*n,m*n))) for depth,coef in coefficients.items()}))
receipt=dict(records=records,coefficients=coefficients,prices=prices,
             interpretation="Fixed schedule kernel-only cost; held-out shape confirms the polynomial. Deployment requires its own parity/resources/full-panel gates.")
outpath.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(prices=prices)),flush=True)
