"""Official-version kernel timing/certificates, separate from whole-card gates.

No estimator output is changed. Compare original D5/D6, same-depth einsum,
and a streamed D6 leaf on representative shapes and one/four BLAS threads.
All files are unique. Same inputs, repeated timings, and reference errors.
"""
import ast
import gc
import json
from pathlib import Path
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_limits
from packed_arena_generic import layout
from streamed_leaf_layout_v1 import streamed_layout, transform_kernel
from streamed_leaf_winograd_v1 import streamed_leaf

HERE=Path(__file__).resolve().parent
outpath=HERE/"leaf_runtime_benchmark_v1.json"
if outpath.exists():
    raise FileExistsError(outpath)
source=(HERE/"candidate_fullpair_scalar_final_cpcubes_v1.py").read_text()
lines=source.splitlines(keepends=True)
funcs={n.name:"".join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
env=dict(fnp=fnp,streamed_leaf=streamed_leaf)
for name in ("_cp3_bind_arena_views","_split_views_5","_split_views_6","_packed_product_5","_packed_product_6"):
    exec(funcs[name],env)
exec(transform_kernel(funcs["_packed_product_6"]).replace("def _packed_product_6(","def _streamed6("),env)
einsum=funcs["_packed_product_6"].replace("def _packed_product_6(","def _einsum6(")
old="fnp.matmul(current_a, current_b, out=current)"
assert einsum.count(old)==1
einsum=einsum.replace(old,"fnp.einsum('...ik,...kj->...ij',current_a,current_b,out=current,optimize=False)")
exec(einsum,env)
rng=np.random.default_rng(214607)
records=[]
started=time.time()
for threads in (1,4):
    with threadpool_limits(limits=threads):
        for m,k,n in ((128,128,128),(1024,768,4096),(832,4096,832)):
            aa=rng.normal(size=(m,k)).astype(np.float32)/np.float32(np.sqrt(k))
            bb=rng.normal(size=(k,n)).astype(np.float32)
            exact=aa.astype(np.float64)@bb.astype(np.float64)
            a=fnp.asarray(aa);b=fnp.asarray(bb)
            for kind,depth,name in (("d5",5,"_packed_product_5"),("d6",6,"_packed_product_6"),
                                    ("streamed6",6,"_streamed6"),("einsum6",6,"_einsum6")):
                if kind=="streamed6":
                    size,specs,_=streamed_layout(m,k,n)
                else:
                    size,specs,_=layout(m,k,n,depth)
                arena=fnp.empty(size,dtype=fnp.float32)
                views=env["_cp3_bind_arena_views"](arena,specs)
                result=fnp.empty((m,n),dtype=fnp.float32)
                times=[];bills=[]
                for repeat in range(2):
                    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                        t0=time.perf_counter()
                        env[name](a,b,result,views)
                        times.append(time.perf_counter()-t0)
                    bills.append(int(ctx.flops_used))
                error=np.asarray(result).astype(np.float64)-exact
                rel=float(np.linalg.norm(error)/np.linalg.norm(exact))
                assert np.isfinite(error).all() and rel<2e-4,(kind,(m,k,n),rel)
                row=dict(threads=threads,shape=(m,k,n),kind=kind,wall_s=times,flops=bills,
                         relative_frob=rel,max_abs=float(np.max(np.abs(error))),arena_bytes=size*4)
                records.append(row)
                print(json.dumps(row),flush=True)
                del arena,views,result
                gc.collect()
receipt=dict(records=records,wall_s=time.time()-started,
             interpretation="Kernel-only native official-version timings, not a passing full estimator or an official-worker runtime guarantee.")
outpath.write_text(json.dumps(receipt,indent=2)+"\n")
print("LEAF_RUNTIME_BENCHMARK_COMPLETE",flush=True)
