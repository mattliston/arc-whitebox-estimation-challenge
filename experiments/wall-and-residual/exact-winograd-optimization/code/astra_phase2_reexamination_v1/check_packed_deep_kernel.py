"""Small-shape, independently metered lifetime-layout arithmetic certificate."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_limits
from packed_arena_generic import layout
HERE=Path(__file__).resolve().parent;outfile=HERE/"packed_deep_kernel_v1.json"
if outfile.exists():raise FileExistsError(outfile)
spec=importlib.util.spec_from_file_location("deep_kernel_candidate",HERE/"candidate_binary_learned_d6_v1.py")
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
rng=np.random.default_rng(694);records=[]
with threadpool_limits(limits=2):
    for d,shape in ((6,(128,128,256)),(6,(128,256,128)),(3,(128,128,128))):
        m,k,n=shape;A=rng.normal(size=(m,k)).astype(np.float32)/np.sqrt(k);B=rng.normal(size=(k,n)).astype(np.float32)
        ref=A.astype(np.float64)@B.astype(np.float64);size,specs,_=layout(m,k,n,d)
        with flops.BudgetContext(flop_budget=2**38,quiet=True) as ctx:
            arena=fnp.empty(size,dtype=fnp.float32);views=mod._cp3_bind_arena_views(arena,specs)
            out=fnp.empty((m,n),dtype=fnp.float32)
            getattr(mod,f"_packed_product_{d}")(fnp.asarray(A,dtype=fnp.float32),fnp.asarray(B,dtype=fnp.float32),out,views)
        value=np.asarray(out);err=value-ref
        r=dict(depth=d,shape=shape,relative_frob=float(np.linalg.norm(err)/np.linalg.norm(ref)),max_abs=float(np.max(np.abs(err))),flops=int(ctx.flops_used))
        assert r["relative_frob"]<1e-4 and r["max_abs"]<1e-3,r
        records.append(r);print(json.dumps(r),flush=True)
outfile.write_text(json.dumps(dict(pass_gate=True,tests=records),indent=2)+"\n")
