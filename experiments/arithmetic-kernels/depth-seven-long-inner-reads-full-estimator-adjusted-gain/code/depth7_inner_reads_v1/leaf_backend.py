"""Public metered leaf backends on fixed shapes, including the einsum copy."""
from pathlib import Path
import hashlib
import json
import statistics
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent
rng = np.random.default_rng(2026092715)
batch = 7**6
rows = []
for m,k,n in [(8,16,8),(7,16,8),(6,16,7)]:
    A = rng.standard_normal((batch,m,k)).astype(np.float32)
    B = rng.standard_normal((batch,k,n)).astype(np.float32)
    reference = A.astype(np.float64) @ B.astype(np.float64)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        a,b = fnp.asarray(A),fnp.asarray(B)
        outs = {key:fnp.empty((batch,m,n),dtype=fnp.float32) for key in ['matmul','einsum']}
    samples = {key:[] for key in outs}
    for rep in range(-1,5):
        order = ['matmul','einsum'] if rep%2==0 else ['einsum','matmul']
        for key in order:
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                t0 = time.process_time()
                if key=='matmul':
                    fnp.matmul(a,b,out=outs[key])
                else:
                    fnp.einsum('...ik,...kj->...ij',a,b,out=outs[key])
                seconds=time.process_time()-t0
            if rep>=0:
                samples[key].append(dict(cpu_s=seconds,flops=int(ctx.summary_dict()['flops_used'])))
    summary={}
    for key in samples:
        value=np.asarray(outs[key],dtype=np.float64)
        error=float(np.linalg.norm(value-reference)/np.linalg.norm(reference))
        summary[key]=dict(cpu_s=statistics.median(x['cpu_s'] for x in samples[key]),
            flops=samples[key][0]['flops'],relative_error=error)
        assert len({x['flops'] for x in samples[key]})==1
    row=dict(leaf_shape=[m,k,n],batch=batch,samples=samples,summary=summary,
        cpu_ratio=summary['einsum']['cpu_s']/summary['matmul']['cpu_s'],
        bill_ratio=summary['einsum']['flops']/summary['matmul']['flops'])
    rows.append(row)
    print(json.dumps({k:row[k] for k in ['leaf_shape','cpu_ratio','bill_ratio']}),flush=True)
    del A,B,a,b,outs,reference,value
cpu=sum(r['summary']['einsum']['cpu_s'] for r in rows)/sum(r['summary']['matmul']['cpu_s'] for r in rows)
bill=sum(r['summary']['einsum']['flops'] for r in rows)/sum(r['summary']['matmul']['flops'] for r in rows)
gates=dict(cpu=cpu<=.8,bill=bill<=1.05,error=all(r['summary']['einsum']['relative_error']<2e-6 for r in rows))
result=dict(rows=rows,aggregate_cpu_ratio=cpu,aggregate_bill_ratio=bill,gates=gates,
    passed=all(gates.values()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    protocol_sha256=hashlib.sha256((D/'LEAF_BACKEND.md').read_bytes()).hexdigest(),
    targets_accessed=False,scope='Fixed leaf primitive only; no full-product or estimator claim')
with (D/'leaf_backend_result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ['aggregate_cpu_ratio','aggregate_bill_ratio','gates','passed']}),flush=True)
