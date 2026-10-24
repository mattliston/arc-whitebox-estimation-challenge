"""Native helper algebra at small shapes; no network or target arrays."""
import ast
import json
import math
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
HERE=Path(__file__).resolve().parent
out=HERE/'unit.json';assert not out.exists()

def extract(path,name):
 source=path.read_text();lines=source.splitlines()
 node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
 return '\n'.join(lines[node.lineno-1:node.end_lineno])+'\n'

def make(path):
 env=dict(fnp=fnp,flops=flops,_mm=fnp.matmul)
 exec(extract(path,'_sym_upper'),env)
 env['_lr_factor']=lambda A:(fnp.eye(A.shape[0],dtype=fnp.float32),A)
 exec(extract(path,'_flms_cov_star'),env)
 return env['_flms_cov_star']

parent=make(HERE.parent/'candidate_old_both_tail_trace_bounded_v6.py')
candidate=make(HERE/'candidate_v2.py')
rng=np.random.default_rng(968203);n=12
w=rng.normal(size=(n,n)).astype(np.float32)/np.sqrt(np.float32(n))
C=rng.normal(size=(n,n)).astype(np.float64);C[0,0]=-2.
A=rng.normal(size=(n,n)).astype(np.float32)
with flops.BudgetContext(flop_budget=10**8,quiet=True) as ctx:
 wf=fnp.array(w);cf=fnp.array(C);af=fnp.array(A)
 cb,sb=parent(wf,cf,af);cn,sn=candidate(wf,cf,af)
 basec=np.asarray(cb).copy();newc=np.asarray(cn).copy()
 bases=np.asarray(sb).copy();news=np.asarray(sn).copy()
 first=np.asarray(fnp.matmul(cf.astype(fnp.float32),wf)).copy()
expected=np.array([math.fsum(float(w[i,j])*float(first[i,j]) for i in range(n)) for j in range(n)])
diagerror=float(np.max(np.abs(np.diag(newc)-expected)))
off=~np.eye(n,dtype=bool)
assert diagerror<1e-13 and np.array_equal(basec[off],newc[off]) and np.array_equal(bases,news)
record=dict(passed=True,scalar_float64_sum_max=diagerror,offdiagonal_bytes_exact=True,star_bytes_exact=True,
            old_diagonal_error_max=float(np.max(np.abs(np.diag(basec)-expected))),
            native_flops=int(ctx.summary_dict()['flops_used']),network_arrays_accessed=False,
            limitation='Small native helper algebra with ordinary matmul, not wide fast-kernel accuracy.')
with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record,indent=2),flush=True)
