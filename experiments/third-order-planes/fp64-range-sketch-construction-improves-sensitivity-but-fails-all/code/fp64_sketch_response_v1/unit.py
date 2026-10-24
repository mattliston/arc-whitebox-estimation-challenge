import importlib.util
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE,DIAG,read,write,pin,verify,seal_check

seal_check()
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('stable_sketch_unit',HERE/'candidate.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
with flops.BudgetContext(flop_budget=2**41,quiet=True):spec.loader.exec_module(mod)

def check(fields,name):
    j,w,o=[fields[k].astype(np.float64) for k in ['J','weight','Omega']]
    y=j@(w[:,None]*(j.T@o))
    if 'Qp' in fields:
        q,c=[fields[k].astype(np.float64) for k in ['Qp','core']]
        y+=q@(c@(q.T@o))
    reference,_=np.linalg.qr(y,mode='reduced')
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        ff={k:fnp.asarray(v) for k,v in fields.items() if k in ['J','weight','Omega','Qp','core']}
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        got=mod._stable_shared_sketch_qr(ff['J'],ff['weight'],ff['Omega'],ff.get('Qp'),ff.get('core'))
    q=np.asarray(got).copy();rank=q.shape[1]
    assert q.dtype==np.float32 and np.isfinite(q).all()
    orth=float(np.linalg.norm(q.astype(float).T@q-np.eye(rank))/np.sqrt(rank))
    projector=float(np.linalg.norm(q.astype(float)@q.T-reference@reference.T)/np.sqrt(rank))
    assert orth<=1e-5 and projector<=2e-6
    return dict(name=name,shape=list(q.shape),orthogonality=orth,projector_relative=projector,
                reference_Q_rounded_exact=q.tobytes()==reference.astype(np.float32).tobytes(),
                actual_flops=ctx.summary_dict()['flops_used'])

rows=[]
rng=np.random.default_rng(20260920691)
for old,strided in [(False,False),(True,False),(True,True)]:
    j=rng.normal(size=(12,14 if strided else 7)).astype(np.float32)
    if strided:j=j[:,::2]
    f=dict(J=j,weight=rng.random(7).astype(np.float32),Omega=rng.normal(size=(12,5)).astype(np.float32))
    if old:f.update(Qp=rng.normal(size=(12,5)).astype(np.float32),core=np.eye(5,dtype=np.float32))
    rows.append(check(f,f'tiny_old{old}_strided{strided}'))
for row in read(DIAG/'analysis.json')['rows']:
    verify(row['trace'])
    with np.load(row['trace']['path'],allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
    rows.append(check(f,f"{row['label']}_layer{row['layer']}"))
write(HERE/'unit.json',dict(passed=True,rows=rows,wide_cases=40,tiny_cases=3,targets_opened=False))
print(dict(passed=True,cases=len(rows),max_projector_relative=max(r['projector_relative'] for r in rows)),flush=True)
