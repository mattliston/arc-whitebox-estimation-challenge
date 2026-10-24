from pathlib import Path
import sys,types,json,hashlib,gc
import numpy as np
import flopscope as flops
HERE=Path(__file__).resolve().parent;TRACE=HERE.parent/'current_nystrom_origin_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
for e in read(HERE/'seal.json')['sources']:assert pin(e['path'])==e
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('stable_reference_prepare');mod.__file__=str(HERE/'candidate.py');sys.modules[mod.__name__]=mod
exec(compile((HERE/'candidate.py').read_text(),mod.__file__,'exec'),mod.__dict__)
rows=[]
for name in ('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus'):
    for r in read(TRACE/name/'complete.json')['rows']:
        e=r['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:a={k:z[k].copy() for k in ('J','weight','Qp','core_in','rows')}
        J=a['J'].astype('float64');Qp=a['Qp'].astype('float64');indices=a['rows'];C=a['core_in'].astype('float64')
        expected=J@(a['weight'].astype('float64')[:,None]*J[indices].T)+Qp@(C@Qp[indices].T)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            inputs=[mod.fnp.asarray(a[k]) for k in ('J','weight','Qp','core_in','rows')]
            actual=np.asarray(mod._sr_wide_sketch(*inputs)).copy()
        assert actual.dtype==np.dtype('float64') and actual.shape==expected.shape
        assert actual.tobytes()==expected.tobytes() and np.isfinite(actual).all()
        row=dict(name=name,mini=r['mini'],shape=list(actual.shape),bytes_exact=True,flops=int(ctx.summary_dict()['flops_used']))
        rows.append(row);print(json.dumps(row),flush=True)
        del a,J,Qp,C,expected,inputs,actual;gc.collect()
assert mod._SR_STATS==dict(late_dispatch=0,wide_sketch=10)
with (HERE/'preparation.json').open('x') as f:json.dump(dict(passed=True,rows=rows,final_targets_read=False),f,indent=2);f.write('\n')
