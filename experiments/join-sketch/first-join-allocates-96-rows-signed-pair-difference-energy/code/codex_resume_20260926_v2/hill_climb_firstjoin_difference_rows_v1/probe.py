"""Root-only tiny exact-helper fixture: NATIVE -B probe.py BUILD_SHA RESULT."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def reference(diagonal,J,weight,cohorts,layer=7,rank=384):
    energy=np.argsort(-diagonal.astype(np.float64),kind='stable')
    if layer!=7:return np.sort(energy[:rank])
    score=None
    for a,w in cohorts[:3]:
        h=w//2; diff=J[:,a:a+h]-J[:,a+h:a+w]
        term=(diff*diff)@weight[a:a+h]
        score=term if score is None else score+term
    remaining=energy[288:]
    chosen=remaining[np.argsort(-score[remaining].astype(np.float64),kind='stable')[:96]]
    return np.sort(np.concatenate((energy[:288],chosen)))


def main():
    assert len(sys.argv)==3
    result=Path(sys.argv[2]).resolve(); assert not result.exists()
    assert Path(sys.executable).absolute()==D.parent.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    bp=pin(D/'build.json'); assert bp['sha256']==sys.argv[1]
    build=json.loads((D/'build.json').read_text())
    for key in ('parent','parent_build','candidate','builder','fragment','primitive','protocol','policy'):
        assert pin(build[key]['path'])==build[key]
    assert build['parent']['sha256']=='88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
    source=Path(build['candidate']['path']).read_text(); parent=Path(build['parent']['path']).read_text()
    inverse=source.encode()[:-build['suffix_bytes']].decode()
    for edit in reversed(build['edits']):
        assert inverse.count(edit['after'])==1
        inverse=inverse.replace(edit['after'],edit['before'],1)
    assert inverse==parent
    node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_fjdr_rows'][-1]
    env={'fnp':fnp};exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact pinned selector>','exec'),env)
    select=env['_fjdr_rows']; rows=448; cohorts=[(0,4),(4,6),(10,2)]
    coeff=np.array([1,2,-1,-2,1,2,4,-1,-2,-4,2,-2],dtype=np.float32)
    weight=coeff*coeff; diagonal=np.arange(rows,0,-1,dtype=np.float32)
    J=(np.arange(rows*12).reshape(rows,12)%7-3).astype(np.float32)
    for a,w in cohorts:
        h=w//2;J[:,a+h:a+w]=J[:,a:a+h]
        J[320:,a]+=4
    state=dict(columns=0,joins=[],cohorts=cohorts)
    results=[]
    for name,values,diag in [('distinct',J,diagonal),('zero',np.zeros_like(J),diagonal),
                             ('ties',J,np.ones_like(diagonal)),('signed',-J,diagonal)]:
        with flops.BudgetContext(flop_budget=10000000,quiet=True) as ctx:
            j,d,w,c=map(fnp.asarray,(values,diag,weight,coeff))
            snapshots=[np.asarray(x).copy() for x in (j,d,w,c)]
            out=np.asarray(select(7,d,j,w,c,state,0,12,384)).copy()
            repeated=np.asarray(select(7,d,j,w,c,state,0,12,384)).copy()
            fallback=np.asarray(select(10,d,j,w,c,state,0,12,320)).copy()
            after=np.asarray(select(7,d,j,w,c,state,0,12,384)).copy()
            assert np.array_equal(out,reference(diag,values,weight,cohorts))
            assert np.array_equal(out,repeated) and np.array_equal(out,after)
            assert len(np.unique(out))==384 and np.array_equal(out,np.sort(out))
            assert set(np.argsort(-diag,kind='stable')[:288]).issubset(set(out))
            assert np.array_equal(fallback,reference(diag,values,weight,cohorts,10,320))
            assert all(np.array_equal(np.asarray(x),b) for x,b in zip((j,d,w,c),snapshots))
            if name=='distinct':assert not np.array_equal(out,np.arange(384))
            if name=='zero':assert np.array_equal(out,np.arange(384))
            bill=int(ctx.summary_dict()['flops_used']);assert bill>0
        results.append(dict(case=name,flops=bill,rows_sha256=hashlib.sha256(out.tobytes()).hexdigest(),passed=True))
    refusals=[]
    with flops.BudgetContext(flop_budget=10000000,quiet=True):
        j,d,w,c=map(fnp.asarray,(J,diagonal,weight,coeff))
        bad=coeff.copy();bad[2]=1
        for name,cc,st in [('coefficients',fnp.asarray(bad),state),
                          ('old_owner',c,dict(state,columns=2)),
                          ('odd_block',c,dict(state,cohorts=[(0,3),(3,7),(10,2)])),
                          ('gap',c,dict(state,cohorts=[(0,4),(5,6),(11,2)]))]:
            try:select(7,d,j,w,cc,st,0,12,384)
            except RuntimeError:refusals.append(name)
            else:raise AssertionError(name)
    assert len(refusals)==4
    receipt=dict(passed=True,complete=True,candidate=build['candidate'],build=bp,probe=pin(__file__),
                 cases=results,refusals=refusals,all_inputs_unchanged=True,no_persistent_state=True,
                 all_selection_contracts_passed=True,no_predictor_parity_claim=True)
    with result.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
