"""Root-only paid source-bound raw-third identity fixture; 60s/1GB, no model.
Usage: native_python -B probe.py BUILD_SHA RESULT_PATH
"""
from pathlib import Path
import ast,hashlib,json,os,sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def offset(m,c):return np.outer(np.diag(c),m)+2*m[:,None]*c+np.outer(m*m,m)
def main():
    assert len(sys.argv)==3
    result=Path(sys.argv[2]).resolve();assert not result.exists()
    bp=pin(D/'build.json');assert bp['sha256']==sys.argv[1]
    build=json.loads((D/'build.json').read_text())
    for k in ('candidate','parent','builder','fragment','primitive','protocol','diff'):assert pin(build[k]['path'])==build[k]
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    tree=ast.parse((D/'candidate.py').read_text());names={'_E2Hold','_tr14_raw_offset','_target_recenter14'}
    nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
    assert set(nodes)==names
    x=np.array([[.25,-.5,1.,2.],[1.,.75,-.25,.5],[-.5,1.25,.75,-1.],[.125,.25,-.75,1.5]],dtype=np.float64)
    m=x.mean(axis=0);z=x-m;c=z.T@z/len(x)
    raw3=np.einsum('si,si,sj->ij',x,x,x)/len(x)
    target=np.einsum('si,si,sj->ij',z,z,z)/len(x)
    assert np.allclose(target+offset(m,c),raw3,rtol=0,atol=1e-14)
    rows=[]
    with flops.BudgetContext(flop_budget=1_000_000,quiet=True) as ctx:
        env={'fnp':fnp};exec(compile(ast.Module(body=list(nodes.values()),type_ignores=[]),'<source helpers>','exec'),env)
        for name,delta,alpha in [('no_change',np.zeros(4),1.),('partial_covariance',np.array([.03,-.04,.02,-.01]),.92),('opposite_signed',np.array([-.06,.02,-.03,.04]),1.07)]:
            m1=m+delta;c1=c-alpha*(np.outer(m,delta)+np.outer(delta,m)+np.outer(delta,delta))
            arrays=[fnp.asarray(v) for v in (target,m,c,m1,c1)]
            actual=np.asarray(env['_target_recenter14'](*arrays)).copy()
            residual=float(np.max(np.abs(actual+offset(m1,c1)-raw3)))
            assert residual<2e-14 and np.isfinite(actual).all()
            if name=='no_change':assert actual.tobytes()==target.tobytes()
            for paid,original in zip(arrays,(target,m,c,m1,c1)):assert np.asarray(paid).tobytes()==original.tobytes()
            rows.append(dict(case=name,raw_third_max_abs_residual=residual,inputs_unchanged=True))
        flops_used=int(ctx.summary_dict()['flops_used'])
    record=dict(passed=True,build=bp,candidate=build['candidate'],probe=pin(__file__),cases=rows,paid_flops=flops_used,model_executed=False,accuracy_validated=False)
    with result.open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(record))
if __name__=='__main__':main()
