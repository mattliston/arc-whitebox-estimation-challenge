"""Root-only tiny paid polynomial/capture fixture; native Python, 60s/1GB.
Usage: probe.py BUILD_SHA RESULT_PATH. No estimator/model import.
"""
from pathlib import Path
import ast,hashlib,json,os,sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
    assert len(sys.argv)==3
    result=Path(sys.argv[2]).resolve();assert not result.exists()
    bp=pin(D/'build.json');assert bp['sha256']==sys.argv[1]
    build=json.loads((D/'build.json').read_text())
    for k in ('candidate','parent','builder','fragment','primitive','protocol','price','inverse','diff'):assert pin(build[k]['path'])==build[k]
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    tree=ast.parse((D/'candidate.py').read_text());names={'_E2Hold','_sp1_make','_sp1_read'}
    nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names};assert set(nodes)==names
    rows=[]
    with flops.BudgetContext(flop_budget=1_000_000,quiet=True) as ctx:
        calls=[]
        def mm(a,b):calls.append((tuple(a.shape),tuple(b.shape)));return fnp.matmul(a,b)
        def parent(left,right,lam,live,state):return mm(left,(right*lam).T)
        env=dict(fnp=fnp,_E2_LAYER={},_mm=mm,_cp_read_young_sq=parent)
        exec(compile(ast.Module(body=list(nodes.values()),type_ignores=[]),'<source helpers>','exec'),env)
        for case in ('signed_regular','small_contrast','fresh_values'):
            common=np.array([[1.,2.],[2.,-1.],[.5,1.5],[-2.,.25]],np.float32)
            directions=np.array([[1.,.5],[.5,-1.],[2.,.25],[-.5,2.]],np.float32)
            t=np.array([.5,1.],np.float32);s=np.array([-.25,-.5],np.float32)
            coefficient=np.array([.25,-.5],np.float32)
            if case=='small_contrast':common*=4096;t[:]=2**-20;s[:]=-2**-20
            if case=='fresh_values':common+=.25;directions=-directions;t=-t;s=-s
            response=np.array([3,0,2],np.int64);live=np.array([1,3],np.int64)
            plus=common+directions*t;minus=common+directions*s
            pp=np.concatenate((plus,minus),axis=1);lam=np.concatenate((coefficient,-coefficient))
            arrays=[fnp.asarray(v) for v in (common,directions,t,s)]
            packet=env['_sp1_make'](*arrays)
            env['_E2_LAYER']['stable_pair1']=packet
            before=len(calls)
            actual=np.asarray(env['_sp1_read'](fnp.asarray((pp*pp)[response]),fnp.asarray(pp[live]),fnp.asarray(lam),fnp.asarray(live),{'columns':0},fnp.asarray(response),1)).copy()
            assert len(calls)==before+1 and calls[-1]==((3,4),(4,2))
            assert 'stable_pair1' not in env['_E2_LAYER']
            u=common.astype(np.float64)+directions.astype(np.float64)*(t.astype(np.float64)+s.astype(np.float64))/2
            v=directions.astype(np.float64)*(t.astype(np.float64)-s.astype(np.float64))/2
            expected=(u[response]**2+v[response]**2)@(2*coefficient*v[live]).T+(2*u[response]*v[response])@(2*coefficient*u[live]).T
            error=float(np.max(np.abs(actual-expected)));scale=max(1.,float(np.max(np.abs(expected))))
            assert error<=2e-6*scale and np.isfinite(actual).all()
            direct=(pp[response]**2)@(pp[live]*lam).T
            if case=='small_contrast':assert np.max(np.abs(actual-expected))<np.max(np.abs(direct-expected))
            for paid,original in zip(arrays,(common,directions,t,s)):assert np.asarray(paid).tobytes()==original.tobytes()
            rows.append(dict(case=case,polynomial_relative_error=error/scale,one_original_width_paid_gemm=True,inputs_unchanged=True,packet_consumed=True))
        # A mismatched signed pair is refused and the packet is still consumed.
        env['_E2_LAYER']['stable_pair1']=packet
        try:env['_sp1_read'](fnp.asarray((pp*pp)[response]),fnp.asarray(pp[live]),fnp.asarray(np.ones(4,np.float32)),fnp.asarray(live),{'columns':0},fnp.asarray(response),1)
        except RuntimeError as error:assert str(error)=='stable L1 signed pair coefficients'
        else:raise AssertionError('invalid pair accepted')
        assert 'stable_pair1' not in env['_E2_LAYER']
        flops_used=int(ctx.summary_dict()['flops_used'])
    record=dict(passed=True,build=bp,candidate=build['candidate'],probe=pin(__file__),cases=rows,invalid_coefficients_refused=True,paid_flops=flops_used,model_executed=False,accuracy_validated=False)
    with result.open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(record))
if __name__=='__main__':main()
