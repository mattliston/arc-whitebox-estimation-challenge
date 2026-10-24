"""Root-only small paid-helper algebra fixture; no estimator import."""
from pathlib import Path
import ast, hashlib, json, os, sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D=Path(__file__).resolve().parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def main():
    assert len(sys.argv)==3
    out=Path(sys.argv[2]).resolve();assert not out.exists()
    build_pin=pin(D/'build.json');assert build_pin['sha256']==sys.argv[1]
    build=json.loads((D/'build.json').read_text())
    for k in ('parent','candidate','builder','fragment','primitive','protocol','diff'):assert pin(build[k]['path'])==build[k]
    source=Path(build['candidate']['path']).read_text();inverse=source
    for e in reversed(build['edits']):assert inverse.count(e['after'])==1;inverse=inverse.replace(e['after'],e['before'],1)
    assert inverse==Path(build['parent']['path']).read_text()
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    env=dict(fnp=fnp,_E2_LAYER={})
    exec(compile(Path(build['fragment']['path']).read_text(),build['fragment']['path'],'exec'),env)
    n=8;mean=np.arange(1,n+1,dtype=np.float64);u=mean/np.linalg.norm(mean)
    prev=np.array([6,1,4,0]);sel=np.array([2,6]);q=3
    P=((np.arange(len(prev)*q).reshape(len(prev),q)%7)-3).astype(np.float32)/8
    B=((np.arange(n*2).reshape(n,2)%9)-4).astype(np.float32)/16;B[sel,np.arange(2)]=0
    ts=np.array([.25,-.5],np.float32);ss=np.array([-.125,.25],np.float32)
    coeff=np.array([.25,-.125,.5,.75,-.25,-.5,.375],np.float32)
    W=((np.arange(n*6).reshape(n,6)%13)-6).astype(np.float32)/8;tail=np.array([0,3,5])
    columns=np.zeros((n,q+4));columns[prev,:q]=P;columns[:,q:q+2]=B;columns[:,q+2:]=B
    columns[sel,q+np.arange(2)]+=ts;columns[sel,q+2+np.arange(2)]+=ss
    def formula(cols,lam,w,m):
        uu=m/np.sqrt(max(float(m@m),1e-30));tau=cols@(lam*np.sum(cols*cols,axis=0));r2=np.sum(w*w,axis=0);z=w.T@uu
        A=np.sum(lam*(uu@cols)**3);bb=uu@tau
        linear=3*r2/(n+2)*(w.T@tau);correction=((n+2)*A-3*bb)/(n-1)*(z**3-3*r2*z/(n+2))
        return linear+correction,linear,correction
    cases=[]
    def run(name,w,p,previous,selected,base,t,s,lam,cols,m,tail_indices):
        inputs=(w,p,previous,selected,base,t,s,lam,m,tail_indices);before=[v.tobytes() for v in inputs]
        with flops.BudgetContext(flop_budget=1000000,quiet=True) as budget:
            ww,pp,pr,se,bb,tt,ss,ll,mm,ti=map(fnp.asarray,inputs)
            env['_E2_LAYER']['ynt_previous_mean']=mm
            value=np.asarray(env['_ynt_trace'](ww,pp,pr,(p.shape[1],se,bb,tt,ss),ll,ti)).copy()
            expected,linear,correction=formula(cols,lam.astype(np.float64),w[:,tail_indices].astype(np.float64),m)
            assert np.allclose(value,expected,rtol=2e-12,atol=2e-12),(name,value,expected)
            assert before==[v.tobytes() for v in inputs]
            billed=int(budget.summary_dict()['flops_used']);assert billed>0
        cases.append(dict(case=name,passed=True,max_error=float(np.max(np.abs(value-expected))),flops=billed))
        return value,linear,correction
    run('signed_full_ambient',W,P,prev,sel,B,ts,ss,coeff,columns,mean,tail)
    empty=np.zeros((n,0),np.float32);ei=np.zeros(0,np.int64);ef=np.zeros(0,np.float32);allrows=np.arange(n)
    aligned=(mean*.25).astype(np.float32)[:,None];ll=np.array([-.75],np.float32)
    val,_,_=run('mean_aligned_exact',W,aligned,allrows,ei,empty,ef,ef,ll,aligned.astype(np.float64),mean,tail)
    exact=(W[:,tail].astype(np.float64).T@aligned.astype(np.float64))[:,0]**3*float(ll[0])
    assert np.allclose(val,exact,rtol=2e-12,atol=2e-12)
    axis=np.eye(n)[0];atoms=[];lams=[]
    for e in np.eye(n):atoms.extend((axis+e,axis-e,axis));lams.extend((1/6,1/6,-1/3))
    atoms=np.array(atoms,np.float32).T;lams=np.array(lams,np.float32)
    _,_,corr=run('pure_trace_zero_harmonic',W,atoms,allrows,ei,empty,ef,ef,lams,atoms.astype(np.float64),axis,tail)
    assert np.max(np.abs(corr))<2e-12
    # +/- orthogonal-axis quadrature is exact for degree<=3 at fixed z and radius.
    Q=np.linalg.qr(np.column_stack((u,np.eye(n)[:,1:])))[0];orth=Q[:,1:];z=.3;r2=1.7
    ws=np.column_stack([z*u+sgn*np.sqrt(r2-z*z)*orth[:,j] for j in range(n-1) for sgn in (-1,1)])
    exact_mean=float(np.mean(np.sum((ws.T@columns)**3*coeff.astype(np.float64),axis=1)))
    predicted=float(np.mean(formula(columns,coeff.astype(np.float64),ws,mean)[0]))
    assert abs(exact_mean-predicted)<2e-12
    nodes=[x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='_predict_binary'];node=nodes[-1]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'candidate_cleanup','exec'),env)
    env.update(_YNT_REPAIR=[None],_YNT_STATS={})
    for fail in (False,True):
        def parent():
            if fail:raise RuntimeError('fixture')
            return 'ok'
        env['_YNT_PARENT_PREDICT']=parent;env['_E2_LAYER']['ynt_previous_mean']='stale'
        try:assert env['_predict_binary']()=='ok'
        except RuntimeError:assert fail
        assert 'ynt_previous_mean' not in env['_E2_LAYER'] and env['_YNT_STATS']['retired']
    result=dict(passed=True,candidate=build['candidate'],build=build_pin,probe=pin(__file__),cases=cases,
        conditional_quadrature_error=abs(exact_mean-predicted),mean_capture_finally_cleared=True,model_executed=False)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
