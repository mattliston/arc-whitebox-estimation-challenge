"""Native frozen helper vs independent NumPy equation on signed inputs."""
from datetime import datetime,timezone
import hashlib,json,math,sys,types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

H=Path(__file__).resolve().parent;P2=H.parents[1]
OLD=P2/'codex_resume_20260919_v1/joint_incoming_moment_head_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((H/'build.json').read_text());assert sha(H/'candidate.py')==b['candidate_sha256']
assert b['exact_source_inverse'] and b['helpers_ast_inverse'] and b['all_parameter_bytes_exact']
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
m=types.ModuleType('joint_test');m.__file__=str(H/'candidate.py');sys.modules[m.__name__]=m
exec(compile((H/'candidate.py').read_text(),m.__file__,'exec'),m.__dict__)
ref={};exec(compile((OLD/'head.py').read_text(),str(OLD/'head.py'),'exec'),ref)
t=dict(np.load(b['model']['path'],allow_pickle=False));rng=np.random.default_rng(349113)
n=9;A=rng.normal(size=(n,n));C=A@A.T/n+np.eye(n)
C[0,1]+=1e-5;C[2,4]-=2e-5
mu=rng.normal(size=n);K=rng.normal(size=(n,n))*.025
W=rng.normal(size=(n,n)).astype(np.float32)/np.sqrt(n)
hidden=rng.normal(size=(n,64)).astype(np.float32)*.2
cdf=np.vectorize(lambda x:.5*math.erfc(-float(x)/math.sqrt(2)),otypes=[float])
rows=[]
for layer in (8,11,14):
    expected=list(ref['correction'](np,cdf,layer,mu,C,K,.015,W,hidden,t))
    v=np.diag(C);s=np.sqrt(np.maximum(v,1e-10));expected[1]*=(v*s)[:,None]*s[None,:]
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        args=[fnp.asarray(x) for x in (mu,C,K,W,hidden)]
        actual=m._jm_corrections(layer,args[0],args[1],args[2],.015,args[3],args[4])
        actual=[np.asarray(x).copy() for x in actual]
    errors=[]
    for a,e in zip(actual,expected):
        diff=float(np.max(np.abs(a-e)));tol=1e-9+1e-5*float(np.max(np.abs(e)))
        assert np.isfinite(a).all() and diff<=tol,(layer,diff,tol)
        assert np.array_equal(np.diag(a),np.zeros(n))
        errors.append(dict(max_abs=diff,tolerance=tol))
    assert np.max(np.abs(actual[2]-actual[2].T))<=1e-12
    rows.append(dict(layer=layer,errors=errors,flops=int(ctx.summary_dict()['flops_used'])))
out=dict(passed=True,rows=rows,candidate_sha256=sha(H/'candidate.py'),model_sha256=sha(Path(b['model']['path'])),
         utc_finished=datetime.now(timezone.utc).isoformat(),truth_accessed=False)
with (H/'unit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2),flush=True)
