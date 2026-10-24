"""Small paid-kernel check against direct differential-operator convolution."""
from pathlib import Path
import ast, hashlib, json, math
import numpy as np
import flopscope as flops
from flopscope import numpy as fnp

HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'candidates.json').read_text())
source=Path(manifest['arms'][0]['candidate']['path'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['arms'][0]['candidate']['sha256']
tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_q3r2_delta')
env={'fnp':fnp};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
kernel=env['_q3r2_delta']
rng=np.random.default_rng(20260927)
n=5
A=rng.normal(size=(9,n)).astype(np.float32)
T=rng.normal(size=(n,n)).astype(np.float32)*.07
skew=rng.normal(size=n).astype(np.float32)*.09
rho=rng.normal(size=(n,n)).astype(np.float32)*.2
ss=np.outer(np.linspace(.5,1.5,n),np.linspace(.5,1.5,n))
cases=[]
for zero in (False,True):
    tt=T if not zero else np.zeros_like(T)
    sk=skew if not zero else np.zeros_like(skew)
    reference=np.zeros((n,n),dtype=np.float64)
    corr=(rho.astype(np.float64)+rho.T.astype(np.float64))*.5
    for i in range(n):
        for j in range(n):
            if i==j:continue
            op=((3,0,float(sk[i])/6),(2,1,float(tt[i,j])/2),
                (1,2,float(tt[j,i])/2),(0,3,float(sk[j])/6))
            value=0.
            for r in range(3):
                part=sum(c*d*.5*float(A[a+p+r,i])*float(A[b+q+r,j])
                         for a,b,c in op for p,q,d in op)
                if r==0:
                    part-=float(sk[i])**2/72*float(A[6,i])*float(A[0,j])
                    part-=float(sk[j])**2/72*float(A[0,i])*float(A[6,j])
                    part-=float(sk[i])*float(sk[j])/36*float(A[3,i])*float(A[3,j])
                value+=corr[i,j]**r/math.factorial(r)*part
            reference[i,j]=value*ss[i,j]
    bills=[]
    for repeat in range(2):
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as context:
            aa=list(fnp.unstack(fnp.asarray(A)))
            tensor=fnp.asarray(tt);skewed=fnp.asarray(sk);correlation=fnp.asarray(rho);scale=fnp.asarray(ss)
            result=kernel(aa,tensor,skewed,correlation,scale)
            actual=np.asarray(result)
            bill=int(context.summary_dict()['flops_used'])
            np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=2e-9)
            np.testing.assert_array_equal(actual,actual.T)
            np.testing.assert_array_equal(np.diag(actual),np.zeros(n))
            bills.append(bill)
            del aa,tensor,skewed,correlation,scale,result
    assert bills[0]==bills[1] and bills[0]>0
    cases.append(dict(zero_skew=zero,max_abs_error=float(np.max(np.abs(actual-reference))),bill=bills[0]))
out=dict(passed=True,candidate=manifest['arms'][0]['candidate'],cases=cases,
    reference='direct 16-term differential operator convolution, three correlation orders')
with (HERE/'native_verification.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out),flush=True)
