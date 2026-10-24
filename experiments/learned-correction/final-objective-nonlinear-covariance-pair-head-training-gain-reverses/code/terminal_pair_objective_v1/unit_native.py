import sys,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import *
verify();build=read(HERE/'build.json');assert pin(HERE/'candidate.py')==build['candidate']
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('terminal_pair_unit');mod.__file__=str(HERE/'candidate.py');sys.modules[mod.__name__]=mod
exec(compile((HERE/'candidate.py').read_text(),mod.__file__,'exec'),mod.__dict__)
p={}
for path in ('model.npz','normalizers.npz'):
    with np.load(HERE/path) as z:p.update({k:z[k] for k in z.files})
with np.load(HERE/'capture_exact_00/row_0035.npz') as z:
    I=np.array([0,2,17,83,147,300,559,721,1009]);F=z['F'][I];s=z['s14'][I];C=z['C'][np.ix_(I,I)];K=z['K'][np.ix_(I,I)]
f=np.clip(F/p['feature_scale'],-8,8);R=(C/np.outer(s,s)).astype(np.float32)/p['rho_scale'];S=(K/np.outer(s*s,s)).astype(np.float32)/p['third_scale']
left=f@p['left.weight'].T;right=f@p['right.weight'].T;raw=np.empty_like(C,dtype=np.float32)
for i in range(len(I)):
    for j in range(len(I)):
        h=(left[i]+right[j])+p['bias'];h=h+R[i,j]*p['edge.weight'][:,0];h=h+S[i,j]*p['edge.weight'][:,1];h=h+S[j,i]*p['edge.weight'][:,2]
        raw[i,j]=np.tanh(h)@p['output.weight'][0]+p['output.bias'][0]
D=(raw+raw.T).astype(np.float64)*(.001*np.outer(s,s));np.fill_diagonal(D,0.)
def call(order):
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        ans=mod._tp_covariance(fnp.asarray(C[np.ix_(order,order)]),fnp.asarray(F[order,:30]),fnp.asarray(K[np.ix_(order,order)]),fnp.asarray(s[order]),fnp.asarray(F[order,30:]))
    return np.asarray(ans).copy(),int(ctx.summary_dict()['flops_used'])
got,bill=call(np.arange(len(I)));error=float(np.max(np.abs(got-(C+D))));bound=1e-10+2e-5*max(float(np.max(np.abs(D))),1e-12)
assert error<=bound,(error,bound)
perm=np.array([8,0,5,2,4,7,1,6,3]);other,bill2=call(perm);equiv=float(np.max(np.abs(other-got[np.ix_(perm,perm)])))
assert equiv<=bound and np.array_equal(np.diag(got),np.diag(C))
sym=float(np.max(np.abs((got-C)-(got-C).T)));assert sym<1e-14
once(HERE/'unit_native_result.json',dict(passed=True,independent_scalar_max_abs=error,bound=bound,equivariance_max_abs=equiv,delta_symmetry_error=sym,diagonal_exact=True,bills=[bill,bill2],candidate=build['candidate']))
print(json.dumps(dict(passed=True,maximum_error=error)),flush=True)
