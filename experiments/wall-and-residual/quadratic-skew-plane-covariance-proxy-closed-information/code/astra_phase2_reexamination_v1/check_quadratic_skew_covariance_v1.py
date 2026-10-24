"""Small exact polynomial/Wick and actual-parent response certificates."""
import inspect
import itertools
import json
from pathlib import Path
import textwrap
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from quadratic_skew_covariance_v1 import proxy,response
from radial_mehler import make_nonlinear
from k3_radial_numpy import gaussian_raw

HERE=Path(__file__).resolve().parent
OUT=HERE/'quadratic_skew_covariance_certificate_v1.json';assert not OUT.exists()
rng=np.random.default_rng(230905)
n=4
x=rng.normal(size=(n,n));R=x@x.T+3*np.eye(n)
R/=np.sqrt(np.outer(np.diag(R),np.diag(R)))
t=rng.normal(size=(n,n,n))*.07
T=sum(t.transpose(p) for p in itertools.permutations(range(3)))/6
ri=np.linalg.inv(R)
nodes,weights=hermegauss(5);weights/=np.sqrt(2*np.pi)
ix=np.array(list(itertools.product(range(5),repeat=n)))
z=nodes[ix]@np.linalg.cholesky(R).T;wt=np.prod(weights[ix],axis=1)
y=z@ri
q=(np.einsum('iab,sa,sb->si',T,y,y)-np.einsum('iab,ab->i',T,ri))/6
third=np.einsum('s,si,sj,sk->ijk',wt,q,z,z)
third=third+third.transpose(1,0,2)+third.transpose(1,2,0)
qq=np.einsum('s,si,sj->ij',wt,q,q)
base4=np.einsum('s,si,sj,sk,sl->ijkl',wt,q,q,z,z)
raw4=np.zeros((n,)*4)
for pair in itertools.combinations(range(4),2):
    rest=tuple(i for i in range(4) if i not in pair)
    order=pair+rest
    raw4+=base4.transpose(np.argsort(order))
for pair in ((0,1,2,3),(0,2,1,3),(0,3,1,2)):
    part=np.einsum('ij,kl->ijkl',qq,R)+np.einsum('ij,kl->ijkl',R,qq)
    raw4-=part.transpose(np.argsort(pair))
one=np.einsum('ija,ab,klb->ijkl',T,ri,T)
pred=(4/9)*(one+one.transpose(0,2,1,3)+one.transpose(0,2,3,1))
S=np.array([[T[i,i,j] for j in range(n)] for i in range(n)])
Q22,Q31,q4,info=proxy(R,S,np.arange(n))
D=S@ri@S.T
condition_error=0.
for i in range(n):
    for j in range(n):
        if i==j:
            vec=S[i]
        else:
            ids=[i,j]
            vec=R[:,ids]@np.linalg.solve(R[np.ix_(ids,ids)],[S[i,j],S[j,i]])
        condition_error=max(condition_error,abs(Q22[i,j]-(4/9)*(D[i,j]+2*vec@ri@vec)),
                            abs(Q31[i,j]-(4/3)*(S[i]@ri@vec)))

# Independent derivative: change the actual parent's cumulant arguments,
# with an explicit independent Q31 argument instead of its radial ansatz.
source=textwrap.dedent(inspect.getsource(make_nonlinear(6,2,True,True)))
source=source.replace('metric="identity"):','metric="identity",q31_override=None):')
needle='T31=t4[:,None]*rho if k31 else None'
assert source.count(needle)==1
source=source.replace(needle,needle+'\n    if q31_override is not None: T31=q31_override')
original=make_nonlinear(6,2,True,True)
env=dict(original.__globals__,gauss_order=6,edge_order=2,local=True,k31=True)
exec(source,env);nonlinear=env['nonlinear']
sigma=np.exp(rng.normal(size=n)*.2);C=R*np.outer(sigma,sigma)
mu=rng.normal(size=n)*sigma
K=S*(sigma[:,None]**2*sigma[None,:]);k4base=np.full(n,.04)*sigma**4
K22base=.04*(1+2*R*R)/3*np.outer(sigma*sigma,sigma*sigma)
T31base=.04*(R-np.eye(n))
m,*_=nonlinear(mu,C,K,k4base,K22base,'covariance',q31_override=T31base)
s,a,pdf,g,mg,*_=gaussian_raw(mu,np.diag(C))
dS=rng.normal(size=(n,n))*.01;dQ22=rng.normal(size=(n,n))*.01
dQ22=(dQ22+dQ22.T)/2;dQ31=rng.normal(size=(n,n))*.01;dq4=rng.normal(size=n)*.01
analytic=response(R,s,a,pdf,g,mg/s,m/s,dS,dQ22,dQ31,dq4)
eps=1e-4;covs=[]
for sign in (-1,1):
    _,cv,*_=nonlinear(mu,C,K+sign*eps*dS*(s[:,None]**2*s[None,:]),
                     k4base+sign*eps*dq4*s**4,
                     K22base+sign*eps*dQ22*np.outer(s*s,s*s),'covariance',
                     q31_override=T31base+sign*eps*dQ31)
    covs.append(cv)
finite=(covs[1]-covs[0])/(2*eps)
errors=dict(first_order_third_max=float(np.max(np.abs(third-T))),
            second_order_fourth_max=float(np.max(np.abs(raw4-pred))),
            conditional_formula_max=float(condition_error),
            pair_response_max=float(np.max(np.abs(finite-analytic))),
            pair_response_relative=float(np.linalg.norm(finite-analytic)/np.linalg.norm(analytic)))
passed=all(errors[k]<1e-12 for k in ('first_order_third_max','second_order_fourth_max','conditional_formula_max')) and errors['pair_response_relative']<1e-7
result=dict(passed=passed,errors=errors,solve=info,qualification='Exact finite-degree Gaussian integration, n4 only; proxy approximation accuracy untested, no labels.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
assert passed
