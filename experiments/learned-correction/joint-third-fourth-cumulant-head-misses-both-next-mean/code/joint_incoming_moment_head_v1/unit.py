import ast,itertools,math,resource,time
from common import *
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from head import features,correction,predict_planes
from math_core import initialize,coefficients,forward,backward,consumer,losses
from moments import central,storage_bounds,gaussian_fourth_variances
verify();assert all(x['num_threads']==1 for x in threadpool_info());start=time.monotonic();rng=np.random.default_rng(2392071)
def cdf(x):return np.array([.5*math.erfc(-float(v)/math.sqrt(2)) for v in x],np.float64)
centering=[]
for shift in (0.,2.):
    X=rng.normal(size=(127,7))+.6*rng.normal(size=(127,1))**2+shift*rng.normal(size=(1,7))
    N=len(X);mu=np.mean(X,axis=0);m2=np.mean(X**2,axis=0);m3=np.mean(X**3,axis=0);m4=np.mean(X**4,axis=0)
    M11=X.T@X/N;M21=(X**2).T@X/N;M31=(X**3).T@X/N;M22=(X**2).T@(X**2)/N
    K21,K31,K22,C,v=central(mu,m2,m3,m4,M11,M21,M31,M22)
    Z=X-mu;Cc=Z.T@Z/N;vc=np.diag(Cc)
    direct=[(Z**2).T@Z/N,(Z**3).T@Z/N-3*vc[:,None]*Cc,(Z**2).T@(Z**2)/N-np.outer(vc,vc)-2*Cc*Cc]
    errors=[float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a,b in zip((K21,K31,K22),direct)];assert max(errors)<=5e-11,errors
    rounded=central(mu,m2,m3,m4,*[x.astype(np.float32) for x in (M11,M21,M31,M22)])
    bounds=storage_bounds(mu,m2,*[x.astype(np.float32) for x in (M11,M21,M31,M22)])
    for actual,exact,bound in zip(rounded[1:3],(K31,K22),bounds):assert np.all(np.abs(actual-exact)<=bound+1e-10)
    centering.append(dict(shift=shift,relative=errors,storage_bounds_passed=True))
wick=[]
for va,vb,cross in ((1.,2.,0.),(2.,3.,.7),(.3,.9,-.2)):
    C=np.array([[va,cross],[cross,vb]]);V31,V22=gaussian_fourth_variances(C,1)
    for labels,expected in [((0,0,0,1),V31[0,1]),((0,0,1,1),V22[0,1])]:
        exact=math.fsum(math.prod(C[labels[i],labels[j]] for i,j in enumerate(p)) for p in itertools.permutations(range(4)))
        assert abs(exact-expected)<=1e-12*max(1,abs(expected));wick.append(dict(covariance=C.tolist(),labels=labels,value=exact))

sourcetext=SOURCE.read_text();tree=ast.parse(sourcetext);node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pair_series_fused'][-1]
scope={'fnp':np};source=ast.get_source_segment(sourcetext,node);exec(compile(source,str(SOURCE),'exec'),scope);responses=[]
for asymmetric,floor in ((False,False),(True,False),(False,True),(True,True)):
    n=9;Q=rng.normal(0,.1,(n,n));C=Q@Q.T+np.eye(n)
    if asymmetric:C+=rng.normal(0,.02,(n,n))*(1-np.eye(n))
    if floor:C[0,0]=1e-12;C[0,1:]*=1e-6;C[1:,0]*=1e-6
    s=np.sqrt(np.maximum(np.diag(C),1e-10));a=rng.normal(0,.7,n);p=np.exp(-.5*a*a)/math.sqrt(2*math.pi);g=cdf(a)
    B,weight=coefficients(C,s,a,p,g);R=C/np.outer(s,s);np.fill_diagonal(R,0)
    jets=np.zeros((7,n));jets[1]=g;jets[2]=p;jets[3]=-a*p;jets[4]=(a*a-1)*p;jets[5]=-(a**3-3*a)*p
    F=np.stack([jets]*3);G=F.copy();Z=np.zeros((n,n));z=np.zeros(n);D=rng.normal(0,.02,(3,n,n))
    for x in D:np.fill_diagonal(x,0)
    baseline=scope['_pair_series_fused'](F,G,R,Z,Z,Z,z,z,n)[0]
    actual=(scope['_pair_series_fused'](F,G,R,D[0],D[2],D[1],z,z,n)[0]-baseline)*np.outer(s,s);np.fill_diagonal(actual,0)
    c=dict(B=B,Wnext=rng.normal(0,.2,(n,n)),gnext=rng.uniform(.1,.3,n),weight=weight,target=rng.normal(0,.1,(3,n,n)))
    expected,_=consumer(c,D);err=float(np.linalg.norm(actual-expected)/np.linalg.norm(actual));assert err<=2e-11
    responses.append(dict(asymmetric=asymmetric,active_floor=floor,relative=err))

n=9;F=rng.normal(0,.3,(n,140));R=rng.normal(0,.2,(n,n));S=rng.normal(0,.2,(n,n));t=initialize(np.float64);t['output']=rng.normal(0,.1,(32,3));scale=np.array([.1,.2,.15])
for x in c['target']:np.fill_diagonal(x,0)
energy=losses(c,np.zeros((3,n,n)));pred=forward(F,R,S,t,scale);value,terms,grad=losses(c,pred,energy,True);der=backward(F,R,S,t,scale,grad);gradchecks=[]
for j in range(12):
    direction={k:rng.normal(size=t[k].shape) for k in KEYS};norm=np.sqrt(sum(np.sum(v*v) for v in direction.values()));direction={k:v/norm for k,v in direction.items()}
    exact=sum(float(np.sum(der[k]*direction[k])) for k in KEYS);eps=1e-5;vals=[]
    for sign in (-1,1):
        tt={k:t[k]+sign*eps*direction[k] for k in KEYS};vals.append(losses(c,forward(F,R,S,tt,scale),energy)[0])
    finite=(vals[1]-vals[0])/(2*eps);err=abs(finite-exact)/max(1e-7,abs(exact),abs(finite));assert err<=3e-6,(j,err)
    gradchecks.append(dict(direction=j,relative=err,analytic=exact,finite=finite))

kernels=[]
for n in (17,70):
    q=rng.normal(size=(n,7));C=q@q.T/n+np.eye(n);C+=rng.normal(0,.0001,(n,n))*(1-np.eye(n));mu=rng.normal(0,.3,n)
    K=rng.normal(0,.02,(n,n));W=rng.normal(0,.04,(n,n)).astype(np.float32);H=rng.normal(size=(n,64)).astype(np.float32)
    t=initialize();t['output']=rng.normal(0,.1,(32,3)).astype(np.float32)
    t.update(feature_scale=np.ones(140,np.float32),rho_scale=np.array(.2,np.float32),third_scale=np.array(.05,np.float32),target_scale=np.array([.005,.01,.006],np.float32))
    expected=correction(np,cdf,9,mu,C,K,.02,W,H,t)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):args=[fnp.asarray(x) for x in (mu,C,K,W,H)];ft={k:fnp.asarray(v) for k,v in t.items()}
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        actual=correction(fnp,flops.stats.norm.cdf,9,*args[:3],.02,*args[3:],ft)
        repeat=correction(fnp,flops.stats.norm.cdf,9,*args[:3],.02,*args[3:],ft)
    errors=[float(np.linalg.norm(np.asarray(x)-y)/np.linalg.norm(y)) for x,y in zip(actual,expected)];assert max(errors)<=3e-5
    assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(actual,repeat));assert all(np.array_equal(np.diag(np.asarray(x)),np.zeros(n)) for x in actual)
    pi=rng.permutation(n);pj=rng.permutation(n);permuted=correction(np,cdf,9,mu[pi],C[np.ix_(pi,pi)],K[np.ix_(pi,pi)],.02,W[np.ix_(pj,pi)],H[pj],t)
    perrors=[float(np.linalg.norm(x-y[np.ix_(pi,pi)])/np.linalg.norm(y)) for x,y in zip(permuted,expected)];assert max(perrors)<=3e-5
    assert np.array_equal(np.asarray(actual[2]),np.asarray(actual[2]).T)
    t['output']*=0;zeros=correction(np,cdf,9,mu,C,K,.02,W,H,t);assert all(np.array_equal(x,np.zeros_like(x)) for x in zeros)
    kernels.append(dict(width=n,fnp_relative=errors,permutation_relative=perrors,repeat_exact=True,zero_exact=True))

n=1024;q=rng.normal(0,.1,(n,7));C=q@q.T+np.eye(n);mu=rng.normal(0,.3,n);K=rng.normal(0,.02,(n,n))
W=rng.normal(0,.04,(n,n)).astype(np.float32);H=rng.normal(size=(n,64)).astype(np.float32);k22=np.ones((n,n));T31=np.zeros((n,n))
t=initialize();t['output']=rng.normal(0,.1,(32,3)).astype(np.float32)
t.update(feature_scale=np.ones(140,np.float32),rho_scale=np.array(.2,np.float32),third_scale=np.array(.05,np.float32),target_scale=np.array([.005,.01,.006],np.float32))
with flops.BudgetContext(flop_budget=2**41,quiet=True):args=[fnp.asarray(x) for x in (mu,C,K,W,H,k22,T31)]
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    ft={k:fnp.asarray(v) for k,v in t.items()};da,db,dc=correction(fnp,flops.stats.norm.cdf,9,*args[:3],.02,*args[3:5],ft)
    ka=args[2]+da;kb=args[6]+db;kc=args[5]+dc
bill=ctx.summary_dict()['flops_used'];assert bill*7<=15_000_000_000
assert all(np.isfinite(np.asarray(x)).all() for x in (ka,kb,kc))
write(HERE/'unit_result.json',dict(passed=True,centering=centering,gaussian_variance_wick=wick,directed_source_responses=responses,gradient_checks=gradchecks,kernels=kernels,
      one_layer_cold_component_flops=bill,conservative_seven_layer_flops=bill*7,actual_full_candidate_billed=False,
      wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,component_flops=bill*7,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),flush=True)
