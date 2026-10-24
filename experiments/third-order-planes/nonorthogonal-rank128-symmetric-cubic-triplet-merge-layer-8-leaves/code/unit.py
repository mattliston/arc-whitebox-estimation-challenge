"""Materialized tensor checks for factored solves and symmetrized readers."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
from algebra import Products,fit,balance,cp_query,triplet_query,triplet_plane,norms
HERE=Path(__file__).resolve().parent
assert all(t['num_threads']==1 for t in threadpool_info())
seal=json.loads((HERE/'sources_sealed.json').read_text())
for x in seal['sources']+seal['external_pins']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']

def dense_cp(f,lam):return np.einsum('ir,jr,kr,r->ijk',f,f,f,lam)
def dense_triplet(fs):
    t=np.einsum('ir,jr,kr->ijk',*fs)
    return t,sum(t.transpose(p) for p in itertools.permutations(range(3)))/6
def dense_fit(t,r,seed):
    n=t.shape[0];rng=np.random.Generator(np.random.PCG64DXSM(seed))
    fs=[np.zeros((n,r)),np.linalg.qr(rng.standard_normal((n,r)),mode='reduced')[0],np.linalg.qr(rng.standard_normal((n,r)),mode='reduced')[0]]
    for sweep in range(2):
        for i in range(3):
            j,k=[x for x in range(3) if x!=i]
            unfolding=t.transpose(i,j,k).reshape(n,n*n)
            kr=np.einsum('jr,kr->jkr',fs[j],fs[k]).reshape(n*n,r)
            m=unfolding@kr;g=kr.T@kr;rho=1e-8*np.trace(g)/r
            fs[i]=np.linalg.solve(g+rho*np.eye(r),m.T).T
        lengths=np.stack([np.sqrt(np.sum(x*x,axis=0)) for x in fs])
        gm=np.prod(lengths,axis=0)**(1/3)
        fs=[x*(gm/lengths[i])[None,:] for i,x in enumerate(fs)]
    return fs

rng=np.random.Generator(np.random.PCG64DXSM(93092012));checks=[]
def check(name,a,b):
    error=float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(1.,np.linalg.norm(a),np.linalg.norm(b)))
    assert error<=1e-10,(name,error)
    checks.append(dict(name=name,scaled_error=error))

for n,r in [(3,2),(5,3),(7,4)]:
    f=rng.normal(size=(n,9));lam=rng.normal(size=9);t=dense_cp(f,lam)
    fs,solves=fit(f,lam,r,Products(),seed=93092012+n)
    ref=dense_fit(t,r,93092012+n)
    check(f'als_endpoint_n{n}',dense_triplet(fs)[0],dense_triplet(ref)[0])
    for i in range(3):
        j,k=[x for x in range(3) if x!=i]
        fact=f@(lam[:,None]*(f.T@fs[j])*(f.T@fs[k]))
        dense=np.einsum('ijk,jr,kr->ir',t,fs[j],fs[k])
        check(f'mttkrp_n{n}_mode{i}',fact,dense)
    for case,factors in [('fit',fs),('signed',[rng.normal(size=(n,r)) for _ in range(3)]),('zero',[np.zeros((n,r)),rng.normal(size=(n,r)),rng.normal(size=(n,r))])]:
        raw,sym=dense_triplet(factors);a=rng.normal(size=(n,6));b=rng.normal(size=(n,6))
        check(f'{case}_query_n{n}',triplet_query(factors,a,b,Products(),'unit'),np.einsum('ijk,jq,kq->iq',sym,a,b))
        check(f'{case}_plane_n{n}',triplet_plane(factors,Products(),'unit'),np.einsum('iij->ij',sym))
        check(f'{case}_diagonal_n{n}',np.sum(factors[0]*factors[1]*factors[2],axis=1),np.einsum('iii->i',sym))
        for perm in itertools.permutations(range(3)):check(f'{case}_symmetry_n{n}_{perm}',sym,sym.transpose(perm))
        nr=norms(f,lam,factors,Products(),block=4)
        check(f'{case}_norm_target_n{n}',nr['target_norm_squared'],np.sum(t*t))
        check(f'{case}_norm_unsym_n{n}',nr['unsymmetrized_error_squared'],np.sum((t-raw)**2))
        check(f'{case}_norm_sym_n{n}',nr['symmetrized_error_squared'],np.sum((t-sym)**2))
        if case!='zero':
            balanced,_=balance([factors[0]*3,factors[1]*.2,factors[2]/.6])
            check(f'{case}_balance_n{n}',dense_triplet(balanced)[0],raw)
    check(f'cp_query_n{n}',cp_query(f,lam,a,b,Products(),'unit'),np.einsum('ijk,jq,kq->iq',t,a,b))
result=dict(passed=True,checks=checks,maximum_scaled_error=max(c['scaled_error'] for c in checks),checks_count=len(checks),no_convergence_or_recovery_claim=True,truth_accessed=False,new_model_calls=0)
with (HERE/'unit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
