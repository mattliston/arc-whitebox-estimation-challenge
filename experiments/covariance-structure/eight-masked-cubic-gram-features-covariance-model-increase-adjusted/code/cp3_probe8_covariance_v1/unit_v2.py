"""Native probe algebra and all 42 frozen Torch supplier outputs; root runs."""
import sys,types
import flopscope as flops
from common import *

verify()
for e in read(HERE/'native_seal_v2.json')['sources']:assert pin(e['path'])==e
probe_only=sys.argv[1:]==['--probe-only']
assert not sys.argv[1:] or probe_only
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
mod=types.ModuleType('cp8_native_helper');sys.modules[mod.__name__]=mod
if probe_only:
    mod.__file__=str(DEPLOY_PARENT)
    exec(compile(DEPLOY_PARENT.read_text(),str(DEPLOY_PARENT),'exec'),mod.__dict__)
    exec(compile((HERE/'fragment_v2.py').read_text(),str(HERE/'fragment_v2.py'),'exec'),mod.__dict__)
else:
    for seal in ('build.json','unit_seal.json'):
        for e in read(HERE/seal)['sources']:assert pin(e['path'])==e
    mod.__file__=str(HERE/'candidate.py')
    exec(compile((HERE/'candidate.py').read_text(),mod.__file__,'exec'),mod.__dict__)
    mod.Estimator().setup(types.SimpleNamespace(width=1024,depth=16))

def probe_fixture(n,k,seed):
    rng=np.random.default_rng(seed)
    pp=rng.normal(size=(n,k)).astype(np.float32)*np.float32(.2)
    lam=rng.normal(size=k).astype(np.float32)
    s=rng.uniform(.5,1.3,size=n).astype(np.float64)
    signs=[(rng.integers(0,2,size=(n,8),dtype=np.int8).astype(np.float32)*np.float32(2)-np.float32(1)) for _ in range(2)]
    Omega,Psi=signs
    inv=(1/s).astype(np.float32)
    A=pp.T@(Omega*inv[:,None]);B=pp.T@(Psi*inv[:,None])
    expected=(pp@(lam[:,None]*A*B))*inv[:,None]
    # Explicit FP64 rank-three tensor contraction is independent of the factorized order.
    pbar=pp.astype(np.float64)*inv.astype(np.float64)[:,None]
    tensor=np.einsum('ia,ja,ka,a->ijk',pbar,pbar,pbar,lam.astype(np.float64))
    dense=np.einsum('ijk,jd,kd->id',tensor,Omega.astype(np.float64),Psi.astype(np.float64))
    arrays=[pp,lam,s,Omega,Psi];before=[x.tobytes() for x in arrays]
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        actual=np.asarray(mod._cp8_probe(*[mod.fnp.asarray(x) for x in arrays]))
    assert all(x.tobytes()==b for x,b in zip(arrays,before))
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    assert np.array_equal(actual,expected),'Native/NumPy operation-order mismatch'
    relative=float(np.linalg.norm(actual-dense)/max(np.linalg.norm(dense),1e-30))
    assert relative<=2e-6,relative
    return dict(n=n,k=k,seed=seed,operation_order_bytes_exact=True,explicit_tensor_relative_l2=relative,
        input_ownership=True,paid_flops=int(ctx.summary_dict()['flops_used']))

probe_rows=[probe_fixture(7,5,2026092767),probe_fixture(11,13,2026092768)]
if probe_only:
    once(HERE/'probe_unit_v2.json',dict(passed=True,rows=probe_rows,source=pin(HERE/'fragment_v2.py')))
    print(json.dumps(dict(event='probe_unit_complete',passed=True,rows=probe_rows)),flush=True)
    raise SystemExit(0)

P={}
for name in ('model.npz','normalizers.npz'):
    with np.load(HERE/name,allow_pickle=False) as z:P.update({k:z[k].copy() for k in z.files})

def reference(F,C,K,s,m,Z):
    F=np.clip(F.astype(np.float32)/P['feature_scale'],-8,8)
    R=(C/np.outer(s,s)).astype(np.float32)/P['rho_scale']
    S=(K/np.outer(s*s,s)).astype(np.float32)/P['third_scale']
    G=((Z@Z.T)/np.float32(8))/P['global_scale']
    L=F@P['left.weight'].T;B=F@P['right.weight'].T;parts=[]
    for lo in range(0,len(F),64):
        hi=min(lo+64,len(F));z=L[lo:hi,None,:]+B[None,:,:]+P['bias']
        z=z+R[lo:hi,:,None]*P['edge.weight'][:,0]
        z=z+S[lo:hi,:,None]*P['edge.weight'][:,1]
        z=z+S[:,lo:hi].T[:,:,None]*P['edge.weight'][:,2]
        z=z+G[lo:hi,:,None]*P['edge.weight'][:,3]
        parts.append((np.tanh(z)@P['output.weight'].T+P['output.bias'])[:,:,0])
    raw=np.concatenate(parts);A=(raw+raw.T).astype(np.float64)*(.001*np.outer(s,s))
    A=(A+A.T)*.5;np.fill_diagonal(A,0);h=m/np.linalg.norm(m)
    # Independent dense solve, not the deployed rank-one inverse formula.
    u=np.linalg.solve(np.diag(1-2*h*h)+np.outer(h,h),2*(A@h))
    D=A-(np.outer(u,h)+np.outer(h,u))*.5;np.fill_diagonal(D,0)
    return D

def check(arrays,expected,label):
    before=[x.tobytes() for x in arrays]
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        actual=np.asarray(mod._cp8_delta(*[mod.fnp.asarray(x) for x in arrays]))
    assert all(x.tobytes()==b for x,b in zip(arrays,before))
    assert np.isfinite(actual).all() and np.all(np.diag(actual)==0)
    symmetry=float(np.max(np.abs(actual-actual.T)))
    m=arrays[4];projection=float(np.max(np.abs(actual@(m/np.linalg.norm(m)))))
    ratio=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30))
    assert symmetry<1e-14 and projection<1e-10,(label,symmetry,projection)
    assert ratio<=2e-5,(label,ratio)
    row=dict(label=label,relative_l2=ratio,max_abs=float(np.max(np.abs(actual-expected))),
        symmetry_max_abs=symmetry,mean_annihilation_max_abs=projection,
        exact_zero_diagonal=True,input_ownership=True,helper_flops=int(ctx.summary_dict()['flops_used']))
    print(json.dumps(row),flush=True);return row

rng=np.random.default_rng(2026092767);rows=[]
for label,n in (('tiny_signed',9),('tiny_asymmetric',13)):
    F=rng.normal(size=(n,94)).astype(np.float32)*P['feature_scale'];s=rng.uniform(.4,1.2,size=n)
    C=rng.normal(size=(n,n))*.015*np.outer(s,s);K=rng.normal(size=(n,n))*.006*np.outer(s*s,s)
    m=rng.uniform(.6,1.1,size=n);Z=rng.normal(size=(n,8)).astype(np.float32)
    arrays=[F,C,K,s,m,Z];rows.append(check(arrays,reference(*arrays),label))

base,_=tables();forecast={x['mini']:x['packet'] for x in read(HERE/'fit_result.json')['rows']}
assert set(forecast)==set(TRAIN)
for mini in TRAIN:
    c,p=base[mini]['packet'],forecast[mini]
    assert pin(c['path'])==c and pin(p['path'])==p
    with np.load(c['path'],allow_pickle=False) as a,np.load(p['path'],allow_pickle=False) as z:
        for l in LAYERS:
            D=unpack(z[f'D_{l}_upper'])
            arrays=[a[f'{key}_{l}'].copy() for key in ('F','C','K','sigma','m')]
            entry=feature_entry(mini,l)
            arrays.append(checked_npz(entry,['Z'])['Z'])
            rows.append(check(arrays,D,f'mini{mini:04d}_post{l}'))
assert len(rows)==44
once(HERE/'unit_result.json',dict(passed=True,rows=rows,probe_rows=probe_rows,full_width_cells=42,
    model=pin(HERE/'model.npz'),candidate=pin(HERE/'candidate.py'),probes=pin(HERE/'probes.npz'),
    exact_zero_diagonal=True,input_ownership=True,relative_tolerance=2e-5,
    production_setup=True,gram_route='ordinary FP32 width-eight GEMM',
    limitations='Tiny probe algebra and full supplier comparisons use saved Z; no saved full-width P.'))
print('NATIVE_CP8_HELPER_QUALIFIED',flush=True)
