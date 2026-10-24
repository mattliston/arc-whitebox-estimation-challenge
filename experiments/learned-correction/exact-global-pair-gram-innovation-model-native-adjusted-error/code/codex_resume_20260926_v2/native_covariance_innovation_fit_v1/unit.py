"""Native helper fidelity against independent NumPy and frozen model states."""
from pathlib import Path
import sys,types,json,hashlib
import numpy as np
import flopscope as flops
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'recurrent_covariance_observer_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
for e in read(HERE/'build.json')['sources']:assert pin(e['path'])==e
for e in read(HERE/'unit_seal.json')['sources']:assert pin(e['path'])==e
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('covariance_supplier_unit');mod.__file__=str(HERE/'candidate.py');sys.modules[mod.__name__]=mod
exec(compile((HERE/'candidate.py').read_text(),mod.__file__,'exec'),mod.__dict__);mod._cf_init()
P={}
for name in ('model.npz','normalizers.npz'):
    with np.load(HERE/name) as z:P.update({k:z[k].copy() for k in z.files})
def reference(F,C,K,s,m):
    F=np.clip(F.astype('float32')/P['feature_scale'],-8,8)
    R=(C/np.outer(s,s)).astype('float32')/P['rho_scale']
    S=(K/np.outer(s*s,s)).astype('float32')/P['third_scale']
    L=F@P['left.weight'].T;B=F@P['right.weight'].T;parts=[]
    for lo in range(0,len(F),64):
        hi=min(lo+64,len(F));h=L[lo:hi,None,:]+B[None,:,:]+P['bias']
        h=h+R[lo:hi,:,None]*P['edge.weight'][:,0]
        h=h+S[lo:hi,:,None]*P['edge.weight'][:,1]
        h=h+S[:,lo:hi].T[:,:,None]*P['edge.weight'][:,2]
        parts.append((np.tanh(h)@P['output.weight'].T+P['output.bias'])[:,:,0])
    raw=np.concatenate(parts);A=(raw+raw.T).astype('float64')*(.001*np.outer(s,s))
    # Solve the multiplier system directly, independently of its rank-one inverse.
    a=(A+A.T)*.5;np.fill_diagonal(a,0);h=m/np.linalg.norm(m)
    G=np.diag(1-2*h*h)+np.outer(h,h)
    u=np.linalg.solve(G,2*(a@h))
    D=a-(np.outer(u,h)+np.outer(h,u))*.5;np.fill_diagonal(D,0)
    return D
def check(F,C,K,s,m,expected,label):
    arrays=[F,C,K,s,m];before=[x.tobytes() for x in arrays]
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        D=np.asarray(mod._cf_delta(*[mod.fnp.asarray(x) for x in arrays]))
    assert all(x.tobytes()==b for x,b in zip(arrays,before))
    assert np.isfinite(D).all() and np.all(np.diag(D)==0)
    assert np.max(abs(D-D.T))<1e-14 and np.max(abs(D@(m/np.linalg.norm(m))))<1e-10
    ratio=float(np.linalg.norm(D-expected)/max(np.linalg.norm(expected),1e-30))
    assert ratio<=2e-5,(label,ratio)
    row=dict(label=label,relative_l2=ratio,max_abs=float(np.max(abs(D-expected))),helper_flops=int(ctx.summary_dict()['flops_used']))
    print(json.dumps(row),flush=True);return row
rng=np.random.default_rng(2026092747);rows=[]
for label,n in (('tiny_signed',9),('tiny_asymmetric',13)):
    F=rng.normal(size=(n,94)).astype('float32')*P['feature_scale'];s=rng.uniform(.4,1.2,size=n)
    C=rng.normal(size=(n,n))*.015*np.outer(s,s);K=rng.normal(size=(n,n))*.006*np.outer(s*s,s)
    m=rng.uniform(.6,1.1,size=n)
    rows.append(check(F,C,K,s,m,reference(F,C,K,s,m),label))
capture={x['mini']:x['packet'] for x in read(OLD/'capture/complete.json')['rows']}
forecast={x['mini']:x['packet'] for x in read(HERE/'fit_result.json')['rows']}
for mini in (57,68):
    c,p=capture[mini],forecast[mini];assert pin(c['path'])==c and pin(p['path'])==p
    with np.load(c['path']) as a,np.load(p['path']) as z:
        for l in range(8,15):
            D=np.zeros((1024,1024));ij=np.triu_indices(1024,1);D[ij]=z[f'D_{l}_upper'];D=D+D.T
            args=[a[f'{key}_{l}'].copy() for key in ('F','C','K','sigma','m')]
            rows.append(check(*args,D,f'mini{mini:04d}_post{l}'))
with (HERE/'unit.json').open('x') as f:json.dump(dict(passed=True,rows=rows,
    model=pin(HERE/'model.npz'),candidate=pin(HERE/'candidate.py'),input_ownership_passed=True),f,indent=2)
print('NATIVE_HELPER_QUALIFIED',flush=True)
