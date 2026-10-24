"""Independent direct site sums and scalar variance reconciliation."""
from pathlib import Path
import hashlib, json, math
import numpy as np
from scipy.special import ndtr

D=Path(__file__).resolve().parent
pins=json.loads((D/'seal.json').read_text())
for p,h in pins['sources'].items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
rows=[]
def M(a,v):
    s=np.sqrt(v)
    z=np.divide(a,s,out=np.zeros_like(a),where=s>0)
    return np.where(s>0,a*ndtr(z)+s*np.exp(-z*z/2)/math.sqrt(2*math.pi),np.maximum(a,0))
for rank in (35,100):
    folder=D/f'rank{rank:04d}'
    rec=json.loads((folder/'result.json').read_text())
    ref=dict(np.load(folder/'reference.npz'))
    wit=dict(np.load(folder/'witness.npz'))
    st=dict(np.load(folder/'sufficient_statistics.npz'))
    wp=Path(pins['weights'][str(rank)]['path'])
    assert hashlib.sha256(wp.read_bytes()).hexdigest()==pins['weights'][str(rank)]['sha256']
    weights=np.load(wp).astype(np.float64)
    X=wit['X']; n=weights.shape[1]
    assert np.array_equal(X,np.random.default_rng(2709270000+rank).standard_normal((128,n))[:8])
    suffix=[None]*16; suffix[-1]=np.eye(n)
    for l in range(14,-1,-1): suffix[l]=(weights[l+1]*ref['g'][l+1][None,:])@suffix[l+1]
    cp=np.broadcast_to(ref['m'][-1],X.shape).copy()
    ca=cp.copy(); Lpost=np.eye(n)
    xp=X.copy(); xm=-X
    for l,w in enumerate(weights):
        Lpre=w.T@Lpost
        y=X@Lpre.T
        mu,g,m,d=(ref[k][l] for k in ('mu','g','m','d'))
        mp=M(mu+y,d); mm=M(mu-y,d)
        cp+=(mp-m-g*y)@suffix[l]
        ca+=((mp+mm)/2-m)@suffix[l]
        Lpost=g[:,None]*Lpre
        xp=np.maximum(xp@w,0); xm=np.maximum(xm@w,0)
    cp+=X@Lpost.T
    pe=float(np.max(np.abs((xp-cp)-wit['ordinary_residual'])))
    ae=float(np.max(np.abs(((xp+xm)/2-ca)-wit['antithetic_residual'])))
    assert max(pe,ae)<2e-12,(rank,pe,ae)
    errs=[]
    for i,k0 in enumerate(st['keys']):
        k=str(k0)
        components=[]
        for j in range(n):
            s=math.fsum(map(float,st['sums'][:,i,j]))
            q=math.fsum(map(float,st['sums2'][:,i,j]))
            components.append((q-s*s/4096)/4095)
        vv=math.fsum(components)/n
        errs.append(abs(vv-rec['mean_per_output_variances'][k]))
    assert max(errs)<1e-14
    reports=json.loads((folder/'reference_qualification.json').read_text())['reports']
    rows.append(dict(rank=rank,ordinary_direct_sum_max_abs=pe,antithetic_direct_sum_max_abs=ae,
                     variance_fsum_max_abs=max(errs),roundoff_clamps=sum(x['negatives'] for x in reports),
                     circuits=rec['circuits']))
    del weights,suffix,ref,wit,st
passes=any(all(r['circuits'][arm]['passes_variance_screen'] for r in rows)
           for arm in ('ordinary_residual','antithetic_residual'))
assert not passes
result=dict(passed=True,information_gate_passed=passes,rows=rows,source_pins_exact=True,
            control_known_mean_verified=True,actual_worker_or_native_cost_qualified=False,
            validation_accessed=False,submitted=False)
with (D/'verification.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
