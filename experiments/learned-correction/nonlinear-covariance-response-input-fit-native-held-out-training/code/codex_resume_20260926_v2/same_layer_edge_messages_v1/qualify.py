"""One target-free forward/null/backward qualification on FIT38."""
from pathlib import Path
import json
import resource
import time
import model
HERE=Path(__file__).resolve().parent
assert not (HERE/'qualification.json').exists()
api=model.initialize();t=api['torch'];np=api['np']
# Independent small-array algebra and permutation checks precede all full passes.
rng=np.random.default_rng(2026092703)
R=rng.normal(size=(5,5)).astype('float32');R=(R+R.T)/2
T=rng.normal(size=(5,5)).astype('float32')
a=np.linspace(-2,2,5,dtype='float32');p=np.exp(-a*a/2)/np.float32(np.sqrt(2*np.pi))
F=rng.normal(size=(5,14)).astype('float32');fs=np.linspace(.5,2,14,dtype='float32')
args=[t.from_numpy(x) for x in (F,R,T,a,p)]
edge={'feature_scale':t.from_numpy(fs)}
actual=api['feature'](*args,edge,'message').numpy()
R0=R.copy();T0=T.copy();np.fill_diagonal(R0,0);np.fill_diagonal(T0,0)
phi=np.stack((p,a*p,(a*a-1)*p),axis=1)
expected=np.concatenate((R0@phi/32,(R0*R0)@phi,T0@phi/32,T0.T@phi/32),axis=1)
assert np.allclose(actual,expected,rtol=2e-6,atol=2e-7)
order=np.array([3,0,4,1,2]);permargs=[F[order],R[np.ix_(order,order)],T[np.ix_(order,order)],a[order],p[order]]
permuted=api['feature'](*[t.from_numpy(x) for x in permargs],edge,'message').numpy()
assert np.allclose(permuted,actual[order],rtol=2e-6,atol=2e-7)
assert np.array_equal(api['feature'](*args,edge,'control').numpy(),(F/fs)[:,:12])
tiny=dict(max_abs=float(np.max(np.abs(actual-expected))),permutation_max_abs=float(np.max(np.abs(permuted-actual[order]))))
rank=38;w=api['weights'](rank)
started=time.monotonic()
def progress(label):
    return lambda layer:print(json.dumps(dict(event='layer',phase=label,layer=layer,
        elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)
with t.no_grad():
    cd=api['card'](w)
    base=api['run'](cd,progress=progress('base')).detach().clone()
    del cd;model.collect()
    zero=t.zeros((2,12,16),dtype=t.float32)
    cd=api['card'](w,zero)
    pred=api['run'](cd,progress=progress('zero')).detach().clone()
    maxabs=float((base-pred).abs().max())
    assert maxabs==0,maxabs
    del cd,pred;model.collect()
theta=t.zeros((2,12,16),dtype=t.float32,requires_grad=True)
cd=api['card'](w,theta)
pred=api['run'](cd,checkpointed=True,progress=progress('gradient'))
assert t.equal(base,pred.detach())
q=t.sin(t.arange(1024,dtype=t.float64)*.6180339887498948)
loss=(pred[-1]*q).mean()*1e4
print(json.dumps(dict(event='backward_start',elapsed_s=time.monotonic()-started)),flush=True)
loss.backward()
assert theta.grad is not None and t.isfinite(theta.grad).all() and float(theta.grad.norm())>0
record=dict(passed=True,rank=rank,tiny_algebra=tiny,zero_max_abs=maxabs,gradient_norm=float(theta.grad.norm()),
            gradient_max_abs=float(theta.grad.abs().max()),objective=float(loss.detach()),
            targets_accessed=False,optimizer_steps=0,binding=api['binding'],
            elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'qualification.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
print(json.dumps(record),flush=True)
