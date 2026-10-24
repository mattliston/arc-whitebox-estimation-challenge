import gc,math,resource,time
import numpy as np
from threadpoolctl import threadpool_info
from common import *
from model import forward,objective
verify();verify('data_sealed.json');assert all(t['num_threads']==1 for t in threadpool_info())
arm=sys.argv[1];assert arm in ('output_only','nonlinear');hidden=arm=='nonlinear'
with np.load(HERE/'bank_parameters.npz',allow_pickle=False) as z:p0={k:z[k].copy() for k in KEYS}
assert sum(v.size for v in p0.values())==528
p={k:v.copy() for k,v in p0.items()};train={}
for rank in TRAIN:
    with np.load(HERE/f'data_{rank:04d}.npz',allow_pickle=False) as z:train[rank]={k:z[k] for k in z.files}
def cell(rank,j):return {k:(v if k=='factor' else v[j]) for k,v in train[rank].items()}
norms=[]
for rank in TRAIN:
    for j in range(7):
        d=cell(rank,j);u=d['U'].astype(np.float64);w=d['W'].astype(np.float64);r=d['response'].astype(np.float64)
        y=np.sum(w*(u@w),axis=0)*r;norms.append((float(np.mean(u*u)),float(np.mean(y*y))))
        assert np.array_equal(forward(d,p)-d['base'],np.zeros_like(d['base']))
nc,nv=np.mean(norms,axis=0);assert min(nc,nv)>0
update_keys=KEYS if hidden else ('output',)
m={k:np.zeros_like(p[k],dtype=np.float64) for k in update_keys};v={k:x.copy() for k,x in m.items()}
history=[];step=0;start=time.monotonic();clipped=0;maximum_norm=0.
for epoch in range(40):
    totals=np.zeros(3)
    for idx in np.random.default_rng(810926+epoch).permutation(42):
        rank=TRAIN[int(idx)//7];j=int(idx)%7;d=cell(rank,j)
        loss,g,detail=objective(d,p,p0,float(nc),float(nv),hidden=hidden)
        assert math.isfinite(loss) and all(np.isfinite(x).all() for x in g.values())
        norm=math.sqrt(sum(float(np.sum(g[k]*g[k])) for k in update_keys));maximum_norm=max(maximum_norm,norm)
        scale=min(1.,5/max(norm,1e-30));clipped+=scale<1
        lr=5e-5+.5*(5e-4-5e-5)*(1+math.cos(math.pi*step/1679));step+=1
        for k in update_keys:
            grad=g[k]*scale;m[k]=.9*m[k]+.1*grad;v[k]=.999*v[k]+.001*grad*grad
            delta=lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            p[k]=(p[k].astype(np.float64)-delta).astype(np.float32);assert np.isfinite(p[k]).all()
        totals+=loss,detail['cov']/nc,detail['response']/nv
        del d,g;gc.collect() if step%42==0 else None
    rec=dict(epoch=epoch+1,step=step,mean_update_loss=float(totals[0]/42),
        mean_update_cov_ratio=float(totals[1]/42),mean_update_response_ratio=float(totals[2]/42),wall_s=time.monotonic()-start)
    history.append(rec);print(dict(arm=arm,**rec),flush=True)
assert step==1680
if not hidden:assert all(np.array_equal(p[k],p0[k]) for k in KEYS if k!='output')
dest=HERE/f'{arm}_parameters.npz'
with dest.open('xb') as f:np.savez(f,**p)
write(HERE/f'{arm}_fit.json',dict(completed=True,arm=arm,epochs=40,updates=step,train=list(TRAIN),valid_not_decoded=True,
    trained_parameters=528 if hidden else 16,parameters=pin(dest),bank_parameters=pin(HERE/'bank_parameters.npz'),
    covariance_norm=float(nc),response_norm=float(nv),history=history,clipped_steps=clipped,maximum_unclipped_gradient_norm=maximum_norm,
    finite=True,final_epoch_only=True,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,arm=arm,parameters=pin(dest)),flush=True)
