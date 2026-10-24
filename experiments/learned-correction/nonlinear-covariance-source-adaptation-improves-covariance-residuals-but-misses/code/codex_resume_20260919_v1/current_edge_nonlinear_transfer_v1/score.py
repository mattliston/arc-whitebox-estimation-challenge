import resource
import numpy as np
from threadpoolctl import threadpool_info
from common import *
verify();verify('data_sealed.json');verify('models_sealed.json');seal=verify('predictions_sealed.json');done('predict')
assert all(t['num_threads']==1 for t in threadpool_info());rows=[]
for rank in VALID:
    with np.load(HERE/f'data_{rank:04d}.npz',allow_pickle=False) as z:a={k:z[k].astype(np.float64) for k in ('U','W','response')}
    for arm in ('output_only','nonlinear'):
        with np.load(HERE/f'predictions_{arm}_{rank:04d}.npz',allow_pickle=False) as z:delta=z['delta'].astype(np.float64)
        layers=[]
        for j,l in enumerate(range(8,15)):
            u=a['U'][j];w=a['W'][j];r=a['response'][j];p=delta[j]
            y=np.sum(w*(u@w),axis=0)*r;v=np.sum(w*(p@w),axis=0)*r
            cb=float(np.mean(u*u));vb=float(np.mean(y*y));ce=float(np.mean((p-u)**2));ve=float(np.mean((v-y)**2))
            cq=float(np.mean(p*p)-2*np.mean(p*u)+cb);vq=float(np.mean(v*v)-2*np.mean(v*y)+vb)
            assert abs(cq-ce)<=2e-10*max(ce,1e-30) and abs(vq-ve)<=2e-10*max(ve,1e-30)
            layers.append(dict(layer=l,covariance_base=cb,covariance_residual=ce,response_base=vb,response_residual=ve,
                covariance_ratio=ce/cb,response_ratio=ve/vb,covariance_identity_relative=abs(cq-ce)/max(ce,1e-30),response_identity_relative=abs(vq-ve)/max(ve,1e-30)))
        cov=sum(x['covariance_residual'] for x in layers)/sum(x['covariance_base'] for x in layers)
        val=sum(x['response_residual'] for x in layers)/sum(x['response_base'] for x in layers)
        rows.append(dict(rank=rank,arm=arm,layers=layers,covariance_ratio=cov,response_ratio=val))
gates=[]
for rank in VALID:
    control=next(x for x in rows if x['rank']==rank and x['arm']=='output_only')
    nonlinear=next(x for x in rows if x['rank']==rank and x['arm']=='nonlinear')
    relative=nonlinear['response_ratio']/control['response_ratio']
    gates.append(dict(rank=rank,covariance_ratio=nonlinear['covariance_ratio'],response_ratio=nonlinear['response_ratio'],
        response_over_control=relative,passed=nonlinear['covariance_ratio']<=.98 and nonlinear['response_ratio']<=.95 and relative<=.98))
write(HERE/'result.json',dict(completed=True,passed=all(x['passed'] for x in gates),rows=rows,gates=gates,
    prediction_seal=pin(HERE/'predictions_sealed.json'),ordinary_1B_targets_opened=False,native_candidate=False,
    complete_adjusted_score=False,new_samples=0,fit_count=2,full_native_calls=0,
    qualification='Frozen-current-state Gaussian-gain innovation proxy on reused coherent8M references; held networks excluded from both fits. No changed native trajectory or deployable score established.',
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,passed=all(x['passed'] for x in gates),gates=gates,
    rows=[{k:v for k,v in x.items() if k!='layers'} for x in rows]),flush=True)
