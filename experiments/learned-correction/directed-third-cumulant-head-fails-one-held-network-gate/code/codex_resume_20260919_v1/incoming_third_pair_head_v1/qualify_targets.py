import resource,time
from common import *
import numpy as np
from threadpoolctl import threadpool_info
sys.path.insert(0,str(A))
from public_third_centering_v1 import central_plane,raw_f32_rounding_bound,gaussian_third_variance
verify();verify('fit_sources_sealed.json');assert all(x['num_threads']==1 for x in threadpool_info());start=time.monotonic()
ordinal=int(sys.argv[1]);rr=row(ordinal);download=read(HERE/f'download_mini{ordinal:04d}.json');check(download['payload'])
assert download['passed'] and read(HERE/'frozen_inputs_completed.json')['completed']
if rr['role']=='heldout':
    f=read(HERE/'model_frozen.json');check(f['model']);assert f['training_completed']
capture=read(HERE/f'capture_mini{ordinal:04d}.json');check(capture['capture']);check(capture['weights'])
W=np.load(capture['weights']['path'],allow_pickle=False)
allowed=('global_index','mlp_name','n_samples','sample_seed','pre_mean','pre_m2','pre_m3','pre_M11','pre_M21','mean','M11')
with np.load(download['payload']['path'],allow_pickle=False) as z:data={k:z[k] for k in allowed}
identity={k:data[k].item() for k in allowed[:4]};assert identity['global_index']==ordinal and identity['mlp_name']==rr['mlp_name']
count=int(identity['n_samples']);assert count==100000000 and int(identity['sample_seed'])==(20260824^(ordinal*2654435761%2**63))
for k in ('pre_mean','pre_m2','pre_m3','mean'):assert data[k].shape==(16,1024) and data[k].dtype==np.float64 and np.isfinite(data[k]).all()
for k in ('pre_M11','pre_M21','M11'):assert data[k].shape==(16,1024,1024) and data[k].dtype==np.float32 and np.isfinite(data[k]).all()
checks={}
for matrix,marginal in (('pre_M11','pre_m2'),('pre_M21','pre_m3')):
    diag=np.diagonal(data[matrix],axis1=1,axis2=2);units=np.spacing(np.abs(diag)).astype(np.float64)
    error=float(np.max(np.abs(diag.astype(np.float64)-data[marginal])/np.maximum(units,1e-45)))
    checks[matrix+'_diagonal']=dict(max_ulps=error,passed=error<=2)
w0=W[0].astype(np.float64);C0=w0.T@w0;v0=np.diag(C0);mu0=data['pre_mean'][0]
empC=data['pre_M11'][0].astype(np.float64)-np.outer(mu0,mu0)
zm=mu0/np.sqrt(v0/count);zc=(empC-C0)/np.sqrt((np.outer(v0,v0)+C0*C0)/count)
z3=central_plane(mu0,data['pre_m2'][0],data['pre_M11'][0],data['pre_M21'][0])/np.sqrt(gaussian_third_variance(C0,count))
for name,z,limit,maxlimit in [('input_mean',zm,2,12),('input_covariance',zc,2,None),('input_third',z3,1.5,12)]:
    rms=float(np.sqrt(np.mean(z*z)));maximum=float(np.max(np.abs(z)))
    checks[name]=dict(rms=rms,maximum=maximum,passed=rms<=limit and (maxlimit is None or maximum<=maxlimit))
targets=[];output=HERE/f'targets_mini{ordinal:04d}';output.mkdir(exist_ok=False)
with np.load(capture['capture']['path'],allow_pickle=False) as z:
    for l in LAYERS:
        mu=data['pre_mean'][l];postm=data['mean'][l-1];w=W[l].astype(np.float64)
        C=data['pre_M11'][l].astype(np.float64)-np.outer(mu,mu)
        Cp=data['M11'][l-1].astype(np.float64)-np.outer(postm,postm)
        me=float(np.sqrt(np.mean((mu-w.T@postm)**2)));ce=float(np.linalg.norm(C-w.T@Cp@w)/np.linalg.norm(C))
        checks[f'affine_{l:02d}']=dict(mean_rms=me,covariance_relative=ce,passed=me<=1e-6 and ce<=1e-5)
        K=central_plane(mu,data['pre_m2'][l],data['pre_M11'][l],data['pre_M21'][l]);residual=K-z[f'K_{l:02d}'];np.fill_diagonal(residual,0)
        bound=raw_f32_rounding_bound(mu,data['pre_M11'][l],data['pre_M21'][l]);np.fill_diagonal(bound,0)
        ratio=float(np.linalg.norm(bound)/np.linalg.norm(residual));checks[f'rounding_{l:02d}']=dict(bound_residual_rms_ratio=ratio,passed=ratio<=.05)
        s=z[f'sigma_{l:02d}'];target=residual/(s[:,None]**2*s[None,:]);assert np.isfinite(target).all()
        path=output/f'layer{l:02d}.npy';np.save(path,target);targets.append(dict(layer=l,target=pin(path)))
passed=all(x['passed'] for x in checks.values())
record=dict(passed=passed,ordinal=ordinal,role=rr['role'],identity=identity,checks=checks,targets=targets,decoded_keys=list(allowed),
            public_1B_gt_mean_decoded=False,source=download['payload'],inputs=pin(HERE/f'inputs_mini{ordinal:04d}.json'),
            late_MC_noise_removed=False,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE/f'targets_mini{ordinal:04d}.json',record);print(record,flush=True);assert passed,checks
