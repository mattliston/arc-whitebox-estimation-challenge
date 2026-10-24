import resource,time
from common import *
import numpy as np
from threadpoolctl import threadpool_info
from moments import central,storage_bounds,gaussian_fourth_variances
verify();verify('fit_sources_sealed.json');assert all(x['num_threads']==1 for x in threadpool_info())
assert read(HERE/'unit_result.json')['passed'];start=time.monotonic();ordinal=int(sys.argv[1]);assert ordinal in TRAIN+VALID
if ordinal in VALID:
    frozen=read(HERE/'model_frozen.json');assert frozen['training_completed']
    for key in ('model','normalizers','training'):check(frozen[key])
old=read(OLD/f'targets_mini{ordinal:04d}.json');download=read(OLD/f'download_mini{ordinal:04d}.json');capture=read(OLD/f'capture_mini{ordinal:04d}.json')
assert old['passed'] and download['passed'] and capture['passed'];assert old['identity']['n_samples']==100000000
check(download['payload']);check(capture['capture']);check(capture['weights'])
allowed=('pre_mean','pre_m2','pre_m3','pre_m4','pre_M11','pre_M21','pre_M31','pre_M22')
with np.load(download['payload']['path'],allow_pickle=False) as z:raw={k:z[k] for k in allowed}
for k in allowed:
    expected=(16,1024,1024) if 'M' in k else (16,1024);dtype=np.float32 if 'M' in k else np.float64
    assert raw[k].shape==expected and raw[k].dtype==dtype and np.isfinite(raw[k]).all()
checks={}
for name in ('pre_M31','pre_M22'):
    d=np.diagonal(raw[name],axis1=1,axis2=2);units=np.abs(np.spacing(d)).astype(np.float64)
    err=float(np.max(np.abs(d.astype(np.float64)-raw['pre_m4'])/np.maximum(units,1e-45)))
    checks[name+'_diagonal']=dict(max_raw_float32_ulps=err,passed=err<=2)
W=np.load(capture['weights']['path'],mmap_mode='r',allow_pickle=False);w0=W[0].astype(np.float64);C0=w0.T@w0
v31,v22=gaussian_fourth_variances(C0,100000000)
first=central(*[raw[k][0] for k in allowed])
for name,plane,var in [('gaussian_K31',first[1],v31),('gaussian_K22',first[2],v22)]:
    z=plane/np.sqrt(var);rms=float(np.sqrt(np.mean(z*z)));maximum=float(np.max(np.abs(z)))
    checks[name]=dict(standardized_rms=rms,standardized_max=maximum,passed=rms<=1.5 and maximum<=12)
directory=HERE/f'targets_mini{ordinal:04d}';directory.mkdir(exist_ok=False);records=[]
with np.load(capture['capture']['path'],allow_pickle=False) as caps:
    for l in LAYERS:
        K21,K31,K22,_,_=central(*[raw[k][l] for k in allowed]);C=caps[f'C_{l:02d}'];v=np.diag(C);s=caps[f'sigma_{l:02d}'];radial=caps[f'radial_{l:02d}'].item()
        k4=radial*v*v;R=C/np.outer(s,s);T31=(k4/(v*v))[:,None]*R
        T22=(radial*(np.outer(v,v)+2*C*C)/3)/np.outer(v,v)
        target=np.stack([(K21-caps[f'K_{l:02d}'])/(v[:,None]*s[None,:]),K31/(s[:,None]**3*s[None,:])-T31,K22/np.outer(v,v)-T22])
        for x in target:np.fill_diagonal(x,0)
        original_entry=next(r for r in old['targets'] if r['layer']==l);check(original_entry['target'])
        expected=np.load(original_entry['target']['path'],allow_pickle=False)*(s*s/v)[:,None]
        err=float(np.linalg.norm(target[0]-expected)/np.linalg.norm(expected));checks[f'third_units_{l:02d}']=dict(relative=err,passed=err<=1e-11)
        bounds=storage_bounds(raw['pre_mean'][l],raw['pre_m2'][l],raw['pre_M11'][l],raw['pre_M21'][l],raw['pre_M31'][l],raw['pre_M22'][l])
        units=(s[:,None]**3*s[None,:],np.outer(v,v))
        for j,name in enumerate(('K31','K22')):
            ratio=float(np.linalg.norm(bounds[j]/units[j])/np.linalg.norm(target[j+1]));checks[f'{name}_rounding_{l:02d}']=dict(bound_residual_rms_ratio=ratio,passed=ratio<=.05)
        assert np.isfinite(target).all();path=directory/f'layer{l:02d}.npy';np.save(path,target)
        records.append(dict(layer=l,target=pin(path),target22_relative_asymmetry=float(np.linalg.norm(target[2]-target[2].T)/np.linalg.norm(target[2]))))
passed=all(x['passed'] for x in checks.values())
write(HERE/f'targets_mini{ordinal:04d}.json',dict(passed=passed,ordinal=ordinal,role='train' if ordinal in TRAIN else 'reused_validation',
      records=records,checks=checks,decoded_keys=list(allowed),source=download['payload'],previous_qualification=pin(OLD/f'targets_mini{ordinal:04d}.json'),
      public1B_labels_decoded=False,late_MC_noise_removed=False,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(ordinal=ordinal,passed=passed,checks=checks),flush=True);assert passed,checks
