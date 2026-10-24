import gc,resource,time
import numpy as np
from threadpoolctl import threadpool_info
from common import *
from model import forward
verify();done('unit');assert all(t['num_threads']==1 for t in threadpool_info())
rank=int(sys.argv[1]);assert rank in TRAIN+VALID;start=time.monotonic()
row=next(r for r in read(OLD/'captures_sealed.json')['rows'] if r['rank']==rank);check(row['capture'])
with np.load(row['capture']['path'],allow_pickle=False) as z:cap={k:z[k] for k in z.files}
params={k[6:]:v for k,v in cap.items() if k.startswith('param_')};p0={k:params[k] for k in KEYS}
if rank==TRAIN[0]:
    with (HERE/'bank_parameters.npz').open('xb') as f:np.savez(f,**params)
else:
    with np.load(HERE/'bank_parameters.npz',allow_pickle=False) as z:
        assert set(z.files)==set(params)
        assert all(np.array_equal(z[k],params[k]) for k in z.files)
arrays={k:[] for k in ('F','R','S','pf','gf','sf','base','U','W','response')};reconstruction=[]
for l in range(8,15):
    d={k:cap[f'edge_{k}_{l:02d}'] for k in ('F','R','S','pf','gf','sf')};d['factor']=params['cov_factor']
    base=forward(d,p0);ref=cap[f'edge_correction_{l:02d}']
    relative=float(np.linalg.norm(base-ref)/np.linalg.norm(ref));assert relative<=3e-6,(rank,l,relative)
    assert np.array_equal(forward(d,p0)-base,np.zeros_like(base))
    reconstruction.append(dict(layer=l,relative=relative,zero_delta_exact=True))
    for k in ('F','R','S','pf','gf','sf'):arrays[k].append(d[k])
    arrays['base'].append(base)
# Every current input and source reconstruction above is checked before labels.
W=np.load(weights(rank),allow_pickle=False).astype(np.float64)
with np.load(truth(rank),allow_pickle=False) as z:
    assert int(z['P'])==8_000_000
    tm=z['mc_post'].astype(np.float64);tc=z['mc_post_cov'].astype(np.float64)
    pm=z['mc_pre_m'][8].astype(np.float64);pc=z['mc_pre_cov'][8].astype(np.float64)
mr=float(np.linalg.norm(W[8].T@tm[7]-pm)/np.linalg.norm(pm));cr=float(np.linalg.norm(W[8].T@tc[7]@W[8]-pc)/np.linalg.norm(pc))
assert max(mr,cr)<=1e-4
with np.load(OLD/f'statistics_{rank:04d}.npz',allow_pickle=False) as z:stats={k:z[k] for k in ('cov_energy','value_energy')}
sym=lambda x:.5*(x+x.T);layers=[]
for j,l in enumerate(range(8,15)):
    pre={k:cap[f'{k}_{l:02d}'].astype(np.float64) for k in ('sigma','alpha','pdf','gate','mg','C_pre')}
    ep=tm[l-1]-cap[f'm_{l-1:02d}'];Ep=sym(tc[l-1]-cap[f'C_post_{l-1:02d}'])
    U=sym(tc[l]-cap[f'C_post_{l:02d}'])-tangent_cov(pre,ep,Ep,W[l])
    wg=W[l+1];resp=cap[f'pdf_{l+1:02d}']/(2*cap[f'sigma_{l+1:02d}'])
    y=np.sum(wg*(U@wg),axis=0)*resp
    ce=float(np.mean(U*U));ve=float(np.mean(y*y))
    ce_match=abs(ce-stats['cov_energy'][j])/ce;ve_match=abs(ve-stats['value_energy'][j])/ve
    assert max(ce_match,ve_match)<=2e-10,(rank,l,ce_match,ve_match)
    U32=U.astype(np.float32);W32=wg.astype(np.float32);r32=resp.astype(np.float32)
    y32=np.sum(W32*(U32@W32),axis=0)*r32
    ur=float(np.linalg.norm(U32-U)/np.linalg.norm(U));yr=float(np.linalg.norm(y32-y)/np.linalg.norm(y))
    assert max(ur,yr)<=3e-6,(rank,l,ur,yr)
    arrays['U'].append(U32);arrays['W'].append(W32);arrays['response'].append(r32)
    layers.append(dict(layer=l,covariance_energy64=ce,response_energy64=ve,covariance_energy_match_relative=ce_match,
        response_energy_match_relative=ve_match,target_rounding_relative=ur,response_rounding_relative=yr))
    del U,U32,y,y32,pre,Ep;gc.collect()
dest=HERE/f'data_{rank:04d}.npz'
with dest.open('xb') as f:np.savez(f,**{k:np.stack(v) for k,v in arrays.items()},factor=params['cov_factor'])
write(HERE/f'data_{rank:04d}.json',dict(completed=True,rank=rank,role='TRAIN' if rank in TRAIN else 'VALID',
    arrays=pin(dest),capture=row['capture'],truth=pin(truth(rank)),weights=pin(weights(rank)),
    old_statistics=pin(OLD/f'statistics_{rank:04d}.npz'),reconstruction=reconstruction,layers=layers,
    affine_mean_relative=mr,affine_covariance_relative=cr,wall_s=time.monotonic()-start,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,rank=rank,arrays_bytes=dest.stat().st_size,wall_s=time.monotonic()-start),flush=True)
