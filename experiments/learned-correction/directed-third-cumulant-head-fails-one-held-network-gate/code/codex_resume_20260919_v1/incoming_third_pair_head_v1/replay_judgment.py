"""Independent target centering and full congruence replay of frozen losses."""
import math,resource,time
from common import *
import numpy as np
from threadpoolctl import threadpool_info
from head import predict_s
verify();verify('fit_sources_sealed.json');verify('replay_sources_sealed.json')
assert all(x['num_threads']==1 for x in threadpool_info());start=time.monotonic()
result=read(HERE/'information_result.json');assert result['completed']
f=read(HERE/'model_frozen.json')
for k in ('model','normalizers','training'):check(f[k])
t=dict(np.load(f['model']['path'],allow_pickle=False));energies=np.load(f['normalizers']['path'],allow_pickle=False)['energies'];records=[]
for ordinal in HELDOUT:
    ir=read(HERE/f'inputs_mini{ordinal:04d}.json');tr=read(HERE/f'targets_mini{ordinal:04d}.json');dr=read(HERE/f'download_mini{ordinal:04d}.json')
    check(dr['payload']);original=next(r for r in result['records'] if r['ordinal']==ordinal)
    with np.load(dr['payload']['path'],allow_pickle=False) as z:
        raw={k:z[k] for k in ('pre_mean','pre_m2','pre_M11','pre_M21')}
    capture=read(HERE/f'capture_mini{ordinal:04d}.json');check(capture['capture']);caps=np.load(capture['capture']['path'],allow_pickle=False)
    cell_terms=[];layer_records=[]
    for ri,rt in zip(ir['records'],tr['targets']):
        l=ri['layer'];check(ri['input']);check(rt['target']);c=dict(np.load(ri['input']['path'],allow_pickle=False))
        saved=np.load(rt['target']['path'],allow_pickle=False);mu=raw['pre_mean'][l]
        centered=raw['pre_M21'][l].astype(np.float64)-2*mu[:,None]*raw['pre_M11'][l].astype(np.float64)
        centered-=raw['pre_m2'][l][:,None]*mu[None,:];centered+=2*mu[:,None]**2*mu[None,:]
        s=c['sigma'];target=(centered-caps[f'K_{l:02d}'])/(s[:,None]**2*s[None,:]);np.fill_diagonal(target,0)
        terr=float(np.linalg.norm(target-saved)/np.linalg.norm(saved));assert terr<=1e-10,terr
        a=caps[f'alpha_{l:02d}'];p=caps[f'pdf_{l:02d}'];g=caps[f'gate_{l:02d}']
        R=caps[f'C_{l:02d}']/np.outer(s,s);np.fill_diagonal(R,0)
        jet=[g,p,-a*p,(a*a-1)*p];L=np.zeros_like(R)
        for r in range(3):L+=(R**r/math.factorial(r))*np.outer(jet[r+1],jet[r])
        L*=np.outer(s,s)/2;np.fill_diagonal(L,0)
        lerr=float(np.linalg.norm(L-c['L'])/np.linalg.norm(c['L']));assert lerr<=1e-13,lerr
        pred=predict_s(np,c['F'],c['R'],c['S'],t);W=c['Wnext'].astype(np.float64);off=1024*1023
        terms=[]
        for e in (-saved,pred.astype(np.float64)-saved):
            np.fill_diagonal(e,0);q=L*e;cov=q+q.T;np.fill_diagonal(cov,0)
            response=np.diag(W.T@cov@W)*c['gnext']
            terms.append(np.array([np.dot(e.ravel(),(c['weight']*e).ravel())/off,np.vdot(cov,cov)/off,
                                   math.fsum(float(x)*float(x) for x in response)/1024]))
        ol=next(x for x in original['layers'] if x['layer']==l)
        expected=np.array([ol['zero_energies'],ol['model_energies']]);terms=np.array(terms)
        error=float(np.max(np.abs(terms-expected)/expected));assert error<=5e-12,error
        cell_terms.append(terms/energies[l-8]);layer_records.append(dict(layer=l,target_relative=terr,kernel_relative=lerr,energy_relative=error))
    pooled=np.array([[math.fsum(float(v[arm,k]) for v in cell_terms) for k in range(3)] for arm in range(2)])
    ratios=pooled[1]/pooled[0];rerr=float(np.max(np.abs(ratios-original['ratios'])/ratios));assert rerr<=5e-12
    passed=bool(np.all(ratios<=np.array([.85,.85,.90])));assert passed==original['passed']
    records.append(dict(ordinal=ordinal,ratios=ratios.tolist(),passed=passed,pooled_relative=rerr,layers=layer_records))
    caps.close();del raw;print(records[-1],flush=True)
assert all(r['passed'] for r in records)==result['passed']
write(HERE/'judgment_replay.json',dict(passed=True,information_gate_passed=result['passed'],records=records,
      unchanged_frozen_model=f['model'],new_fit=False,public1B_labels_decoded=False,wall_s=time.monotonic()-start,
      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
