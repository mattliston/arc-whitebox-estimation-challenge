import resource,time
from common import *
import numpy as np
from threadpoolctl import threadpool_info
from math_core import forward,losses
from head import predict_s
verify();verify('fit_sources_sealed.json');assert all(x['num_threads']==1 for x in threadpool_info());start=time.monotonic()
f=read(HERE/'model_frozen.json');assert f['training_completed']
for k in ('model','normalizers','training'):check(f[k])
t=dict(np.load(f['model']['path'],allow_pickle=False));energies=np.load(f['normalizers']['path'],allow_pickle=False)['energies'];records=[]
for ordinal in HELDOUT:
    ir=read(HERE/f'inputs_mini{ordinal:04d}.json');tr=read(HERE/f'targets_mini{ordinal:04d}.json');assert ir['passed'] and tr['passed'] and tr['role']=='heldout'
    numerator=np.zeros(3);denominator=np.zeros(3);layers=[]
    for ri,rt in zip(ir['records'],tr['targets']):
        assert ri['layer']==rt['layer'];l=ri['layer'];check(ri['input']);check(rt['target'])
        with np.load(ri['input']['path'],allow_pickle=False) as z:c={k:z[k] for k in ('F','R','S','L','weight','Wnext','gnext')}
        c['target']=np.load(rt['target']['path'],allow_pickle=False)
        pred=predict_s(np,c['F'],c['R'],c['S'],t)
        independent=forward(np.clip(c['F']/t['feature_scale'],-8,8),c['R']/t['rho_scale'],c['S']/t['third_scale'],t,t['target_scale'])
        assert np.array_equal(pred,independent)
        zero=losses(c,np.zeros((1024,1024)));value=losses(c,pred)
        assert np.isfinite(value).all() and np.min(zero)>0
        numerator+=value/energies[l-8];denominator+=zero/energies[l-8]
        layers.append(dict(layer=l,zero_energies=zero.tolist(),model_energies=value.tolist(),ratios=(value/zero).tolist()))
    ratio=numerator/denominator;passed=bool(np.all(ratio<=np.array([.85,.85,.90])))
    records.append(dict(ordinal=ordinal,ratios=ratio.tolist(),passed=passed,layers=layers));print(records[-1],flush=True)
write(HERE/'information_result.json',dict(completed=True,passed=all(r['passed'] for r in records),records=records,
      consumer_order=['weighted_directed_third','covariance_response','next_gaussian_mean_response'],gates=[.85,.85,.90],
      model=f['model'],normalizers=f['normalizers'],new_native_candidate=False,new_official_score=False,
      public1B_labels_opened=0,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(information_passed=all(r['passed'] for r in records)),flush=True)
