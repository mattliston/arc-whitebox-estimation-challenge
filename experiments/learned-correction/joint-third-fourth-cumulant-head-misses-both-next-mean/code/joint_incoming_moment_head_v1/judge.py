import resource,time
from common import *
import numpy as np
from threadpoolctl import threadpool_info
from data import load_cell
from head import predict_planes
from math_core import forward,losses
verify();verify('fit_sources_sealed.json');assert all(x['num_threads']==1 for x in threadpool_info());start=time.monotonic()
f=read(HERE/'model_frozen.json');assert f['training_completed']
for k in ('model','normalizers','training'):check(f[k])
t=dict(np.load(f['model']['path'],allow_pickle=False));energy=np.load(f['normalizers']['path'],allow_pickle=False)['energies'];records=[]
for ordinal in VALID:
    numerator=np.zeros(5);denominator=np.zeros(5);layers=[]
    for l in LAYERS:
        c=load_cell(ordinal,l);pred=np.stack(predict_planes(np,c['F'],c['R'],c['S'],t))
        other=forward(np.clip(c['F']/t['feature_scale'],-8,8),c['R']/t['rho_scale'],c['S']/t['third_scale'],t,t['target_scale']);assert np.array_equal(pred,other)
        zero=losses(c,np.zeros((3,1024,1024)));value=losses(c,pred);assert np.isfinite(value).all() and np.min(zero)>0
        numerator+=value/energy[l-8];denominator+=zero/energy[l-8]
        layers.append(dict(layer=l,zero_energies=zero.tolist(),model_energies=value.tolist(),ratios=(value/zero).tolist()))
    ratios=numerator/denominator;passed=bool(np.all(ratios[3:]<=[.90,.90]));records.append(dict(ordinal=ordinal,ratios=ratios.tolist(),passed=passed,layers=layers))
    print(records[-1],flush=True)
write(HERE/'information_result.json',dict(completed=True,passed=all(r['passed'] for r in records),records=records,
      metric_order=['weighted_T21','weighted_T31','weighted_T22','joint_directed_covariance','joint_next_mean'],joint_gates=[.90,.90],
      reused_validation=True,model=f['model'],normalizers=f['normalizers'],actual_native_candidate_tested=False,public1B_labels_decoded=False,
      wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(information_passed=all(r['passed'] for r in records)),flush=True)
