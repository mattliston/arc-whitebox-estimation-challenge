"""Fixed second-net source gate; no fitting or parameter selection."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'positive_path_replication_r100_v1.json';assert not OUT.exists()
f=np.load(HERE/'positive_path_replication_inputs_r100_v1.npz');correction=f['changed']-f['parent']
E={name:np.load(HERE/f'positive_path_replication_truth_r100_{name}_v1.npz')['next_K21']-f['parent'] for name in ('A','B')}
def metrics(weight):
    a=E['A']*weight;b=E['B']*weight;c=correction*weight
    signal=float(np.mean(a*b));dot=float(.5*np.mean((a+b)*c));energy=float(np.mean(c*c))
    return dict(signal_error_ratio=(signal-2*dot+energy)/signal,signal_cosine=dot/np.sqrt(signal*energy),
                correction_relative_norm=np.sqrt(energy/signal),A_ratio=float(np.sum((a-c)**2)/np.sum(a*a)),B_ratio=float(np.sum((b-c)**2)/np.sum(b*b)),
                observed_noise_fraction=float(.25*np.mean((a-b)**2)/(.5*np.mean(a*a+b*b))))
result=dict(unweighted=metrics(1.),weighted=metrics(f['consumer_weight']))
v=result['weighted'];passed=v['signal_error_ratio']<=.75 and v['A_ratio']<=.8 and v['B_ratio']<=.8 and result['unweighted']['signal_error_ratio']<=1
record=dict(rank=100,layer=12,results=result,recursive_source_gate=bool(passed),qualification='Second-net fixed source replication; true previous m/C, full next plane. No final score or measured bill.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
