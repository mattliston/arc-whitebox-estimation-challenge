"""Fixed perfect-fourth-emission value gate, two independent source streams."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from residual_one_high_oracle_v1 import distinct_plane

HERE=Path(__file__).resolve().parent;OUT=HERE/'fourth_emission_oracle_r35_l12_v2.json';ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
f=np.load(HERE/'residual_one_high_inputs_r35_l12_v1.npz');U=f['U'];W=f['next_weight'];indices=f['indices']
expected=hashlib.sha256((HERE/'fourth_emission_inputs_r35_l12_v1.npz').read_bytes()).hexdigest()
E={};F={};M={};provenance={};start=time.time()
with threadpool_limits(limits=4):
    for name in ('A','B'):
        p=HERE/f'fourth_emission_truth_r35_l12_{name}_v2.npz';meta=json.loads(p.with_suffix('.json').read_text())
        assert meta['frozen_input_sha256']==expected and meta['output_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
        z=np.load(p);tpath=HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz'
        assert meta['previous_stream_sha256']==hashlib.sha256(tpath.read_bytes()).hexdigest()
        t=np.load(tpath);assert int(z['P'])==8000000 and int(z['seed_xor'])==int(t['seed_xor'])
        E[name]=t['next_K21']-f['model_next_K21'];M[name]=z['delta_source_slices']
        F[name]=distinct_plane(M[name],U,W,indices);provenance[name]=meta
    def metrics(weight):
        a=E['A']*weight;b=E['B']*weight;c=F['A']*weight;d=F['B']*weight
        s=float(np.mean(a*b));v=float(np.mean(c*d));dot=float(.5*np.mean(a*d+b*c))
        assert s>0
        return dict(baseline_signal_mse_point_estimate=s,correction_signal_mse_point_estimate=v,
                    correction_noise_mse_point_estimate=float(.5*np.mean((c-d)**2)),
                    signal_error_ratio=(s-2*dot+v)/s,
                    signal_cosine=dot/np.sqrt(s*v) if v>0 else None,
                    correction_relative_norm=np.sqrt(v/s) if v>0 else None,
                    A_target_B_correction_ratio=float(np.sum((a-d)**2)/np.sum(a*a)),
                    B_target_A_correction_ratio=float(np.sum((b-c)**2)/np.sum(b*b)))
    result=dict(unweighted=metrics(1.),weighted=metrics(f['consumer_weight']))
    v=result['weighted'];gate=(v['signal_cosine'] is not None and v['signal_error_ratio']<=.95
          and v['signal_cosine']>=.4 and v['A_target_B_correction_ratio']<1 and v['B_target_A_correction_ratio']<1
          and result['unweighted']['signal_error_ratio']<=1)
np.savez(ART,A_correction=F['A'],B_correction=F['B'],indices=indices,consumer_weight=f['consumer_weight'])
record=dict(rank=35,layer=12,results=result,perfect_fourth_source_gate=bool(gate),
            source_slice_signal_energy=float(np.mean(M['A']*M['B'])),source_slice_noise_energy=float(.5*np.mean((M['A']-M['B'])**2)),
            acquisition=provenance,wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='Oracle true-fourth source minus actual current radial distinct emission. Two independent8M streams; unbiased batch fourth cumulants. Cross-products are sampling-noise point estimates, not confidence bounds or shared-arithmetic-bias removal. Fixed q4/32-row natural-strength gate. No usable fourth predictor, closed-loop, raw or adjusted score.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='acquisition'},indent=2),flush=True)
