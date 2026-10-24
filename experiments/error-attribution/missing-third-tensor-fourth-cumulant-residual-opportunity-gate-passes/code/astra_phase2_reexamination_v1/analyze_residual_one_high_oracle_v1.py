"""Two-stream point estimates for the fixed distinct residual one-high cell."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from residual_one_high_oracle_v1 import plane,distinct_plane

HERE=Path(__file__).resolve().parent
OUT=HERE/'residual_one_high_oracle_r35_l12_v1.json'
ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
fpath=HERE/'residual_one_high_inputs_r35_l12_v1.npz'
f=np.load(fpath);fhash=hashlib.sha256(fpath.read_bytes()).hexdigest()
U=f['U'];W=f['next_weight'];indices=f['indices'];model=f['model_slices']
streams={};corrections={};errors={};metadata={};art={}
start=time.time()
with threadpool_limits(limits=4):
    for name in ('A','B'):
        path=HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz'
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['frozen_input_sha256']==fhash
        assert meta['output_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        t=np.load(path)
        assert int(t['P'])==8000000 and np.array_equal(t['indices'],indices)
        E=t['next_K21']-f['model_next_K21']
        delta=t['slices']-model
        full=plane(delta,U,W,indices)
        distinct=distinct_plane(delta,U,W,indices)
        # Small finite-precision departures from tensor-slice consistency.
        N=np.einsum('jb,ajk->abk',U,t['slices'],optimize=True)
        symmetry=float(np.linalg.norm(N-N.transpose(1,0,2))/np.linalg.norm(N))
        assert symmetry<1e-4,symmetry
        errors[name]=E;corrections[name]=dict(full=full,distinct=distinct)
        art.update({f'{name}_error':E,f'{name}_full':full,f'{name}_distinct':distinct})
        metadata[name]=dict(seed_xor=meta['seed_xor'],source_sha256=meta['source_sha256'],
                            output_sha256=meta['output_sha256'],tensor_double_slice_asymmetry=symmetry,
                            slice_difference_norm=float(np.linalg.norm(delta)))
    assert metadata['A']['seed_xor']!=metadata['B']['seed_xor']

    def metrics(arm,weight):
        EA=errors['A']*weight;EB=errors['B']*weight
        FA=corrections['A'][arm]*weight;FB=corrections['B'][arm]*weight
        signal=float(np.mean(EA*EB));correction=float(np.mean(FA*FB))
        alignment=float(.5*np.mean(EA*FB+EB*FA))
        observed=float(.5*np.mean(EA*EA+EB*EB))
        noise=float(.5*np.mean((EA-EB)**2))
        correction_noise=float(.5*np.mean((FA-FB)**2))
        assert signal>0 and correction>0,(signal,correction)
        return dict(baseline_observed_mse=observed,baseline_signal_mse_point_estimate=signal,
                    baseline_noise_mse_point_estimate=noise,baseline_noise_fraction=noise/observed,
                    correction_signal_mse_point_estimate=correction,
                    correction_noise_mse_point_estimate=correction_noise,
                    signal_alignment_point_estimate=alignment,
                    signal_natural_error_ratio=(signal-2*alignment+correction)/signal,
                    signal_correction_error_cosine=alignment/np.sqrt(signal*correction),
                    signal_correction_relative_norm=np.sqrt(correction/signal),
                    A_target_B_correction_ratio=float(np.sum((EA-FB)**2)/np.sum(EA*EA)),
                    B_target_A_correction_ratio=float(np.sum((EB-FA)**2)/np.sum(EB*EB)))

    result={arm:dict(unweighted=metrics(arm,1.),leading_consumer_weighted=metrics(arm,f['consumer_weight'])) for arm in ('full','distinct')}
    gated=result['distinct']['leading_consumer_weighted']
    gate=(gated['signal_natural_error_ratio']<=.7 and gated['signal_correction_error_cosine']>=.6
          and gated['A_target_B_correction_ratio']<1 and gated['B_target_A_correction_ratio']<1)
np.savez(ART,**art,indices=indices,consumer_weight=f['consumer_weight'])
record=dict(rank=35,layer=12,next_layer=13,q=4,queried_next_K21_rows=indices.tolist(),
            frozen_input_sha256=fhash,streams=metadata,results=result,opportunity_gate=bool(gate),
            wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='Two independent8M streams, both disjoint from host-source seed. Cross-products separate sampling noise only as point estimates, not confidence bounds; shared MPS bias remains. Fixed leading K21-consumer weight, not actual final-MSE or adjusted score. Distinct-only arm is the predeclared gate; full one-high arm diagnostic. Oracle residual not available at deployment.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
