"""Frozen target-free joint payload perturbations, NumPy then native MPS."""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PKEYS = ('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','gain')
QKEYS = ('w0','b0','w2','b2','w4','b4')
CASES = (('zero',0.),('large_plus',.05),('large_minus',-.05),('half_plus',.025),('half_minus',-.025))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',choices=('numpy','mps'),required=True)
    backend = parser.parse_args().backend
    out = HERE/f'joint_prefix_final_finite_{backend}_v1.json'
    artifact = out.with_suffix('.npz')
    payload = HERE/'joint_prefix_final_finite_payloads_v1.npz'
    assert not out.exists() and not artifact.exists()
    assert json.loads((HERE/'joint_prefix_final_gradients_v1.json').read_text())['passed']
    start = time.time()
    if backend == 'numpy':
        assert not payload.exists()
        from threadpoolctl import threadpool_limits
        from joint_prefix_final_numpy_v1 import run
        threadpool_limits(limits=3)
        original = {'P':dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz')),
                    'Q':dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))}
        rng = np.random.Generator(np.random.PCG64DXSM(930596))
        directions = {}
        for prefix,keys in (('P',PKEYS),('Q',QKEYS)):
            for key in keys:
                value = original[prefix][key]
                direction = rng.normal(size=value.shape)
                direction *= max(float(np.sqrt(np.mean(value.astype(np.float64)**2))),1e-3)/np.sqrt(np.mean(direction**2))
                directions[prefix+'_'+key] = direction
        frozen = {}
        for case,epsilon in CASES:
            for prefix,values in original.items():
                for key,value in values.items():
                    delta = directions.get(prefix+'_'+key)
                    frozen[case+'_'+prefix+'_'+key] = value.copy() if delta is None else (value.astype(np.float64)+epsilon*delta).astype(value.dtype)
        np.savez(payload,**frozen)
    else:
        assert 'threadpoolctl' not in sys.modules
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') == '0'
        import torch
        from centered_bank_autodiff_v3 import ExplicitBoundBankFullCard
        torch.set_num_threads(3)
        reference_json = json.loads((HERE/'joint_prefix_final_finite_numpy_v1.json').read_text())
        reference = np.load(HERE/'joint_prefix_final_finite_numpy_v1.npz')
        assert reference_json['baseline_passed']
    frozen = dict(np.load(payload))
    records,saved = [],{}
    largest_gpu = 0
    for rank in (35,100):
        weights = np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'/'weights.npy',mmap_mode='r')
        for case,epsilon in CASES:
            P = {k[len(case)+3:]:v for k,v in frozen.items() if k.startswith(case+'_P_')}
            Q = {k[len(case)+3:]:v for k,v in frozen.items() if k.startswith(case+'_Q_')}
            if backend == 'numpy':
                prediction = run(weights,P,Q)
            else:
                card = ExplicitBoundBankFullCard(rank,'mps')
                p = {k:torch.tensor(v,device='mps',dtype=torch.float32) for k,v in P.items()}
                card.Q0 = {k:torch.tensor(v,device='mps',dtype=torch.float32) for k,v in Q.items()}
                with torch.no_grad():
                    pred,_ = card.run(P=p)
                    prediction = np.maximum(pred.cpu().double().numpy(),0)
                torch.mps.synchronize()
                largest_gpu = max(largest_gpu,int(torch.mps.driver_allocated_memory()))
                del card,p,pred
                gc.collect();torch.mps.empty_cache()
            saved[f'{case}_{rank}'] = prediction
            if case == 'zero':
                baseline = np.load(HERE/'centered_parent_numerics_v1.npz')[f'baseline_{rank}']
                err = float(np.max(np.abs(prediction-baseline)))
            elif backend == 'mps':
                err = float(np.max(np.abs(prediction-reference[f'{case}_{rank}'])))
            else:
                err = None
            record = dict(rank=rank,case=case,epsilon=epsilon,maximum_reference_error=err,
                          finite=bool(np.isfinite(prediction).all()),wall_seconds=time.time()-start)
            records.append(record)
            print(json.dumps(record),flush=True)
    comparisons = []
    if backend == 'mps':
        for rank in (35,100):
            for size,epsilon in (('large',.05),('half',.025)):
                actual = (saved[f'{size}_plus_{rank}'][-1]-saved[f'{size}_minus_{rank}'][-1])/(2*epsilon)
                expected = (reference[f'{size}_plus_{rank}'][-1]-reference[f'{size}_minus_{rank}'][-1])/(2*epsilon)
                relative = float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30))
                comparisons.append(dict(rank=rank,size=size,relative=relative,
                                        reference_response_rms=float(np.sqrt(np.mean(expected**2))),passed=relative<=.05))
    baseline_passed = all(r['maximum_reference_error'] <= 2e-6 for r in records if r['case']=='zero')
    passed = baseline_passed and all(r['finite'] for r in records)
    if backend == 'mps':
        passed = passed and all(r['maximum_reference_error']<=3e-6 for r in records if r['case']!='zero') and all(r['passed'] for r in comparisons)
    np.savez(artifact,**saved)
    result = dict(passed=bool(passed),baseline_passed=bool(baseline_passed),backend=backend,
                  records=records,central_response_comparisons=comparisons,target_access=False,
                  parameter_payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                  utc_finished=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-start,
                  peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  largest_observed_gpu_driver_bytes=largest_gpu,
                  qualification='Independent full nonlinear implementation parity for one fixed target-free joint P/Q direction; not a global linearity, training or score result. MPS stage consumes exact NumPy-stage float32 payload bytes.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    assert passed


if __name__ == '__main__':
    main()
