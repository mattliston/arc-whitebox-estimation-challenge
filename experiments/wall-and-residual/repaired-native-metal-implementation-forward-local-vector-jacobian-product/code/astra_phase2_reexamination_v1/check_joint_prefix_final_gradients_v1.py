"""Target-free complete P0+Q parameter VJP fidelity and resource profile."""
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from full_carrier_bank_autodiff_v1 import BankFullCard
from centered_bank_autodiff_v3 import ExplicitBoundBankFullCard

HERE = Path(__file__).resolve().parent
OUT = HERE/'joint_prefix_final_gradients_v1.json'
PKEYS = ('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','gain')
QKEYS = ('w0','b0','w2','b2','w4','b4')


def array(t):
    return t.detach().cpu().double().numpy().copy()


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def main():
    assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') == '0'
    assert 'threadpoolctl' not in sys.modules
    original = json.loads((HERE/'centered_bank_baseline_mps_f32_v3.json').read_text())
    assert original['passed']
    assert original['card_source_sha256'] == hashlib.sha256((HERE/'centered_bank_autodiff_v3.py').read_bytes()).hexdigest()
    torch.set_num_threads(3)
    start = time.time()
    records, saved = [], {}
    largest_gpu = 0
    for rank in (35,100):
        reference = np.load(HERE/'centered_parent_numerics_v1.npz')[f'baseline_{rank}']
        rng = np.random.Generator(np.random.PCG64DXSM(930496+rank))
        cot = rng.normal(size=1024); cot /= np.linalg.norm(cot)
        saved[f'cotangent_{rank}'] = cot
        gradients, profiles = {}, {}
        for device in ('cpu','mps'):
            stage_start = time.time()
            card = BankFullCard(rank, 'cpu') if device == 'cpu' else ExplicitBoundBankFullCard(rank, 'mps')
            P, Q = dict(card.P0), dict(card.Q0)
            params, names = [], []
            for namespace, keys, prefix in ((P,PKEYS,'P'),(Q,QKEYS,'Q')):
                for key in keys:
                    namespace[key] = namespace[key].clone().requires_grad_()
                    params.append(namespace[key]); names.append(prefix+'_'+key)
            card.Q0 = Q
            prediction, _ = card.run(P=P, training=True)
            prediction = torch.where(prediction < 0, torch.zeros_like(prediction), prediction)
            objective = (prediction[-1]*torch.tensor(cot, device=device, dtype=card.dtype)).sum()
            grads = torch.autograd.grad(objective, params)
            grad_np = {k: array(g) for k,g in zip(names,grads)}
            mean = array(prediction)
            finite = all(np.isfinite(x).all() for x in grad_np.values())
            error = float(np.max(np.abs(mean-reference)))
            if device == 'mps':
                torch.mps.synchronize()
                largest_gpu = max(largest_gpu,int(torch.mps.driver_allocated_memory()))
            profile = dict(rank=rank, device=device, wall_seconds=time.time()-stage_start,
                           parameter_count=sum(x.numel() for x in params), finite=finite,
                           maximum_mean_error=error,
                           group_norms={k:float(np.linalg.norm(v)) for k,v in grad_np.items()},
                           process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            profiles[device] = profile
            gradients[device] = grad_np
            saved[f'prediction_{device}_{rank}'] = mean
            for key,value in grad_np.items(): saved[f'gradient_{device}_{rank}_{key}'] = value
            print(json.dumps(dict(stage='gradient_complete',**profile)),flush=True)
            del card,P,Q,params,names,prediction,objective,grads,grad_np,mean
            gc.collect()
            if device == 'mps': torch.mps.empty_cache()
        group_errors = {k:relative(gradients['mps'][k],gradients['cpu'][k]) for k in gradients['cpu']}
        cpu_flat = np.concatenate([v.ravel() for v in gradients['cpu'].values()])
        mps_flat = np.concatenate([v.ravel() for v in gradients['mps'].values()])
        pooled = relative(mps_flat,cpu_flat)
        groups_pass = all(group_errors[k] <= .01 or np.linalg.norm(gradients['cpu'][k]) < 1e-9
                          for k in group_errors)
        passed = pooled <= .001 and groups_pass and all(p['finite'] and p['maximum_mean_error'] <= 2e-6 for p in profiles.values())
        row = dict(rank=rank,passed=bool(passed),pooled_gradient_relative=pooled,
                   group_relative_errors=group_errors,profiles=profiles)
        records.append(row)
        print(json.dumps(row),flush=True)
    np.savez(OUT.with_suffix('.npz'),**saved)
    result = dict(passed=all(r['passed'] for r in records),records=records,
                  Pkeys=PKEYS,Qkeys=QKEYS,target_access=False,training=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  card_source_sha256=original['card_source_sha256'],
                  output_sha256=hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest(),
                  utc_finished=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-start,
                  peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  largest_observed_gpu_driver_bytes=largest_gpu,
                  qualification='Point VJP equivalence for jointly trainable prefix and actual scalar finalQ; no parameter updates, targets, global smoothness or score gain. CPU process/GPU memory values are not independent additive peaks.')
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    assert result['passed']


if __name__ == '__main__':
    main()
