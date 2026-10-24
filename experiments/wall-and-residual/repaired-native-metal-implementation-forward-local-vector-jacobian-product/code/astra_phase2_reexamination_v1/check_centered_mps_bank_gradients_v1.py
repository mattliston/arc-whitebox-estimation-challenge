"""Fixed target-free whole-bank VJP and actual-cell finite differences."""
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from full_carrier_bank_autodiff_v1 import BankFullCard
from centered_bank_autodiff_v3 import ExplicitBoundBankFullCard

HERE = Path(__file__).resolve().parent
OUT = HERE/'centered_mps_bank_gradients_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0'
assert 'threadpoolctl' not in __import__('sys').modules
forward = json.loads((HERE/'centered_bank_baseline_mps_f32_v3.json').read_text())
assert forward['passed']
assert forward['card_source_sha256']==hashlib.sha256((HERE/'centered_bank_autodiff_v3.py').read_bytes()).hexdigest()
torch.set_num_threads(3)


def array(t):
    return t.detach().cpu().double().numpy().copy()


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


records,saves = [],{}
start = time.time()
max_gpu_driver = 0
for rank in (35,100):
    rng = np.random.Generator(np.random.PCG64DXSM(930096+rank))
    cot = rng.normal(size=1024); cot /= np.linalg.norm(cot)
    direction = np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'/'probes.npz')['direction_13']
    reference = np.load(HERE/'centered_parent_numerics_v1.npz')[f'baseline_{rank}']
    masks = np.load(HERE/f'full_carrier_local_cell_r{rank}_v1.npz')['baseline_masks'][:30]
    gradients,predictions = {},{}
    for device in ('cpu','mps'):
        card = BankFullCard(rank,'cpu') if device=='cpu' else ExplicitBoundBankFullCard(rank,'mps',audit_selections=True)
        controls = torch.zeros((16,1024),device=device,dtype=card.dtype,requires_grad=True)
        weights = torch.tensor(cot,device=device,dtype=card.dtype)
        pred,_ = card.run(controls=controls,training=True)
        forward_masks = None if device=='cpu' else np.stack([x[1] for x in card.selections])
        if device=='mps':
            assert len(forward_masks)==30
        objective = (pred[-1]*weights).sum()
        gradient, = torch.autograd.grad(objective,controls)
        gradients[device] = array(gradient)
        predictions[device] = array(pred)
        if device=='mps':
            torch.mps.synchronize()
            max_gpu_driver = max(max_gpu_driver,int(torch.mps.driver_allocated_memory()))
        print(json.dumps(dict(rank=rank,stage='reverse_complete',device=device,
                              finite=bool(np.isfinite(gradients[device]).all()),
                              wall_s=time.time()-start,process_rss_peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))),flush=True)
        del controls,weights,pred,objective,gradient,card
        gc.collect()
        if device=='mps':
            torch.mps.empty_cache()
    grad_rel = relative(gradients['mps'],gradients['cpu'])
    layer_rel = [relative(a,b) for a,b in zip(gradients['mps'],gradients['cpu'])]
    mean_errors = {k:float(np.max(np.abs(np.maximum(v,0)-reference))) for k,v in predictions.items()}
    member_equal = np.array_equal(forward_masks,masks)
    actual_derivative = float(gradients['mps'][13]@direction)
    cpu_derivative = float(gradients['cpu'][13]@direction)
    fd_records = []
    card = ExplicitBoundBankFullCard(rank,'mps',audit_selections=True)
    with torch.no_grad():
        for eps in (2e-4,1e-4):
            responses = []
            stable = []
            for sign in (1.,-1.):
                card.selections = []
                control = torch.zeros((16,1024),device='mps')
                control[13] = torch.tensor(sign*eps*direction,device='mps',dtype=torch.float32)
                prediction,_ = card.run(controls=control)
                # CPUf64 contraction avoids hiding signal in an f32 sum.
                responses.append(array(prediction[-1]))
                stable.append(bool(np.array_equal(np.stack([x[1] for x in card.selections]),masks)))
            fd = float(((responses[0]-responses[1])/(2*eps))@cot)
            rel = abs(fd-actual_derivative)/max(abs(actual_derivative),1e-12)
            fd_records.append(dict(epsilon=eps,finite_difference=fd,relative_to_reverse=rel,
                                   membership_equal_plus_minus=stable,passed=all(stable) and rel<=.05))
            print(json.dumps(dict(rank=rank,stage='finite_difference',**fd_records[-1])),flush=True)
    half_relative = abs(fd_records[0]['finite_difference']-fd_records[1]['finite_difference'])/max(abs(actual_derivative),1e-12)
    finite = all(np.isfinite(x).all() for x in gradients.values())
    passed = finite and max(mean_errors.values())<=2e-6 and member_equal and grad_rel<=.01 and half_relative<=.05 and all(r['passed'] for r in fd_records)
    row = dict(rank=rank,passed=bool(passed),finite=bool(finite),forward_max_errors=mean_errors,
               baseline_membership_equal=bool(member_equal),gradient_relative=grad_rel,
               gradient_relative_by_layer=layer_rel,layer13_reverse_derivative=actual_derivative,
               cpu_layer13_reverse_derivative=cpu_derivative,finite_differences=fd_records,fd_half_relative=half_relative)
    records.append(row)
    for name,g in gradients.items():
        saves[f'gradient_{name}_{rank}'] = g
        saves[f'prediction_{name}_{rank}'] = predictions[name]
    saves[f'cotangent_{rank}'] = cot
    print(json.dumps(row),flush=True)
    del card,prediction,control
    gc.collect()
    torch.mps.empty_cache()
np.savez(OUT.with_suffix('.npz'),**saves)
result = dict(passed=all(r['passed'] for r in records),records=records,target_access=False,fallback_enabled=False,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              card_source_sha256=forward['card_source_sha256'],
              output_sha256=hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest(),
              wall_s=time.time()-start,peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              largest_observed_gpu_driver_bytes=max_gpu_driver,utc_finished=datetime.now(timezone.utc).isoformat(),
              qualification='Target-free local whole-bank gradient/finite-difference fidelity, fixed random projection and archived late control direction. No global smoothness, parameter update, score or deployment claim.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
assert result['passed'],result
