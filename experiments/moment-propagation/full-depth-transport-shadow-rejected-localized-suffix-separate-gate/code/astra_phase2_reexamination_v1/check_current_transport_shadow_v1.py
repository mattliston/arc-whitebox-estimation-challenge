"""No fitting: exact baseline, six physical responses, one adjoint check."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from current_transport_shadow_v1 import Shadow

HERE = Path(__file__).resolve().parent
out = HERE/'current_transport_shadow_fidelity_v1.json'
assert not out.exists()
torch.set_num_threads(2)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
start = time.time(); rows = []; adjoints = []
for rank in (35,100):
    shadow = Shadow(rank,dev)
    folder = HERE/'rows_current_transport_shadow_v1'/f'row_{rank:04d}'
    probes = dict(np.load(folder/'probes.npz'))
    with torch.no_grad():
        zero = shadow.run()
    zero_error = float(zero.abs().max().cpu())
    assert zero_error <= 3e-7, zero_error
    for l in (4,9,13):
        direction = torch.tensor(probes[f'direction_{l}'],device=dev,dtype=torch.float32)
        control = torch.zeros((16,1024),device=dev); control[l] = direction
        derivs = []
        with torch.no_grad():
            for eps in (2e-4,1e-4):
                plus = shadow.run(controls=eps*control)[-1]
                minus = shadow.run(controls=-eps*control)[-1]
                derivs.append(((plus-minus)/(2*eps)).cpu().numpy())
        target = probes[f'response_{l}'][-1]
        target_half = probes[f'response_half_{l}'][-1]
        row = dict(rank=rank,layer=l,parent_max_abs=zero_error,
                   response_relative=float(np.linalg.norm(derivs[0]-target)/np.linalg.norm(target)),
                   response_cosine=float(np.dot(derivs[0],target)/(np.linalg.norm(derivs[0])*np.linalg.norm(target))),
                   full_fd_half_relative=float(np.linalg.norm(target-target_half)/np.linalg.norm(target_half)),
                   shadow_fd_half_relative=float(np.linalg.norm(derivs[0]-derivs[1])/np.linalg.norm(derivs[1])))
        rows.append(row); print(json.dumps(row),flush=True)
        if l == 9:
            scalar = torch.zeros((),device=dev,requires_grad=True)
            probe = torch.tensor(np.random.default_rng(5000+rank).standard_normal(1024),device=dev,dtype=torch.float32)
            response = shadow.run(controls=scalar*control,training=True)[-1]
            value = (response*probe).sum()
            value.backward()
            ad = float(scalar.grad.cpu())
            fd = float(np.dot(derivs[0],probe.cpu().numpy()))
            ar = dict(rank=rank,autograd=ad,finite_difference=fd,relative=abs(ad-fd)/max(abs(ad),abs(fd),1e-12))
            adjoints.append(ar);print(json.dumps(dict(adjoint=ar)),flush=True)
    del shadow
    if dev == 'mps':
        torch.mps.empty_cache()
passed = all(r['response_relative'] <= .10 and r['full_fd_half_relative'] <= .05 and r['shadow_fd_half_relative'] <= .05 for r in rows)
passed = passed and all(r['relative'] <= .05 for r in adjoints)
receipt = dict(device=dev,rows=rows,adjoints=adjoints,fidelity_pass=passed,wall_s=time.time()-start,
               qualification='Target-free current-carrier perturbation fidelity only; no trained or deployed improvement.')
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(fidelity_pass=passed,wall_s=receipt['wall_s'])),flush=True)
