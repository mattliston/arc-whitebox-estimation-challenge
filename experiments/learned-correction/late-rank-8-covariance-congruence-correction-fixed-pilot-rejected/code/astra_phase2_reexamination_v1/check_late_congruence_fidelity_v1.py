"""Prespecified fixed-window fidelity check; never reads training targets."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from current_congruence_shadow_v1 import CongruenceShadow

here = Path(__file__).resolve().parent
out = here/'late_congruence_fidelity_v1.json'
assert not out.exists()
torch.set_num_threads(2)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
started = time.time()
rows = []
for rank in (35, 100):
    z = dict(np.load(here/f'late_congruence_full_response_r{rank}_v1.npz'))
    s = CongruenceShadow(rank, dev)
    u, v = (torch.tensor(z[name], device=dev, dtype=torch.float32) for name in ('u', 'v'))
    with torch.no_grad():
        baseline = s.run_action(u*0, v)
        assert float(baseline.abs().max()) == 0.
        responses = [((s.run_action(eps*u, v)[-1]-s.run_action(-eps*u, v)[-1])/(2*eps)).cpu().numpy()
                     for eps in (.01, .005)]
    r = dict(rank=rank, baseline_perturbation_exact_zero=True,
             full_response_rms=float(np.sqrt(np.mean(z['response'][-1]**2))),
             response_relative=float(np.linalg.norm(responses[0]-z['response'][-1])/np.linalg.norm(z['response'][-1])),
             full_fd_half_relative=float(np.linalg.norm(z['response'][-1]-z['half'][-1])/np.linalg.norm(z['half'][-1])),
             shadow_fd_half_relative=float(np.linalg.norm(responses[0]-responses[1])/np.linalg.norm(responses[1])))
    rows.append(r)
    print(json.dumps(r), flush=True)
    del s
    if dev == 'mps':
        torch.mps.empty_cache()
passed = all(r['response_relative'] <= .1 and r['full_fd_half_relative'] <= .05 and r['shadow_fd_half_relative'] <= .05 for r in rows)
receipt = dict(rows=rows, fidelity_pass=passed, wall_s=time.time()-started,
               qualification='Target-free covariance-action fidelity gate only; no accuracy or release claims. No fitting if this gate fails.')
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(fidelity_pass=passed)), flush=True)
