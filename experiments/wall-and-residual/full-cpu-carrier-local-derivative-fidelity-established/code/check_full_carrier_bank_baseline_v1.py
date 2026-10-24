"""Target-free mathematical-bank endpoint parity before a training pilot."""
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from full_carrier_bank_autodiff_v1 import BankFullCard

here = Path(__file__).resolve().parent
out = here/'full_carrier_bank_baseline_v1.json'
arrays = out.with_suffix('.npz')
assert not out.exists() and not arrays.exists()
torch.set_num_threads(3)
start = time.time()
rows,saves = [],{}
for rank in (35,100):
    card = BankFullCard(rank,'cpu')
    with torch.no_grad():
        means,_ = card.run()
    means = means.clamp_min(0).cpu().numpy()
    ref = np.load(here/f'fullpair_scalar_final_closedloop_r{rank}_v1.npz')['mean']
    error = means-ref
    maximum = float(np.max(np.abs(error)))
    row = dict(rank=rank,max_abs=maximum,final_rms=float(np.sqrt(np.mean(error[-1]**2))),
               passed=maximum < 2e-6)
    rows.append(row)
    saves[f'mean_r{rank}'] = means
    print(json.dumps(row),flush=True)
    del card,means
result = dict(rows=rows,passed=all(r['passed'] for r in rows),wall_s=time.time()-start,
              max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              qualification='Offline CPU mathematical scalar-final bank reference parity, not packaged candidate parity, bill, training or score.')
np.savez(arrays,**saves)
out.write_text(json.dumps(result,indent=2)+'\n')
assert result['passed'],result
