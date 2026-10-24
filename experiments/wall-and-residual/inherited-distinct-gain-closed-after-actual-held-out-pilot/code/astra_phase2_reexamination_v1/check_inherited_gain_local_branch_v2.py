"""Target-free derivative certificate on the diagnosed original branch."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
import torch
from torch.autograd import forward_ad
from inherited_distinct_gain_autodiff_v1 import InheritedDistinctCard

HERE=Path(__file__).resolve().parent
OUT=HERE/'inherited_gain_local_branch_v2.json'
ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
original=json.loads((HERE/'inherited_distinct_gain_fidelity_v1.json').read_text())
diagnosis=json.loads((HERE/'inherited_gain_selection_diagnosis_v1.json').read_text())
saved=np.load(HERE/'inherited_distinct_gain_fidelity_v1.npz')
fixed=np.load(HERE/'inherited_gain_selection_diagnosis_v1.npz')
assert original['passed'] is False and diagnosis['original_three_replays_byte_exact']
theta=torch.tensor(original['theta'],dtype=torch.float64)
torch.set_num_threads(3)
start=time.time();records=[];arrays={}
for rank in (35,100):
    with torch.no_grad(),forward_ad.dual_level():
        dual=forward_ad.make_dual(torch.zeros(15,dtype=torch.float64),theta)
        card=InheritedDistinctCard(rank,dual)
        result,_=card.run()
        primal,tangent=forward_ad.unpack_dual(result.clamp_min(0))
        primal=primal.numpy().copy();tangent=tangent.numpy().copy()
    base=saved[f'base_{rank}']
    finite=saved[f'finite_{rank}'] if rank==35 else fixed['fixed_finite']
    half=saved[f'half_{rank}'] if rank==35 else fixed['fixed_half']
    secants=(finite-base,2*(half-base))
    denominator=max(float(np.linalg.norm(tangent[-1])),1e-30)
    errors=[float(np.linalg.norm(s[-1]-tangent[-1])/denominator) for s in secants]
    maximum=float(np.max(np.abs(primal-base)))
    assert np.isfinite(tangent).all()
    row=dict(rank=rank,baseline_max_abs=maximum,tangent_relative_errors=errors,
             branch='dynamic original' if rank==35 else 'fixed zero membership/order',
             passed=maximum<2e-6 and max(errors)<=.05)
    records.append(row);arrays[f'tangent_{rank}']=tangent
    print(json.dumps(dict(**row,wall_s=time.time()-start)),flush=True)
    del card,result,dual
result=dict(records=records,passed=all(r['passed'] for r in records),
            original_dynamic_fidelity_passed=False,
            wall_s=time.time()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='No labels or fit. Same original two amplitudes. Local-branch derivative fidelity only; original dynamic prerequisite remains failed and original pilot remains unrun. Does not certify globally smooth selection dynamics or any accuracy gain.')
np.savez(ART,**arrays)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
