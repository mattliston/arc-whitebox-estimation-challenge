"""One analytical squared-response repair after the rank-192 TRAIN2 failure."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
parent = HERE/'candidate_old_local_rank192_trace_v1.py'
sha = 'e8f1404661b080d813e0bd0215b873ec6377f7a990d9b3db563fb7648789f992'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == sha
U = np.array(list(itertools.product([-np.sqrt(3),0,0,0,0,np.sqrt(3)], repeat=3)))
rng = np.random.default_rng(20260926)
F = rng.normal(size=(3,18)); lam = np.r_[np.ones(9),-np.ones(9)]
full = ((U@F)**2)@(F*lam).T
short = ((U[:,:2]@F[:2])**2)@(F*lam).T
correction = F@(lam*(np.sum(F*F,axis=0)-np.sum(F[:2]*F[:2],axis=0)))
error = float(np.max(np.abs((full-short).mean(axis=0)-correction)))
assert error < 1e-11
fragment = HERE/'old_local_rank_response_trace_fragment_v2.py'
source = parent.read_text()+'\n\n'+fragment.read_text()
target = HERE/'candidate_old_local_rank192_response_v2.py'
compile(source,str(target),'exec')
with target.open('x') as f:f.write(source)
record = dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate=str(target),
              sha256=hashlib.sha256(target.read_bytes()).hexdigest(),parent_sha256=sha,
              changed='Add rho*F*(lam*discarded_column_norm_squared) to kept response inner rows.',
              algebra=dict(gaussian_quadrature_mean_error=error,passed=True),
              limitation='Only the mean missing quadratic response is restored; its varying second-chaos component is still approximate.',
              protocol=dict(ranks=[35,100],baseline='both_tail_train2_v6',
                            expand_only_if='TRAIN2 adjusted improvement at least 0.5%; otherwise close this route'),
              status='Unqualified training-only repair')
with (HERE/'old_local_rank_response_manifest_v2.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
