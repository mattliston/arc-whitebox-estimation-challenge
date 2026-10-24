"""Pin the exact current radial emission to the previously frozen source cell."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from radial_binary_compact import select
from fourth_emission_oracle_math_v1 import radial_source

HERE=Path(__file__).resolve().parent;OUT=HERE/'fourth_emission_inputs_r35_l12_v1.npz';assert not OUT.exists()
assert json.loads((HERE/'fourth_emission_oracle_certificate_v1.json').read_text())['passed']
f=np.load(HERE/'residual_one_high_inputs_r35_l12_v1.npz');s=np.load(HERE/'pair_centering_source_r35_l12_v1.npz')
assert np.array_equal((s['post_cov']+s['post_cov'].T)/2,f['host_covariance'])
C=s['pre_cov'];v=np.diag(C);sigma=np.sqrt(np.maximum(v,1e-10));alpha=s['mu']/sigma
g=ndtr(alpha);_,alive=select(g,.01);_,births=select(np.minimum(g,1-g),.01)
d=np.exp(-alpha*alpha/2)/(np.sqrt(2*np.pi)*sigma)
radial=float(np.mean(s['k4']/(v*v)));assert np.max(np.abs(s['k4']/(v*v)-radial))<1e-12
g=g*alive;d=d*alive;birth_d=d*births
Bpost=f['post_covariance']*alive[:,None]*alive[None,:]
with threadpool_limits(limits=4):M=radial_source(C,g,d,birth_d,radial,Bpost,f['U'])
np.savez(OUT,U=f['U'],g=g,d=d,birth_d=birth_d,radial=radial,Cpre=C,Bpost=Bpost,
         pre_shift=s['mu'],model_source_slices=M,alive=alive,births=births)
record=dict(rank=35,layer=12,active=int(alive.sum()),births=int(births.sum()),radial=radial,
            output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Frozen current selected-hub radial distinct emission plus actual post trace metric. No new target used.')
OUT.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
