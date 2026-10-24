"""Fixed full-edge Gaussian source information gate; reuse frozen oracle cell."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from radial_binary_compact import select
from full_edge_gaussian_bond3_v1 import slices
from residual_one_high_oracle_v1 import distinct_plane

HERE=Path(__file__).resolve().parent;OUT=HERE/'full_edge_gaussian_bond3_r35_l12_v1.json';ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
assert json.loads((HERE/'full_edge_gaussian_bond3_certificate_v1.json').read_text())['passed']
f=np.load(HERE/'residual_one_high_inputs_r35_l12_v1.npz')
s=np.load(HERE/'pair_centering_source_r35_l12_v1.npz')
assert np.array_equal((s['post_cov']+s['post_cov'].T)/2,f['host_covariance'])
C=s['pre_cov'];sigma=np.sqrt(np.maximum(np.diag(C),1e-10));alpha=s['mu']/sigma
g=ndtr(alpha);_,alive=select(g,.01)
d=np.exp(-alpha*alpha/2)/(np.sqrt(2*np.pi)*sigma);e=-alpha*d/sigma
g=g*alive;d=d*alive;e=e*alive
start=time.time()
with threadpool_limits(limits=4):
    M=slices(C,g,d,e,f['U']);correction=distinct_plane(M,f['U'],f['next_weight'],f['indices'])
    E={name:np.load(HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz')['next_K21']-f['model_next_K21'] for name in ('A','B')}
    def metrics(weight):
        a=E['A']*weight;b=E['B']*weight;c=correction*weight
        signal=float(np.mean(a*b));dot=float(np.mean((a+b)*c)/2);energy=float(np.mean(c*c))
        return dict(signal_error_ratio=(signal-2*dot+energy)/signal,
                    signal_cosine=dot/np.sqrt(signal*energy),correction_relative_norm=np.sqrt(energy/signal),
                    A_ratio=float(np.sum((a-c)**2)/np.sum(a*a)),B_ratio=float(np.sum((b-c)**2)/np.sum(b*b)))
    result=dict(unweighted=metrics(1.),weighted=metrics(f['consumer_weight']))
v=result['weighted'];gate=v['signal_error_ratio']<=.95 and v['signal_cosine']>=.4 and v['A_ratio']<1 and v['B_ratio']<1 and result['unweighted']['signal_error_ratio']<=1
np.savez(ART,slices=M,correction=correction)
record=dict(rank=35,layer=12,active=int(alive.sum()),host_byte_exact=True,results=result,source_gate=bool(gate),
            source_slice_norm=float(np.linalg.norm(M)),wall_s=time.time()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Complete seven Gaussian three-bond diagrams, full covariance edges, exact distinct q4 slices. Natural coefficient1 and fixed32 next-K21 rows, existing two independent8M streams. Not a fitted or recursive estimator, score, or confidence bound.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
