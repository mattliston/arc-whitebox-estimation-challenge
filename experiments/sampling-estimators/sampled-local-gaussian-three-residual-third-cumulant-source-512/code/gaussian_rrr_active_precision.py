"""Research-only Gaussian RRR noise under the declared active reference."""
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np

HERE = Path(__file__).resolve().parent
CAP = HERE.parent / 'astra_phase2_reexamination_v1/current_comp_joint_cell_attribution_v1/capture_v1'
STORE = HERE.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
meta = json.loads((CAP/'restart_packet.json').read_text())
assert meta['rank'] == 35 and meta['layer'] == 8 and not meta['truth_accessed']
assert meta['candidate_sha256'] == 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
spans = {}
started = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(packet, name, aux=False):
    p = meta['packets'][packet]
    if aux:
        p = p['aux']['mapping']
    d = p[name]['array']
    with (CAP/'restart_packet.bin').open('rb') as f:
        f.seek(d['offset'])
        data = f.read(d['bytes'])
    assert hashlib.sha256(data).hexdigest() == d['sha256']
    spans[packet+'.'+name] = d['sha256']
    return np.frombuffer(data, dtype=d['dtype']).reshape(d['shape']).copy()


mu, old = read('pre8','mu'), read('entry8','C')
ix = np.flatnonzero(read('entry8','previous_alive'))
alive = read('pre8','alive')
captured = read('pre8','Cpre')
alpha9, pdf9, sigma9 = (read('pair9', k, True) for k in ('alpha','pdf','sigma'))
assert np.all(sigma9>0)
response = -alpha9*pdf9/(6*sigma9**2)
row = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][35]
wp = (STORE/row['path']).parent/'weights.npy'
weight_sha = sha(wp)
assert weight_sha == '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'
weights = np.load(wp,mmap_mode='r')
w8 = weights[8][ix].astype(np.float64)
q = weights[9].T.astype(np.float64)
q2, q3 = q*q, q*q*q
block = old[np.ix_(ix,ix)]
block = (block+block.T)*.5
chol = np.linalg.cholesky(block)
factor = w8.T @ chol
cov = factor @ factor.T
sigma = np.sqrt(np.diag(cov))
assert np.all(sigma>0)
alpha = mu/sigma
gate = .5*(1+np.array([math.erf(a/math.sqrt(2)) for a in alpha]))
pdf = np.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
mg = mu*gate+sigma*pdf
rec = dict(schema='local_gaussian_rrr_active_precision_v1', research_only=True,
    source_sha256=sha(Path(__file__)), protocol_sha256=sha(HERE/'GAUSSIAN_RRR_ACTIVE_PROTOCOL.md'),
    selected_spans=spans, weights_sha256=weight_sha, truth_accessed=False,
    rank=35, layer=8, consumer=9, source_candidate_sha256=meta['candidate_sha256'],
    reference='Symmetric incoming active covariance, exact FP64 affine push; all inactive variances omitted.',
    active_dimension=int(ix.size), max_covariance_difference=float(np.max(np.abs(cov-captured))),
    rms_covariance_difference=float(np.sqrt(np.mean((cov-captured)**2))),
    relative_covariance_rms_difference=float(np.linalg.norm(cov-captured)/np.linalg.norm(captured)),
    pairs_per_batch=512, primary_gate_variance=1e-10, complete_deployment_bill=False, batches=[])
means = []
for seed in (1909198,1909199):
    z = np.random.default_rng(seed).standard_normal((512,ix.size)) @ factor.T
    ys=[]
    for sign in (1.,-1.):
        dz = sign*z
        r = (np.maximum(mu+dz,0)-mg-gate*dz)*alive
        a, b, c = r@q.T, (r*r)@q2.T, (r*r*r)@q3.T
        y = (a*a*a-3*a*b+2*c)*response
        assert np.isfinite(y).all()
        ys.append(y)
    pair = (ys[0]+ys[1])*.5
    mean = pair.mean(axis=0)
    variance = float(np.mean(np.var(pair,axis=0,ddof=1))/512)
    means.append(mean)
    rec['batches'].append(dict(seed=seed, mean_variance_512_pairs=variance,
        iid_mean_variance_512_draws=float(np.mean(np.var(ys[0],axis=0,ddof=1))/512),
        measured_source_mean_square=float(np.mean(mean*mean)),
        precision_gate_passed=variance<=1e-10))
rec['two_batch_difference_mean_square']=float(np.mean((means[0]-means[1])**2))
rec['two_batch_expected_noise_mean_square']=sum(b['mean_variance_512_pairs'] for b in rec['batches'])
rec['independent_batch_cross_mean']=float(np.mean(means[0]*means[1]))
rec['necessary_precision_gate_passed']=all(b['precision_gate_passed'] for b in rec['batches'])
n,r=1024,int(ix.size)
# Includes Cholesky, factor construction, Gaussian acquisition and six projections.
# Excludes diagnostic full covariance, elementwise work, RNG and all future CP costs.
rec['optimistic_matrix_flops_per_cell']=r**3/3+2*n*r*r+2*512*n*r+12*512*n*n
rec['optimistic_matrix_flops_15_cells']=15*rec['optimistic_matrix_flops_per_cell']
rec['conclusion']=('Necessary precision passes; no accuracy or complete cost is established.'
    if rec['necessary_precision_gate_passed'] else
    'The fixed512-pair active-reference Gaussian RRR constructor fails necessary precision; no expansion or tuning.')
rec['elapsed_s']=time.monotonic()-started
rec['max_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with (HERE/'gaussian_rrr_active_precision.json').open('x') as f:
    json.dump(rec,f,indent=2)
print(json.dumps(rec),flush=True)
