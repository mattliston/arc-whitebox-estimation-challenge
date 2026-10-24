"""Size and source of the strict Gaussian-domain failure; no truth or repair."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CAP = HERE.parent / 'astra_phase2_reexamination_v1/current_comp_joint_cell_attribution_v1/capture_v1'
STORE = HERE.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
meta = json.loads((CAP / 'restart_packet.json').read_text())
spans = {}


def read(packet, name):
    d = meta['packets'][packet][name]['array']
    with (CAP / 'restart_packet.bin').open('rb') as f:
        f.seek(d['offset'])
        data = f.read(d['bytes'])
    assert hashlib.sha256(data).hexdigest() == d['sha256']
    spans[packet + '.' + name] = d['sha256']
    return np.frombuffer(data, dtype=d['dtype']).reshape(d['shape']).copy()


c = read('pre8', 'Cpre')
e = np.linalg.eigvalsh((c+c.T)*.5)
old = read('entry8', 'C')
active = read('entry8', 'previous_alive')
ix = np.flatnonzero(active)
block = old[np.ix_(ix, ix)]
oe = np.linalg.eigvalsh((block+block.T)*.5)
cold_var = old.diagonal()[~active]
row = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][35]
wp = (STORE/row['path']).parent/'weights.npy'
w = np.load(wp, mmap_mode='r')[8].astype(np.float64)
exact = w.T @ ((old+old.T)*.5) @ w
ee = np.linalg.eigvalsh(exact)
rec = dict(schema='gaussian_domain_diagnostic_v1', truth_accessed=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), selected_spans=spans,
    rounded_pre=dict(eigen_min=float(e.min()), eigen_max=float(e.max()),
        negative_count=int(np.sum(e<0)), negative_eigen_rms=float(np.sqrt(np.mean(np.minimum(e,0)**2))),
        negative_energy_fraction=float(np.sum(np.minimum(e,0)**2)/np.sum(e*e))),
    incoming_active=dict(size=int(ix.size), eigen_min=float(oe.min()), eigen_max=float(oe.max()),
        max_asymmetry=float(np.max(np.abs(block-block.T)))),
    incoming_cold=dict(size=int(cold_var.size), variance_min=float(cold_var.min()),
        negative_count=int(np.sum(cold_var<0))),
    exact_fp64_push=dict(eigen_min=float(ee.min()), negative_count=int(np.sum(ee<0)),
        max_change=float(np.max(np.abs(exact-c))), rms_change=float(np.sqrt(np.mean((exact-c)**2)))))
with (HERE/'gaussian_domain_diagnostic.json').open('x') as f:
    json.dump(rec,f,indent=2)
print(json.dumps(rec),flush=True)
