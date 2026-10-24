"""Byte-certified sidecar histories on the existing coherent8M source rows."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE/'generate_covariance_innovation_probe_v1.py'
source = path.read_text()
old = "mean_truth=np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{args.rank:04d}.npz')['mc_post'].astype(np.float64)\ncov_truth=np.load(g19/f'mps_cov2_rank{args.rank:04d}.npz')"
new = "cov_truth=np.load(g19/f'mps_cov2_rank{args.rank:04d}.npz')\nassert int(cov_truth['P']) == 8000000\nmean_truth=cov_truth['mc_post'].astype(np.float64)"
assert source.count(old) == 1
source = source.replace(old, new)
edits = (
    ('rows_covariance_innovation_probe_v1', 'rows_learned_pair_history_v1'),
    ('import radial_compact_feed_fullpair_v1 as base',
     'import radial_compact_feed_fullpair_v1 as base\nfrom radial_features import features\nfrom radial_gnn_numpy import gnn_step'),
    ("saved={k:[] for k in ('basis','rho','sigma','target','weight','response')}",
     "saved={k:[] for k in ('basis','rho','sigma','target','weight','response','history')}\nP0=dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))\nstate=np.zeros((1024,64),np.float32)"),
    ('def hook(l,m,C,K21,aux,w):',
     "def hook(l,m,C,K21,aux,w):\n    global state\n    x=features(l,m,C,K21,aux,w)\n    _,state=gnn_step(l,x,aux['sigma'],w,state,P0)\n    if 8<=l<=14:saved['history'].append(state.copy())"),
    ('with threadpool_limits(limits=4):', 'with threadpool_limits(limits=2):'),
    ("np.savez(out/'data.npz',**data)",
     "old_path=here/'rows_covariance_innovation_probe_v2'/f'row_{args.rank:04d}/data.npz'\nold=np.load(old_path)\nparity={k:bool(np.array_equal(data[k],old[k])) for k in old.files}\nassert all(parity.values()),(args.rank,parity)\nassert data['history'].shape==(7,1024,64)\nnp.save(out/'history.npy',data['history'])"),
    ("provenance='Current input matrices", "byte_parity=parity, original_row_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),\n    response_payload_sha256=hashlib.sha256((here/'payload_compact_feed_fullpair_v1.npz').read_bytes()).hexdigest(),\n    history_sha256=hashlib.sha256((out/'history.npy').read_bytes()).hexdigest(),\n    protocol='coherent_8m_v2_frozen_P0_sidecar',\n    provenance='Current input matrices"),
)
for before, after in edits:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
exec(compile(source, str(path), 'exec'), dict(__file__=str(path), __name__='__main__', hashlib=hashlib))
