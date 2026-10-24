"""Honest unchanged teacher trajectory; new full-cubic invariants only."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step
from conditional_quadratic_tensor_v1 import quadratic_features

here = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--rank', type=int, required=True)
args = ap.parse_args()
assert 35 <= args.rank < 59
out = here/'rows_conditional_quadratic_tensor_probe_v1'/f'row_{args.rank:04d}'
assert not out.exists()
assert json.loads((here/'conditional_quadratic_tensor_certificate_v1.json').read_text())['passed']
out.mkdir(parents=True)
W = np.load(here/'rows_current_transport_shadow_v2'/f'row_{args.rank:04d}'/'weights.npy')
truth = np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{args.rank:04d}.npz')['mc_post'].astype(np.float64)
P = dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))
source = parent.PREDICT_SOURCE
old = '        host = m.copy()'
assert source.count(old) == 1
source = source.replace(old, '        if l >= 10:\n'
                         '            aux["quadratic_tensor"],aux["quadratic_capture"] = quadratic_features(Cpre,pp,lam,W[l],star,bpre,Bpre)\n'+old)
env = dict(parent.env, quadratic_features=quadratic_features)
exec(compile(source, __file__, 'exec'), env)
state = np.zeros((1024, 64), np.float32)
xs, hosts, sigmas, deltas, new, captured = [], [], [], [], [], []


def hook(l, m, C, K21, aux, w):
    global state
    x = features(l, m, C, K21, aux, w)
    delta, state = gnn_step(l, x, aux['sigma'], w, state, P)
    xs.append(x.copy())
    hosts.append(m.copy())
    sigmas.append(aux['sigma'].copy())
    deltas.append(delta.copy())
    if l >= 10:
        new.append(aux['quadratic_tensor'].copy())
        captured.append(aux['quadratic_capture'])
    # Current target first enters state after all current features freeze.
    return truth[l]-m if l >= 1 else np.zeros_like(m)


started = time.time()
with threadpool_limits(limits=2):
    env['predict'](W, hook=hook, gate_floor=.01, cov32=True, metric='covariance', collect_states=False)
x, host, sigma, delta = np.stack(xs), np.stack(hosts), np.stack(sigmas), np.stack(deltas)
base = here/'rows_compact_feed_fullpair_v1'/f'row_{args.rank:04d}'
assert np.array_equal(host, np.load(base/'host.npy')), 'host drift'
assert np.array_equal(sigma, np.load(base/'sigma.npy')), 'sigma drift'
assert np.array_equal(x[:, :, :20], np.load(base/'raw20.npy')), 'raw feature drift'
assert np.array_equal(x[:, :, 20:], np.load(base/'mech.npy')), 'mechanistic feature drift'
old = np.load(here/'rows_joint_plane_probe_v1'/f'row_{args.rank:04d}'/'data.npz')
assert np.array_equal(delta, old['delta']), 'head drift'
feat = np.stack(new)
assert feat.shape == (6, 1024, 8) and np.isfinite(feat).all()
np.save(out/'features.npy', feat)
r = dict(rank=args.rank, parent_byte_exact=True, layers=list(range(10, 16)),
         top8_variance_fraction=captured, feature_rms=np.sqrt(np.mean(feat.astype(np.float64)**2, axis=(0, 1))).tolist(),
         one_step_mse=np.mean((truth-host)**2, axis=1).tolist(), wall_s=time.time()-started,
         provenance='All new contractions and unchanged parent features precede the current target correction. No trajectory mutation.')
(out/'result.json').write_text(json.dumps(r, indent=2)+'\n')
print(json.dumps(dict(rank=args.rank, parent_byte_exact=True, wall_s=r['wall_s'])), flush=True)
