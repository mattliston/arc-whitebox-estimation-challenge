"""Actual new closed loop, paired with immutable archived matched control."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from mean_response_transport_reference_v1 import ResponseCard
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input, final_delta

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--rank', type=int, required=True)
a = ap.parse_args()
assert a.rank in (37, 39, 42, 48)
OUT = HERE/f'mean_response_transport_pilot_judge_r{a.rank}_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
started = time.time()
W = np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{a.rank:04d}/weights.npy')
pp = HERE/'payload_mean_response_transport_pilot_prefix_v1.npz'
qp = HERE/'payload_mean_response_transport_pilot_final_v1.npz'
P, Q = dict(np.load(pp)), dict(np.load(qp))
state = np.zeros((1024, 64), np.float32)
saved = {}
def hook(l, m, C, K21, aux, w):
    global state
    dm, state = gnn_step(l, features(l, m, C, K21, aux, w), aux['sigma'], w, state, P)
    return dm
def final_hook(l, m, aux, w):
    h = final_input(final_scalar_features(l, m, aux, w), w, state, P)
    saved.update(h=h.copy(), host=m.copy(), sigma=aux['sigma'].copy(), prefix_state=state.copy())
    return final_delta(h, aux['sigma'], Q)
card = ResponseCard(enabled=True, final=True)
with threadpool_limits(limits=2):
    mean, _ = card.predict(W, hook=hook, final_hook=final_hook, gate_floor=.01,
                           cov32=True, metric='covariance')
frozen = np.load(HERE/'rows_mean_response_transport_final_pilot_v1'/f'row_{a.rank:04d}/data.npz')
parity = {k: bool(np.array_equal(saved[k], frozen[k])) for k in saved}
assert all(parity.values()), parity
archive = HERE/f'positive_path_pilot_judge_r{a.rank}_v1.json'
old = json.loads(archive.read_text())
old_predictions = np.load(archive.with_suffix('.npz'))
for branch in ('bank', 'control'):
    names = (('payload_compact_feed_fullpair_v1.npz', 'payload_fullpair_final_head_v1.npz')
             if branch == 'bank' else
             ('payload_positive_path_pilot_prefix_control_v1.npz', 'payload_positive_path_pilot_final_control_v1.npz'))
    for key, name in zip(('P', 'Q'), names):
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == old['payload_hashes'][branch][key]
predictions = {branch: old_predictions[branch] for branch in ('bank', 'control')}
predictions['response'] = np.maximum(mean, 0)
truth = np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{a.rank:04d}.npz')['mc_post'].astype(np.float64)
layers = {k: np.mean((v-truth)**2, axis=1).tolist() for k, v in predictions.items()}
final = {k: v[-1] for k, v in layers.items()}
for branch in ('bank', 'control'):
    assert final[branch] == old['final_mse'][branch]
np.savez(OUT.with_suffix('.npz'), **predictions)
record = dict(rank=a.rank, final_mse=final, layer_mse=layers, final_inputs_byte_exact=parity,
              response_bank_ratio=final['response']/final['bank'],
              response_control_ratio=final['response']/final['control'],
              prefix_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),
              final_sha256=hashlib.sha256(qp.read_bytes()).hexdigest(),
              immutable_control_receipt_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              wall_s=time.time()-started,
              qualification='Actual NumPy response closed loop, final input/state byte parity. Existing archived bank/control predictions and payload hashes verified, original1e9-truth MSE exactly reproduced. Four-net pilot only, not metered/panel/official.')
OUT.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k: v for k, v in record.items() if k != 'layer_mse'}), flush=True)
