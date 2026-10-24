"""Fixed natural-strength two-net screen; no target fitting."""
from pathlib import Path
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from complete_heavy_path_birth_v1 import predict
from radial_compact_feed_fullpair_v1 import predict as parent_predict
from radial_features import features
from radial_gnn_numpy import gnn_step

here = Path(__file__).resolve().parent
out = here/'complete_heavy_path_birth_screen_v1.json'
assert not out.exists()
assert json.loads((here/'complete_heavy_path_birth_certificate_v1.json').read_text())['passed']
prior = json.loads((here/'compact_correlation_tail_v1.json').read_text())['records']
P = dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))
rows = []
started = time.time()
with threadpool_limits(limits=4):
    for rank in (35, 100):
        folder = here/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
        W = np.load(folder/'weights.npy')
        truth = np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz')['mc_post']
        archived_open = next(r['final_mse'] for r in prior if r['rank'] == rank and r['gaussian_order'] == 6 and r['edgeworth_correlation_order'] == 2)
        parent_mean, _ = parent_predict(W, gate_floor=.01, cov32=True, metric='covariance', collect_states=False)
        current_unclipped = float(np.mean((parent_mean[-1]-truth[-1])**2))
        assert abs(current_unclipped/archived_open-1) < 1e-6, (rank, 'archive parity')
        old_open = float(np.mean((np.maximum(parent_mean[-1], 0)-truth[-1])**2))
        for learned in (False, True):
            state = np.zeros((1024, 64), np.float32)
            def hook(l, m, C, K21, aux, w):
                global state
                x = features(l, m, C, K21, aux, w)
                delta, state = gnn_step(l, x, aux['sigma'], w, state, P)
                return delta
            mean, states = predict(W, hook=hook if learned else None, gate_floor=.01,
                                   cov32=True, metric='covariance', collect_states=True)
            old = float(np.mean((np.maximum(np.load(folder/'baseline.npy')[-1], 0)-truth[-1])**2)) if learned else old_open
            value = float(np.mean((np.maximum(mean[-1], 0)-truth[-1])**2))
            pred = here/f'complete_heavy_path_birth_r{rank}_learned{int(learned)}_v1.npz'
            assert not pred.exists()
            np.savez(pred, mean=mean)
            row = dict(rank=rank, frozen_head=learned, parent_raw_final_mse=old,
                       raw_final_mse=value, ratio=value/old,
                       max_relative_column_shift=max(r['path_shift_relative'] for r in states),
                       layer_mse=np.mean((np.maximum(mean, 0)-truth)**2, axis=1).tolist())
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'layer_mse'}), flush=True)
passed = all(r['ratio'] <= .9 and r['max_relative_column_shift'] < .1 for r in rows if not r['frozen_head'])
receipt = dict(rows=rows, predeclared_raw_gate_pass=passed, wall_s=time.time()-started,
               qualification='Natural-coefficient open-host gate; frozen matched head is diagnostic only. Not a panel/adjusted/official score.')
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(predeclared_raw_gate_pass=passed)), flush=True)
