"""Actual current carrier, exported action and payload, four held-out nets."""
from pathlib import Path
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_late_congruence_probe_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step

here = Path(__file__).resolve().parent
out = here/'late_congruence_actual_verdict_v1.json'
assert not out.exists()
assert json.loads((here/'late_congruence_training_pilot_v1.json').read_text())['epochs'] == 5
P0 = dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))
P1 = dict(np.load(here/'payload_late_congruence_pilot_v1.npz'))
Q = dict(np.load(here/'action_late_congruence_pilot_v1.npz'))
started = time.time()
rows = []
with threadpool_limits(limits=4):
    for rank in (37, 39, 42, 48):
        folder = here/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
        W = np.load(folder/'weights.npy')
        baseline = np.load(folder/'baseline.npy')
        state = np.zeros((1024, 64), np.float32)
        stats = []
        def hook(l, m, C, K21, aux, w):
            global state
            x = features(l, m, C, K21, aux, w)
            delta, state = gnn_step(l, x, aux['sigma'], w, state, P1 if l >= 10 else P0)
            return delta
        def cov_action(l, m, c, aux):
            if not 10 <= l <= 14:
                return c
            hf = np.clip((state-Q['centers'][l])/Q['scales'][l], -8, 8)
            hf = np.concatenate((hf, np.ones((1024, 1), np.float32)), axis=1)
            sigma = aux['sigma']
            # The carrier has already masked C. Recover the same active mask
            # from its selection rule, not from targets or diagonal values.
            from radial_binary_compact import select
            _, mask = select(aux['gate'], .01)
            u = .02*sigma[:, None]*np.tanh(hf@Q['u'][l-10])*mask[:, None]/32
            v = (hf@Q['v'])*mask[:, None]/(sigma[:, None]*32)
            cv = c@v
            change = u@cv.T+cv@u.T+u@(v.T@cv)@u.T
            stats.append(dict(layer=l, covariance_relative_change=float(np.linalg.norm(change)/np.linalg.norm(c)),
                              u_norm=float(np.linalg.norm(u)), v_norm=float(np.linalg.norm(v))))
            return c+change
        mean, _ = predict(W, hook=hook, cov_action=cov_action, gate_floor=.01,
                          cov32=True, metric='covariance', collect_states=False)
        assert np.array_equal(mean[:10], baseline[:10]), 'prefix drift'
        truth = np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz')['mc_post']
        mse = np.mean((np.maximum(mean, 0)-truth)**2, axis=1)
        old = np.mean((np.maximum(baseline, 0)-truth)**2, axis=1)
        pred = here/f'late_congruence_actual_r{rank}_v1.npz'
        assert not pred.exists()
        np.savez(pred, mean=mean)
        row = dict(rank=rank, prefix_byte_exact=True, final_mse=float(mse[-1]), parent_mse=float(old[-1]),
                   ratio=float(mse[-1]/old[-1]), layer_mse=mse.tolist(), parent_layer_mse=old.tolist(), actions=stats)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ('layer_mse', 'parent_layer_mse', 'actions')}), flush=True)
passed = all(r['ratio'] <= .92 for r in rows)
receipt = dict(rows=rows, actual_heldout_gate_pass=passed, wall_s=time.time()-started,
               qualification='Actual full dynamic carrier on four fixed held-out pilot nets with stored 1e9 truths. Not the 14-net adjusted panel or an official score.')
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(actual_heldout_gate_pass=passed)), flush=True)
