"""Fixed14-direction finite-difference response on one current Phase2 net."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from distinct_birth_amplitude_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step

here = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--rank', type=int, choices=(35, 100), required=True)
args = ap.parse_args()
out = here/f'distinct_birth_sensitivity_r{args.rank}_v1.npz'
receipt = out.with_suffix('.json')
assert not out.exists() and not receipt.exists()
assert json.loads((here/'distinct_birth_amplitude_certificate_v1.json').read_text())['passed']
folder = here/'rows_current_transport_shadow_v2'/f'row_{args.rank:04d}'
W = np.load(folder/'weights.npy')
P = dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))


def run(amplitudes):
    state = np.zeros((1024, 64), np.float32)
    def hook(l, m, C, K21, aux, w):
        nonlocal state
        x = features(l, m, C, K21, aux, w)
        delta, state = gnn_step(l, x, aux['sigma'], w, state, P)
        return delta
    return predict(W, hook=hook, birth_amplitudes=amplitudes, gate_floor=.01,
                   cov32=True, metric='covariance', collect_states=False)[0]


started = time.time()
responses = []
epsilon = .02
with threadpool_limits(limits=3):
    baseline = run(np.ones(16))
    assert np.array_equal(baseline, np.load(folder/'baseline.npy'))
    print(json.dumps(dict(rank=args.rank, baseline_byte_exact=True)), flush=True)
    for layer in range(1, 15):
        amplitudes = np.ones(16)
        amplitudes[layer] += epsilon
        plus = run(amplitudes)
        amplitudes[layer] -= 2*epsilon
        minus = run(amplitudes)
        responses.append((plus[-1]-minus[-1])/(2*epsilon))
        print(json.dumps(dict(rank=args.rank, layer=layer, elapsed_s=time.time()-started)), flush=True)
np.savez(out, baseline=baseline, jacobian=np.stack(responses, axis=1), epsilon=epsilon)
r = dict(rank=args.rank, layers=list(range(1, 15)), epsilon=epsilon, baseline_byte_exact=True,
         wall_s=time.time()-started, target_access=False)
receipt.write_text(json.dumps(r, indent=2)+'\n')
print(json.dumps(r), flush=True)
