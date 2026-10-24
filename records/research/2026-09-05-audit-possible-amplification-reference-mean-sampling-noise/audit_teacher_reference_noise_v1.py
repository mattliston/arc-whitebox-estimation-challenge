"""Fixed iid1e9 reference-noise model, not a Monte Carlo estimator."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_compact_feed_fullpair_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step

here = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--rank', type=int, choices=(35, 100), required=True)
args = ap.parse_args()
out = here/f'teacher_reference_noise_r{args.rank}_v1.json'
arrays = out.with_suffix('.npz')
assert not out.exists() and not arrays.exists()
folder = here/'rows_current_transport_shadow_v2'/f'row_{args.rank:04d}'
W = np.load(folder/'weights.npy')
truth = np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{args.rank:04d}.npz')['mc_post'].astype(np.float64)
P = dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))


def run(noise):
    supplied = truth+noise
    state = np.zeros((1024, 64), np.float32)
    residuals, hosts, all_x = [], [], []
    def hook(l, m, C, K21, aux, w):
        nonlocal state
        x = features(l, m, C, K21, aux, w)
        delta, state = gnn_step(l, x, aux['sigma'], w, state, P)
        residuals.append(supplied[l]-m-delta)
        hosts.append(m.copy())
        all_x.append(x)
        return supplied[l]-m if l >= 1 else np.zeros_like(m)
    predict(W, hook=hook, gate_floor=.01, cov32=True, metric='covariance', collect_states=False)
    return np.stack(residuals), np.stack(hosts), np.stack(all_x)


started = time.time()
with threadpool_limits(limits=3):
    baseline, host, x = run(np.zeros_like(truth))
    teacher = here/'rows_compact_feed_fullpair_v1'/f'row_{args.rank:04d}'
    assert np.array_equal(host, np.load(teacher/'host.npy'))
    assert np.array_equal(x[:, :, :20], np.load(teacher/'raw20.npy'))
    assert np.array_equal(x[:, :, 20:], np.load(teacher/'mech.npy'))
    print(json.dumps(dict(rank=args.rank, baseline_byte_exact=True)), flush=True)
    rng = np.random.default_rng(971065+args.rank)
    samples = rng.standard_normal((8, 1024))
    activations = []
    h = samples
    for w in W:
        h = np.maximum(h@w.astype(np.float64), 0)
        activations.append(h.copy())
    noise = (np.stack(activations, axis=1)-truth[None, :, :])/np.sqrt(1e9)
    responses = []
    even_responses = []
    for j in range(8):
        plus, _, _ = run(noise[j])
        minus, _, _ = run(-noise[j])
        responses.append((plus-minus)/2)
        even_responses.append((plus+minus)/2-baseline)
        print(json.dumps(dict(rank=args.rank, direction=j, elapsed_s=time.time()-started)), flush=True)
response = np.stack(responses)
even = np.stack(even_responses)
noise_mse = np.mean(response**2, axis=(0, 2))
baseline_mse = np.mean(baseline**2, axis=1)
fraction = float(noise_mse[10:].mean()/baseline_mse[10:].mean())
np.savez(arrays, modeled_reference_noise=noise, residual_noise_response=response,
         nonlinear_even_response=even, baseline_residual=baseline)
r = dict(rank=args.rank, baseline_byte_exact=True, noise_model='iid Gaussian-input sample means with N=1e9; actual sampler metadata unverified',
         noise_directions=8, noise_mse_by_layer=noise_mse.tolist(), baseline_residual_mse_by_layer=baseline_mse.tolist(),
         late_noise_fraction=fraction, late_mean_nonlinear_even_mse=float(np.mean(even[:, 10:]**2)),
         final_noise_mse=float(noise_mse[-1]), final_residual_mse=float(baseline_mse[-1]),
         wall_s=time.time()-started, qualification='Model-based teacher-noise sensitivity, eight second-moment probes. Not measured actual reference error, estimator accuracy, or a general noise-floor claim.')
out.write_text(json.dumps(r, indent=2)+'\n')
print(json.dumps({k: v for k, v in r.items() if not k.endswith('_by_layer')}), flush=True)
