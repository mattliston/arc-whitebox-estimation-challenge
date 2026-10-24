"""Read-only covariance-error diagnosis on the current matched Phase 2 host.

The archived cov/cov2 pre-covariances are byte-identical, NOT independent
replicas. Report their sampling uncertainty explicitly; do not treat an
observed residual spectrum as a noise-corrected representation ceiling.
No truth enters the estimator trajectory or deployable features.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from radial_compact_feed_fullpair_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step

HERE = Path(__file__).resolve().parent
G19 = HERE.parent / 'public_g19_attribution_v1'
STORE = HERE.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ap = argparse.ArgumentParser()
ap.add_argument('--rank', type=int, required=True)
args = ap.parse_args()
out = HERE / f'current_covariance_error_r{args.rank}_v1.json'
assert not out.exists(), out
manifest = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
W = np.load(STORE / Path(manifest[args.rank]['path']).parent / 'weights.npy', mmap_mode='r')
P = dict(np.load(HERE / 'payload_compact_feed_fullpair_v1.npz'))
truth = np.load(G19 / f'mps_cov2_rank{args.rank:04d}.npz')
truth32 = np.load(G19 / f'mps_truth_rank{args.rank:04d}_indep32M.npz')
state = np.zeros((1024, P['head'].shape[1]), np.float32)
records = []


def frob(A):
    return float(np.sum(A * A))


def summarize(l, C, mu):
    T = truth['mc_pre_cov'][l].astype(np.float64)
    mt = truth['mc_pre_m'][l].astype(np.float64)
    E = T - C
    # Difference in raw second moments; mean centering is an exact identity.
    raw = E + np.outer(mt, mt) - np.outer(mu, mu)
    evals, U = np.linalg.eigh((C + C.T) / 2)
    U = U[:, ::-1]
    e_eval = np.linalg.eigvalsh((E + E.T) / 2)
    energy = frob(E)
    raw_energy = frob(raw)
    dv = truth32['mc_pre_v'][l].astype(np.float64) - np.diag(C)
    def diagonal_metrics(A):
        x = np.diag(A)
        return dict(relative_mse=float(np.mean((x-dv)**2)/np.mean(dv**2)),
                    correlation=float(np.corrcoef(x, dv)[0, 1]))
    projections = []
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        Q = U[:, :k]
        EQ = E @ Q
        both = Q @ (Q.T @ EQ) @ Q.T
        either = EQ @ Q.T + Q @ EQ.T - both
        rq = raw @ Q
        rb = Q @ (Q.T @ rq) @ Q.T
        re = rq @ Q.T + Q @ rq.T - rb
        projections.append(dict(rank=k, covariance_trace_share=float(evals[-k:].sum()/evals.sum()),
            observed_error_both_high_fraction=frob(both)/energy,
            observed_error_either_high_fraction=frob(either)/energy,
            observed_raw_second_either_high_fraction=frob(re)/raw_energy,
            either_diagonal=diagonal_metrics(either), both_diagonal=diagonal_metrics(both)))
    sampling_gaussian = (float(np.trace(T))**2 + frob(T))/int(truth['P'])
    rec = dict(layer=l, covariance_frobenius_squared=frob(C), observed_error_frobenius_squared=energy,
        raw_second_error_frobenius_squared=raw_energy,
        gaussian_approx_sampling_frobenius_squared=sampling_gaussian,
        sampling_fraction_of_observed=sampling_gaussian/energy,
        observed_error_top_eigen_fraction=[float(np.sort(e_eval*e_eval)[-k:].sum()/energy) for k in (1,2,4,8,16)],
        full_observed_diagonal=diagonal_metrics(E), projections=projections)
    records.append(rec)
    print(json.dumps(dict(layer=l, error_norm_squared=energy, sampling_fraction=sampling_gaussian/energy,
                         raw_second_norm_squared=raw_energy, high8_either=projections[3]['observed_error_either_high_fraction'])), flush=True)


def hook(l, m, C, K21, aux, w):
    global state
    if l in (4, 8, 12, 14, 15):
        summarize(l, aux['pre_cov'], aux['alpha']*aux['sigma'])
    delta, state = gnn_step(l, features(l, m, C, K21, aux, w), aux['sigma'], w, state, P)
    return delta


start = time.time()
with threadpool_limits(limits=4):
    mean, _ = predict(W, hook=hook, gate_floor=.01, collect_states=False)
mean = np.maximum(mean, 0)
ref = np.load(HERE / f'compact_feed_fullpair_closedloop_r{args.rank}_v1.npz')['mean']
parity = float(np.max(np.abs(mean-ref)))
assert parity < 1e-12, parity
result = dict(rank=args.rank, trajectory_max_abs=parity, covariance_truth_samples=int(truth['P']),
    diagonal_comparison_samples=int(truth32['P']), records=records, wall_s=time.time()-start,
    interpretation='Read-only observed residual geometry, not a deployable supply. Gaussian sampling estimate omits fourth-cumulant contribution; no independent covariance replicas are available in cov/cov2.')
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(rank=args.rank, parity=parity, wall_s=time.time()-start)), flush=True)
