"""Optimistic per-mode opportunity bound, not a sampling estimator."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_compact_feed_fullpair_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input, final_delta

HERE = Path(__file__).resolve().parent
G19 = HERE.parent / 'public_g19_attribution_v1'
G22 = HERE.parent / 'public_g22_operator_features_v1'
STORE = HERE.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, choices=(35, 100), required=True)
    args = ap.parse_args()
    out = HERE / f'spectral_sample_correction_bound_r{args.rank}_v1.json'
    arrays = out.with_suffix('.npz')
    assert not out.exists() and not arrays.exists()
    P = dict(np.load(HERE / 'payload_compact_feed_fullpair_v1.npz'))
    H = dict(np.load(HERE / 'payload_fullpair_final_head_v1.npz'))
    man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    W = np.load(STORE / Path(man[args.rank]['path']).parent / 'weights.npy', mmap_mode='r')
    truth = np.load(G22 / 'truthpost' / f'truthpost_rank{args.rank:04d}.npz')['mc_post'].astype(np.float64)
    reference = np.load(HERE / f'fullpair_scalar_final_closedloop_r{args.rank}_v1.npz')['mean']
    ct = np.load(G19 / f'mps_cov2_rank{args.rank:04d}.npz')
    bank = json.loads((HERE / 'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']
    base_flops = next(x['flops'] for x in bank if x['rank'] == args.rank)
    coeff = json.loads((HERE / 'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
    state = np.zeros((1024, P['head'].shape[1]), np.float32)
    saved = {}

    def hook(l, m, C, K21, aux, w):
        nonlocal state
        if l < 15:
            d, state = gnn_step(l, features(l, m, C, K21, aux, w), aux['sigma'], w, state, P)
            return d
        h = final_input(final_scalar_features(l, m, aux, w), w, state, P)
        d = final_delta(h, aux['sigma'], H).astype(np.float64)
        saved['cov'] = C - np.outer(m, d) - np.outer(d, m) - np.outer(d, d)
        return d

    def price_mm(m, k, n):
        plain = 2*m*k*n-m*n
        ds = (3, 4, 5) if max(m, k, n) <= 1024 else (4,)
        return min(plain, *(sum(a*b for a, b in zip(coeff[str(d)], (m*k*n, m*k, k*n, m*n))) for d in ds))

    def price_samples(N):
        # Shape-limited blocks avoid inventing a huge untested kernel arena.
        # Only analytical products + ReLU are included; other work is omitted.
        return sum(16*(price_mm(min(1024, N-i), 1024, 1024)+min(1024, N-i)*1024)
                   for i in range(0, N, 1024))

    t0 = time.time()
    with threadpool_limits(limits=3):
        means, _ = predict(W, hook=hook, gate_floor=.01, collect_states=False)
        means = np.maximum(means, 0)
        parity = float(np.max(np.abs(means-reference)))
        assert parity < 1e-12, parity
        C = (saved['cov'] + saved['cov'].T)/2
        vals, Q = np.linalg.eigh(C)
        e = Q.T @ (truth[-1]-means[-1])
        true_cov = ct['mc_post_cov'][-1].astype(np.float64)
        cv = np.sum(Q*(true_cov@Q), axis=0)
    E = e*e
    base_raw = float(np.mean(E))
    base_adj = base_raw*max(.1, base_flops/2**41)
    records = []
    for N in (512, 2048, 8192, 32768, 131072):
        sample_flops = price_samples(N)
        util = (base_flops+sample_flops)/2**41
        rec = dict(samples=N, optimistic_sample_flops=sample_flops,
                   optimistic_total_utilization=util)
        for name, variance in (('empirical8M', cv), ('card_model', vals)):
            noise = np.maximum(variance, 0)/N
            gain = np.divide(E, E+noise, out=np.zeros_like(E), where=E+noise > 0)
            risk = (1-gain)**2*E+gain*gain*noise
            raw = float(np.mean(risk))
            adj = raw*max(.1, util)
            rec[name] = dict(oracle_expected_raw=raw, oracle_expected_adjusted=adj,
                             adjusted_ratio=adj/base_adj, raw_ratio=raw/base_raw,
                             oracle_gain_above_half=int(np.count_nonzero(gain > .5)))
        records.append(rec)
    spectrum = []
    for fraction in (.001, .01, .1, 1, 10):
        mask = cv <= fraction*float(np.mean(cv))
        spectrum.append(dict(variance_relative_to_mean=fraction, modes=int(mask.sum()),
                             error_energy_fraction=float(E[mask].sum()/E.sum())))
    receipt = dict(rank=args.rank, baseline_max_abs=parity,
                   baseline_raw=base_raw, baseline_projected_adjusted=base_adj,
                   measured_bank_first_context_flops=base_flops,
                   covariance_reference_samples=int(ct['P']),
                   card_covariance_min_eigenvalue=float(vals.min()),
                   empirical_projected_variance_min=float(cv.min()),
                   empirical_projected_variance_negative_count=int(np.count_nonzero(cv < 0)),
                   empirical_covariance_trace=float(cv.sum()),
                   spectrum=spectrum, records=records, wall_s=time.time()-t0,
                   qualification='Nondeployable per-mode oracle opportunity bound in a target-blind card-covariance basis. Existing documented1e9 means and noisy float32 8M covariance. Cost omits eigensolver, random generation, projection and reductions. No new samples, training, candidate score, measured total bill or runtime claim.')
    np.savez(arrays, error_coordinates=e, empirical_variance=cv, card_variance=vals)
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
