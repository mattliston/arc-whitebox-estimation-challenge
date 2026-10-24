"""Preserve the v1 parity failure; use the exact scalar-final bank mean.

The covariance basis comes from a separate target-blind full-pair endpoint
read of the same prefix. Its endpoint differs by ~1e-7 from the bank's
float64 scalar diagonal read, so it must not supply the scored baseline.
"""
from pathlib import Path

path = Path(__file__).resolve().parent / 'audit_spectral_sample_correction_v1.py'
source = path.read_text()
edits = {
    'from radial_compact_feed_fullpair_v1 import predict':
        'from radial_compact_feed_fullpair_v1 import predict\nfrom compact_fullpair_final_reference_v1 import predict as scalar_predict',
    "f'spectral_sample_correction_bound_r{args.rank}_v1.json'":
        "f'spectral_sample_correction_bound_r{args.rank}_v2.json'",
    '        means = np.maximum(means, 0)':
        '''        full_read_parity = float(np.max(np.abs(np.maximum(means,0)-reference)))
        state = np.zeros_like(state)
        def final_hook(l, m, aux, w):
            h = final_input(final_scalar_features(l,m,aux,w),w,state,P)
            return final_delta(h,aux['sigma'],H)
        means, _ = scalar_predict(W,hook=hook,final_hook=final_hook,collect_states=False)
        means = np.maximum(means, 0)''',
    'receipt = dict(rank=args.rank, baseline_max_abs=parity,':
        'receipt = dict(rank=args.rank, baseline_max_abs=parity, covariance_only_full_read_mean_max_abs=full_read_parity,',
    'target-blind card-covariance basis. Existing':
        'target-blind full-pair covariance endpoint read of the bank prefix; exact scalar-final bank mean baseline. Existing',
}
for old, new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(path), 'exec'), dict(__file__=str(path), __name__='__main__'))
