"""One three-term fit and network-LOO diagnostic on authorized old training data.

No candidate imports, model calls, fresh/OOS access, sweeps, or tuning.
Target is truth minus the preserved matched-C terminal float32 prediction.
Fixed columns: y, mean(y), y^2/mean(y). Each network has weight 1/baseMSE.
Final output uses the incumbent nonnegative clipping and float32 conversion.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
O = D.parents[2]
T = O / 'lead/runs/astra_terminal_central_hole_source_v2'
SEEDS = tuple(range(7103048, 7103064))
PARENT_SHA = '6a145fadb3224501e683e95424bd9ada0ff571041850c7d482969b742d8f8f41'
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS',
           'BLIS_NUM_THREADS')
assert all(os.environ.get(k) == '1' for k in THREADS)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = D / 'FIT_AND_NETWORK_LOO_v1.json'
assert not out.exists()
rows = []

for seed in SEEDS:
    choices = [T / f'run_tch_science_fresh_s{shard}_v1/native_{seed}_parent.json'
               for shard in (0, 1)]
    paths = [p for p in choices if p.exists()]
    assert len(paths) == 1
    certificate = paths[0]
    cert = json.loads(certificate.read_text())
    assert cert['sha256'] == PARENT_SHA and cert['tch_mode'] == -1
    saved = certificate.with_name(certificate.stem + '_pred.npz')
    truth_path = O / f'analyst/fresh/truth_seed{seed}.npz'
    with np.load(saved, allow_pickle=False) as f:
        y = f[str(seed)][-1].astype(np.float64)
    with np.load(truth_path, allow_pickle=False) as f:
        truth = f['mc_post'][-1].astype(np.float64)
    mse = float(np.mean((truth - y) ** 2))
    assert np.isclose(mse, cert['rows'][0]['mse_final'], rtol=1e-12, atol=0)
    scale = float(np.mean(y))
    assert scale > 0 and np.isfinite(scale)
    X = np.column_stack((y, np.full(y.shape, scale), y*y/scale))
    target = truth - y
    G = (X.T @ X) / (len(y) * mse)
    h = (X.T @ target) / (len(y) * mse)
    rows.append(dict(seed=seed, y=y, truth=truth, X=X, G=G, h=h, mse=mse,
                     certificate=str(certificate), certificate_sha256=sha(certificate),
                     prediction_sha256=sha(saved), truth_path=str(truth_path),
                     truth_sha256=sha(truth_path)))

G = sum(row['G'] for row in rows)
h = sum(row['h'] for row in rows)
condition = float(np.linalg.cond(G))
assert np.isfinite(condition) and condition < 1e10
beta = np.linalg.solve(G, h)


def score(row, coefficients):
    y = np.maximum(row['y'] + row['X'] @ coefficients, 0).astype(np.float32)
    mse = float(np.mean((row['truth'] - y.astype(np.float64)) ** 2))
    return dict(mse=mse, ratio=mse/row['mse'], log_ratio=float(np.log(mse/row['mse'])))


details = []
for row in rows:
    loo = np.linalg.solve(G - row['G'], h - row['h'])
    details.append(dict(seed=row['seed'], parent_mse=row['mse'],
                        fitted=score(row, beta), network_loo=score(row, loo),
                        network_loo_coefficients=loo.tolist(),
                        provenance={k:row[k] for k in ('certificate','certificate_sha256',
                                    'prediction_sha256','truth_path','truth_sha256')}))


def aggregate(kind):
    logs = np.array([row[kind]['log_ratio'] for row in details])
    return dict(mean_log_ratio=float(logs.mean()),
                geometric_mse_change_pct=float(100*np.expm1(logs.mean())),
                lower_count=int(np.sum(logs < 0)), count=len(logs))


report = dict(status='TRAINING_DIAGNOSTIC_ONLY_NO_GAIN_ACCEPTANCE',
              source_sha256=sha(__file__), parent_sha256=PARENT_SHA,
              authorized_training_seeds=list(SEEDS),
              terms=['y','mean(y)','y*y/mean(y)'], coefficients=beta.tolist(),
              fit='Fixed unregularized weighted least squares, equal baseline-relative network weighting; one all16 fit.',
              leave_one_network_out='Exactly16 refits excluding one whole network; same fixed basis and weighting.',
              condition_number=condition, fitted=aggregate('fitted'),
              network_loo=aggregate('network_loo'), rows=details,
              previous_negative_evidence='TEAM_LOG.md:742 T84 final residual predictability was negligible on an older torch card; this tests only current matched-native-C output calibration.',
              candidate_imports=0, model_calls=0, fresh_oos_truth_reads=0,
              no_scientific_acceptance=True, no_sweep_or_tuning=True)
with out.open('x') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
print(json.dumps({k:report[k] for k in ('status','coefficients','condition_number','fitted','network_loo')}))
print('report_sha256', sha(out))
