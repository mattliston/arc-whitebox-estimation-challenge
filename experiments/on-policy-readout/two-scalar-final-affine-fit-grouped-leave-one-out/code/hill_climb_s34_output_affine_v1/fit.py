"""One fixed TRAIN16 affine output fit; root executes, no estimator import.

Usage: native-python -B fit.py PROTOCOL_SHA OUTPUT_DIRECTORY
Only final-row saved predictions/targets are fitted. No parameter search.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np

D = Path(__file__).resolve().parent
ROSTER = [38, 40, 41, 43, 44, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 60]
BASE_SHA = '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def verified(item):
    assert pin(item['path']) == item
    return Path(item['path'])


def write_once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def solve(y, truth):
    y = y.reshape(-1).astype(np.float64)
    target = truth.reshape(-1).astype(np.float64) - y
    X = np.column_stack((y, np.ones_like(y)))
    A = X.T @ X
    rhs = X.T @ target
    assert np.isfinite(A).all() and np.isfinite(rhs).all() and (np.diag(A) > 0).all()
    regularized = A + 0.1 * np.diag(np.diag(A))
    theta = np.linalg.solve(regularized, rhs)
    assert theta.shape == (2,) and np.isfinite(theta).all()
    a, b = np.float32(1.0 + theta[0]), np.float32(theta[1])
    assert np.isfinite(a) and np.isfinite(b)
    return theta, a, b


def replay(y, a, b):
    # Three distinct FP32 operations, identical order required in any native port.
    y32 = np.asarray(y, dtype=np.float32)
    product = np.multiply(a, y32, dtype=np.float32)
    shifted = np.add(product, b, dtype=np.float32)
    return np.maximum(shifted, np.float32(0.0))


def main():
    assert len(sys.argv) == 3
    protocol_path = D / 'protocol.json'
    protocol_pin = pin(protocol_path)
    assert protocol_pin['sha256'] == sys.argv[1]
    cfg = json.loads(protocol_path.read_text())
    assert cfg['root_admitted'] and cfg['training_ranks'] == ROSTER
    assert cfg['ridge_diagonal_fraction'] == 0.1 and cfg['endpoint_count'] == 1
    assert cfg['fit_source'] == pin(__file__)
    assert cfg['gates'] == dict(mean_loo_raw_gain_pct_min=0.1, wins_min=10, max_regression_pct=1.0, all_finite=True)
    assert str(Path(sys.executable).absolute()) == cfg['runtime']
    assert np.__version__ == '2.4.6'
    assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
    assert cfg['baseline']['sha256'] == BASE_SHA
    verified(cfg['baseline'])
    parent_cfg = json.loads(verified(cfg['input_protocol']).read_text())
    assert parent_cfg['baseline'] == cfg['baseline'] and parent_cfg['training_ranks'] == ROSTER
    expected_data = [{k: row[k] for k in ('rank', 'native_prediction', 'truth')} for row in parent_cfg['data']]
    assert cfg['data'] == expected_data and [row['rank'] for row in cfg['data']] == ROSTER
    assert cfg['native_anchor'] == parent_cfg['native_anchor']
    anchors = {k: verified(v) for k, v in cfg['native_anchor'].items()}
    manifest = json.loads(anchors['manifest'].read_text())
    assert manifest['sha256'] == BASE_SHA and manifest['ranks'] == ROSTER
    guard = json.loads(anchors['guard'].read_text())
    assert guard['completed'] and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] and not guard['remaining_rows'] and guard['max_swap_growth_bytes'] == 0
    baseline_rows = [json.loads(line) for line in anchors['rows'].read_text().splitlines()]
    assert [row['rank'] for row in baseline_rows] == ROSTER
    assert not set(ROSTER) & {35, 100, 36, 45, 52, 58, 64, 70}
    out = Path(sys.argv[2]).resolve()
    assert out.parent == D and not out.exists()
    # Pin every input before opening any array payload; only these 16 ranks exist here.
    for row in cfg['data']:
        verified(row['native_prediction'])
        verified(row['truth'])
    out.mkdir()
    started = time.monotonic()
    predictions, targets = [], []
    for item in cfg['data']:
        pred = np.load(item['native_prediction']['path'], allow_pickle=False)
        assert pred.shape == (16, 1024) and pred.dtype == np.float64 and np.isfinite(pred).all()
        assert np.array_equal(pred, pred.astype(np.float32).astype(np.float64))
        with np.load(item['truth']['path'], allow_pickle=False) as archive:
            truth = archive['mc_post'][15].astype(np.float64)
        assert truth.shape == (1024,) and np.isfinite(truth).all()
        predictions.append(pred[15].copy())
        targets.append(truth)
    y, truth = np.stack(predictions), np.stack(targets)
    before = np.mean((y - truth) ** 2, axis=1)
    assert np.isfinite(before).all() and (before > 0).all()
    for actual, row in zip(before, baseline_rows):
        assert abs(float(actual) - row['mse']) <= 1e-20
    theta, a, b = solve(y, truth)
    fitted = replay(y, a, b)
    loo = np.empty_like(fitted)
    fold_rows = []
    for i, rank in enumerate(ROSTER):
        keep = [j for j in range(len(ROSTER)) if j != i]
        fold_theta, fold_a, fold_b = solve(y[keep], truth[keep])
        loo[i] = replay(y[i], fold_a, fold_b)
        mse = float(np.mean((loo[i].astype(np.float64) - truth[i]) ** 2))
        change = 100.0 * (mse / float(before[i]) - 1.0)
        fold_rows.append(dict(rank=rank, excluded_from_fit=True, baseline_mse=float(before[i]),
            loo_mse=mse, raw_change_pct=change, a=float(fold_a), b=float(fold_b)))
    all_finite = bool(np.isfinite(fitted).all() and np.isfinite(loo).all()
                      and all(np.isfinite(row['raw_change_pct']) for row in fold_rows))
    assert all_finite
    changes = [row['raw_change_pct'] for row in fold_rows]
    mean_gain = -float(np.mean(changes))
    wins = sum(change < 0 for change in changes)
    max_regression = max(changes)
    gates = dict(mean_loo_raw_gain=mean_gain >= 0.1, wins=wins >= 10,
                 max_regression=max_regression <= 1.0, all_finite=all_finite)
    weights = dict(schema='s34-output-affine-weights-v1', training_ranks=ROSTER,
        theta_float64=theta.tolist(), a=float(a), b=float(b),
        a_float32_bits=int(a.view(np.uint32)), b_float32_bits=int(b.view(np.uint32)),
        ridge_diagonal_fraction=0.1, target='truth-y', features=['y', '1'],
        replay='float32 multiply(a,y); float32 add(b); maximum with float32 zero',
        apply_only_layer=15, preserve_layers=list(range(15)), endpoint_count=1,
        protocol=protocol_pin, fit_source=pin(__file__), baseline=cfg['baseline'])
    write_once(out / 'weights.json', weights)
    for filename, values in [('fit_predictions.npy', fitted), ('loo_predictions.npy', loo)]:
        with (out / filename).open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
    report = dict(schema='s34-output-affine-fit-report-v1', protocol=protocol_pin,
        fit_source=pin(__file__), input_protocol=cfg['input_protocol'], baseline=cfg['baseline'],
        training_ranks=ROSTER, baseline_pooled_mse=float(before.mean()),
        fit_pooled_mse=float(np.mean((fitted.astype(np.float64) - truth) ** 2)),
        loo_pooled_mse=float(np.mean((loo.astype(np.float64) - truth) ** 2)),
        mean_loo_raw_gain_pct=mean_gain, wins=wins, max_loo_regression_pct=max_regression,
        folds=fold_rows, gates=gates, admitted_to_native2=all(gates.values()),
        weights=pin(out / 'weights.json'), fit_predictions=pin(out / 'fit_predictions.npy'),
        loo_predictions=pin(out / 'loo_predictions.npy'),
        full_fit_endpoint_only=True, no_loo_coefficient_selection=True, valid_or_check_targets_accessed=False,
        network_forward_executed=False, native_or_serving_qualified=False,
        resource_retirement_requires_external_guard=True, elapsed_s=time.monotonic()-started)
    write_once(out / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('mean_loo_raw_gain_pct', 'wins',
        'max_loo_regression_pct', 'gates', 'admitted_to_native2', 'weights', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    main()
