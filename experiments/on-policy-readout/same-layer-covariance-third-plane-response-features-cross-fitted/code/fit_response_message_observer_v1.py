"""One sealed, network-cross-fitted linear source test; TRAIN only."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
protocol_path = HERE/'response_message_observer_protocol_v1.json'
protocol = json.loads(protocol_path.read_text())
protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
assert protocol_sha == 'bfa519c390b72a2cd723b11c2ae01c45952d1513691394c158af8cc92040d786'
assert hashlib.sha256((HERE/protocol['parent']).read_bytes()).hexdigest() == protocol['parent_sha256']
tags = ['response_message_capture_train2_v1', 'response_message_capture_train6_v1',
        'response_message_capture_train12_v1']
records = []
for tag, ranks in zip(tags, protocol['capture_groups']):
    guard = json.loads((HERE/f'{tag}_guard.json').read_text())
    assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes'] == 0
    manifest = json.loads((HERE/tag/'manifest.json').read_text())
    captures = json.loads((HERE/tag/'capture_manifest.json').read_text())
    rows = [json.loads(line) for line in (HERE/tag/'rows.jsonl').read_text().splitlines()]
    assert manifest['sha256'] == protocol['parent_sha256'] and manifest['ranks'] == ranks
    assert captures['protocol_sha256'] == protocol_sha and not captures['truth_accessed']
    assert [row['rank'] for row in rows] == ranks
    assert [row['rank'] for row in captures['records']] == ranks
    for cap, row in zip(captures['records'], rows):
        path = HERE/tag/cap['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cap['sha256']
        if tag != tags[-1]:
            assert row['all16_bytes_exact'] and row['bill_exact']
        assert not row['truth_accessed']
        records.append(dict(rank=row['rank'], tag=tag, capture=cap, row=row))
assert [rec['rank'] for rec in records] == protocol['ranks']
assert sum(rec['capture']['bytes'] for rec in records) < 100_000_000
partition = json.loads((P2/'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
assert set(protocol['ranks']).issubset(partition['train'])
assert not set(protocol['ranks']).intersection(partition['valid'])
del partition

out = HERE/'response_message_fit20_v1'
out.mkdir()
started = time.monotonic()
control, history, sigmas, parent, truths, truth_hashes = [], [], [], [], [], []
for rec in records:
    with np.load(HERE/rec['tag']/rec['capture']['path']) as arrays:
        control.append(arrays['control'])
        history.append(arrays['history'])
        sigmas.append(arrays['sigma'])
    parent.append(np.load(HERE/rec['tag']/f"pred_{rec['rank']}.npy")[15])
    truth_path = P2/'fable_refit_v1/truth688'/f"truthpost_rank{rec['rank']:04d}.npz"
    with np.load(truth_path) as arrays:
        truths.append(arrays['mc_post'][15].astype(np.float64))
    truth_hashes.append(dict(rank=rec['rank'], path=str(truth_path),
                             sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest()))

control = np.stack(control)
history = np.stack(history)
features = np.concatenate((control, history), axis=2)
sigma, parent, truth = map(np.stack, (sigmas, parent, truths))
assert control.shape == (20, 1024, 94) and features.shape == (20, 1024, 346)
assert all(np.isfinite(x).all() for x in [features, sigma, parent, truth])
residual = truth - parent
fold_ids = np.arange(20) % protocol['folds']

def fit(x, s, y):
    x = x.reshape(-1, x.shape[-1])
    s, y = s.reshape(-1), y.reshape(-1)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    design = np.column_stack((np.ones(len(x)), (x-center)/scale))
    s_rms = float(np.sqrt(np.mean(s*s)))
    design *= (s/s_rms)[:, None]
    target = y/s_rms
    gram = design.T@design/len(design)
    rhs = design.T@target/len(design)
    penalty = np.full(len(rhs), protocol['ridge'])
    penalty[0] = 0
    gram[np.diag_indices_from(gram)] += penalty
    coefficients = np.linalg.solve(gram, rhs)
    equation_error = float(np.max(np.abs(gram@coefficients-rhs)))
    assert equation_error < 1e-10
    return dict(center=center, scale=scale, coefficients=coefficients), equation_error

def predict(x, s, base, model):
    d = ((x-model['center'])/model['scale'])@model['coefficients'][1:] + model['coefficients'][0]
    return np.maximum(0., base + s*d).astype(np.float32).astype(np.float64)

predictions = {name: np.zeros_like(parent) for name in protocol['models']}
fold_records = []
for fold in range(protocol['folds']):
    train = fold_ids != fold
    test = ~train
    rec = dict(fold=fold, train_ranks=np.asarray(protocol['ranks'])[train].tolist(),
               test_ranks=np.asarray(protocol['ranks'])[test].tolist())
    for name, x in zip(protocol['models'], [control, features]):
        model, error = fit(x[train], sigma[train], residual[train])
        predictions[name][test] = predict(x[test], sigma[test], parent[test], model)
        with (out/f'{name}_fold{fold}.npz').open('xb') as fp:
            np.savez(fp, **model)
        rec[name] = dict(normal_equation_error=error,
                         mse=float(np.mean((predictions[name][test]-truth[test])**2)))
    rec['parent_mse'] = float(np.mean((parent[test]-truth[test])**2))
    rec['augmented_ratio'] = rec['augmented_ridge346']['mse']/rec['parent_mse']
    fold_records.append(rec)
    print(json.dumps(rec), flush=True)

parent_mse = np.mean((parent-truth)**2, axis=1)
control_mse = np.mean((predictions['control_ridge94']-truth)**2, axis=1)
augmented_mse = np.mean((predictions['augmented_ridge346']-truth)**2, axis=1)
ratios = augmented_mse/parent_mse
measurements = dict(pooled_raw_ratio=float(augmented_mse.mean()/parent_mse.mean()),
                    control_raw_ratio=float(control_mse.mean()/parent_mse.mean()),
                    ratio_to_control=float(augmented_mse.mean()/control_mse.mean()),
                    winning_folds=int(sum(r['augmented_ratio'] < 1 for r in fold_records)),
                    winning_networks=int(np.count_nonzero(ratios < 1)),
                    worst_network_raw_ratio=float(ratios.max()))
g = protocol['advance_gates']
gate_checks = dict(pooled_raw=measurements['pooled_raw_ratio'] <= g['pooled_raw_ratio_max'],
                   incremental_information=measurements['ratio_to_control'] <= g['ratio_to_control_max'],
                   folds=measurements['winning_folds'] >= g['winning_folds_min'],
                   networks=measurements['winning_networks'] >= g['winning_networks_min'],
                   worst_network=measurements['worst_network_raw_ratio'] <= g['worst_network_raw_ratio_max'])
passed = all(gate_checks.values())
rng = np.random.default_rng(20260927)
samples = rng.integers(0, 20, size=(50000, 20))
boot = augmented_mse[samples].mean(axis=1)/parent_mse[samples].mean(axis=1)-1
report = dict(protocol_sha256=protocol_sha, parent_sha256=protocol['parent_sha256'],
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              ranks=protocol['ranks'], scope='TRAIN network cross-fitting of a new frozen-output head; not independent VALID or metered candidate evidence',
              parent_mse=float(parent_mse.mean()), control_mse=float(control_mse.mean()),
              augmented_mse=float(augmented_mse.mean()), measurements=measurements,
              bootstrap_ci95_pct=(100*np.quantile(boot, [.025, .975])).tolist(),
              gates=gate_checks, passed=passed, folds=fold_records,
              rows=[dict(rank=rank, fold=int(fold_ids[i]), parent_mse=float(parent_mse[i]),
                         control_mse=float(control_mse[i]), augmented_mse=float(augmented_mse[i]),
                         augmented_ratio=float(ratios[i])) for i, rank in enumerate(protocol['ranks'])],
              truth_records=truth_hashes, labels='TRAIN only; no VALID reads',
              decision=protocol['advance'] if passed else protocol['failure'])
for name, values in predictions.items():
    with (out/f'{name}_oof_predictions.npy').open('xb') as fp:
        np.save(fp, values)
if passed:
    model, error = fit(features, sigma, residual)
    with (out/'augmented_all_train_model.npz').open('xb') as fp:
        np.savez(fp, **model)
    report['all_train_normal_equation_error'] = error
report['elapsed_s'] = time.monotonic()-started
with (out/'report.json').open('x') as fp:
    json.dump(report, fp, indent=2, allow_nan=False)
    fp.write('\n')
print(json.dumps(dict(measurements=measurements, gates=gate_checks, passed=passed,
                      elapsed_s=report['elapsed_s'])), flush=True)
