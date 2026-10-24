"""Fixed cross-fit residual span with train-only weighted orthogonalization."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent.parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
RIDGE = 1e-4


def rms(x, w):
    return np.sqrt(np.sum(x*x*w[:, None], axis=0)/np.sum(w))


def ridge(x, y, w):
    gram = x.T @ (x*w[:, None])
    gram = (gram + gram.T)*.5
    gram.flat[::len(gram)+1] += RIDGE*np.sum(w)
    chol = np.linalg.cholesky(gram)
    return np.linalg.solve(chol.T, np.linalg.solve(chol, x.T @ (w*y)))


def fit(x, z, y, w):
    sx, sz = np.maximum(rms(x, w), 1e-12), np.maximum(rms(z, w), 1e-12)
    xn, zn = x/sx, z/sz
    beta = ridge(xn, y, w)
    root_w = np.sqrt(w)
    u, singular, vt = np.linalg.svd(root_w[:, None]*xn, full_matrices=False)
    retain = singular > 1e-7*singular[0]
    assert retain.any()
    projection = (vt[retain].T/singular[retain]) @ (u[:, retain].T @ (root_w[:, None]*zn))
    residual = zn - xn @ projection
    sr = rms(residual, w)
    keep = sr > 1e-6
    en = residual[:, keep]/sr[keep]
    gamma = ridge(en, y-xn@beta, w) if keep.any() else np.empty(0)
    orthogonality = float(np.max(np.abs(xn.T @ (en*w[:, None])))/np.sum(w)) if keep.any() else 0.
    assert orthogonality < 1e-5, orthogonality
    return dict(sx=sx, sz=sz, beta=beta, projection=projection, sr=sr,
                 keep=keep, gamma=gamma, rank=int(retain.sum()), singular=singular,
                 orthogonality=orthogonality)


def predict(model, x, z):
    xn = x/model['sx']
    control = xn @ model['beta']
    residual = z/model['sz'] - xn @ model['projection']
    extension = (residual[:, model['keep']]/model['sr'][model['keep']]) @ model['gamma']
    return control, control+extension


def certificate():
    # Duplicate features cannot change the control prediction, including off-train.
    t = np.linspace(-1, 1, 40)
    x = np.stack([np.ones_like(t), t, t*t], axis=1)
    z = x @ np.array([[2., -1.], [.25, 3.], [-.7, 1.2]])
    y, w = np.sin(2*t), 1.2+.2*t
    duplicate = fit(x, z, y, w)
    assert not duplicate['keep'].any()
    a, b = predict(duplicate, x, z)
    assert np.array_equal(a, b)
    # A direct augmented normal-equation solve independently checks the two-stage fit.
    z = np.stack([np.sin(3*t), np.cos(2*t), t**3], axis=1)
    model = fit(x, z, y, w)
    xn = x/model['sx']
    en = (z/model['sz'] - xn@model['projection'])[:, model['keep']]/model['sr'][model['keep']]
    design = np.concatenate([xn, en], axis=1)
    gram = design.T @ (design*w[:, None]) + RIDGE*np.sum(w)*np.eye(design.shape[1])
    joint = np.linalg.solve(gram, design.T @ (w*y))
    _, staged = predict(model, x, z)
    error = float(np.max(np.abs(staged - design@joint)))
    assert error < 1e-10, error
    return dict(duplicate_columns_leave_control_exact=True, augmented_solve_max_abs=error,
                 training_orthogonality=model['orthogonality'])


def main():
    started = time.monotonic()
    cert = certificate()
    manifest = read(HERE / 'train_capture/manifest.json')
    assert manifest['completed'] and not manifest['truth_accessed'] and len(manifest['rows']) == 20
    assert manifest['capture_source_sha256'] == sha(HERE / 'capture.py')
    assert manifest['protocol_sha256'] == sha(HERE / 'PROTOCOL.md')
    out = HERE / 'information'
    out.mkdir(exist_ok=False)
    data = []
    for row in manifest['rows']:
        r = row['record']
        w = read(row['watch'])
        assert w['completed'] and w['whole_group_retired'] and w['max_swap_growth_bytes'] == 0
        assert sha(row['completion']) == row['completion_sha256']
        assert sha(r['capture']) == r['capture_sha256']
        with np.load(r['capture']) as file:
            d = {k: file[k].astype(np.float64) for k in file.files}
        sigma = d['sigma']
        control = sigma[:, None]*np.concatenate([np.ones((1024, 1)), d['hidden']], axis=1)
        extension = sigma[:, None]*d['twohop']
        assert control.shape == (1024, 65) and extension.shape == (1024, 10)
        data.append(dict(row=row, prediction=d['prediction'][-1], control=control, extension=extension))
    target_manifest = P2 / 'astra_phase2_reexamination_v1/joint_prefix_final_fit_v1/manifest.json'
    assert sha(target_manifest) == 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
    targets = {r['rank']: r for r in read(target_manifest)['target_sources']}
    with (out / 'inputs_sealed.json').open('x') as stream:
        json.dump(dict(capture_manifest_sha256=sha(HERE / 'train_capture/manifest.json'),
                       source_sha256=sha(__file__), protocol_sha256=sha(HERE / 'PROTOCOL.md'),
                       target_manifest_sha256=sha(target_manifest), ranks=manifest['train'],
                       targets_opened=False, certificate=cert), stream, indent=2)
    target_hashes = {}
    for d in data:
        rank = d['row']['rank']
        path = P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz'
        target_hashes[str(rank)] = sha(path)
        assert str(path) == targets[rank]['target_file'] and sha(path) == targets[rank]['target_sha256']
        assert sha(targets[rank]['weight_file']) == d['row']['record']['weights_sha256']
        truth = np.load(path)['mc_post'][-1].astype(np.float64)
        d.update(truth=truth, residual=truth-d['prediction'])
    predictions = {name: [None]*20 for name in ('control', 'extended')}
    fits = []
    for fold in range(5):
        train = [i for i in range(20) if i % 5 != fold]
        check = [i for i in range(20) if i % 5 == fold]
        x = np.concatenate([data[i]['control'] for i in train])
        z = np.concatenate([data[i]['extension'] for i in train])
        y = np.concatenate([data[i]['residual'] for i in train])
        w = np.concatenate([np.full(1024, data[i]['row']['record']['parent_multiplier']) for i in train])
        model = fit(x, z, y, w)
        for i in check:
            corrections = predict(model, data[i]['control'], data[i]['extension'])
            for arm, correction in zip(('control', 'extended'), corrections):
                predictions[arm][i] = np.maximum(data[i]['prediction']+correction, 0).astype(np.float32)
        fit_path = out / f'fold{fold}_fit.npz'
        np.savez(fit_path, **model)
        rec = dict(fold=fold, train_ranks=[data[i]['row']['rank'] for i in train],
                    check_ranks=[data[i]['row']['rank'] for i in check],
                    control_rank=model['rank'], retained_extension_columns=int(model['keep'].sum()),
                    orthogonality=model['orthogonality'], fit_sha256=sha(fit_path))
        fits.append(rec)
        print(json.dumps(rec), flush=True)
    arrays = {name: np.stack(values) for name, values in predictions.items()}
    assert all(a.shape == (20, 1024) and np.isfinite(a).all() for a in arrays.values())
    np.savez(out / 'oof_predictions.npz', **arrays)
    with (out / 'oof_sealed.json').open('x') as stream:
        json.dump(dict(predictions_sha256=sha(out / 'oof_predictions.npz'), target_sha256=target_hashes,
                       fits=fits, selection_used=False), stream, indent=2)
    rows = []
    for i, d in enumerate(data):
        record = d['row']['record']
        row = dict(rank=d['row']['rank'], fold=i % 5)
        for arm, added in [('unchanged', 0), ('control', 1_000_000), ('extended', record['feature_flops']+2_000_000)]:
            pred = d['prediction'] if arm == 'unchanged' else arrays[arm][i].astype(np.float64)
            raw = float(np.mean((pred-d['truth'])**2))
            flops = record['parent_flops']+added
            row[arm] = dict(raw=raw, flops=flops, adjusted=raw*max(.1, flops/2**41))
        rows.append(row)
    arms = ('unchanged', 'control', 'extended')
    pooled = {a: float(np.mean([r[a]['adjusted'] for r in rows])) for a in arms}
    folds = [{a: float(np.mean([r[a]['adjusted'] for r in rows if r['fold'] == f])) for a in arms}
             for f in range(5)]
    wins = sum(f['extended'] < min(f['control'], f['unchanged']) for f in folds)
    passed = pooled['extended'] <= .95*pooled['unchanged'] and pooled['extended'] <= .97*pooled['control'] and wins >= 4
    result = dict(schema='twohop_terminal_information_v1', exploratory_train_network_cv=True,
                    rows=rows, pooled_adjusted=pooled, fold_adjusted=folds, fold_wins=wins,
                    extended_to_unchanged=pooled['extended']/pooled['unchanged'],
                    extended_to_control=pooled['extended']/pooled['control'], gate_passed=bool(passed),
                    source_sha256=sha(__file__), capture_manifest_sha256=sha(HERE / 'train_capture/manifest.json'),
                    coefficient_projection_train_only=True, certificate=cert, deployment_bill_qualified=False,
                    elapsed_s=time.monotonic()-started, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (out / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'fold_adjusted')}), flush=True)


if __name__ == '__main__':
    main()
