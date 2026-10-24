"""Fixed, network-separated tree fits; does not open CHECK32 or VALID40."""
from pathlib import Path
import hashlib
import json
import resource
import sys
import time
import numpy as np
import lightgbm as lgb
from tree_arrays import export, predict

D = Path(__file__).resolve().parent
H = D.parent
P2 = H.parent
G = H / 'native_final_graph_head_v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg = json.loads((D / 'protocol.json').read_text())
seal = json.loads((D / 'source_seal.json').read_text())
for name, value in seal['files'].items():
    assert sha(D / name) == value, name
assert lgb.__version__ == '4.6.0'


def once(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def dump_arrays(path, data):
    with path.open('xb') as f:
        np.savez(f, **data)


def qualify():
    rng = np.random.default_rng(2026092711)
    x = rng.normal(size=(6000, 7)).astype(np.float32)
    # Piecewise signal, mixed signs, all-zero rows, and unbalanced leaves.
    x[:8] = 0
    y = 1.4*(x[:, 0] > .2) - .7*(x[:, 1] < -.3) + .2*x[:, 2]
    params = dict(cfg['parameters'], min_data_in_leaf=25, lambda_l2=1)
    b = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=19)
    arrays, nodes = export(b.dump_model())
    probes = np.concatenate((x, rng.normal(size=(1500, 7)).astype(np.float32)))
    # Explicit boundary probes check <=, including adjacent float64 values.
    probes = probes.astype(np.float64)
    boundary = []
    for t, k, kind in nodes:
        if kind == 'branch':
            address = t*31+k
            for value in [arrays['threshold'][address],
                          np.nextafter(arrays['threshold'][address], -np.inf),
                          np.nextafter(arrays['threshold'][address], np.inf)]:
                row = np.zeros(7)
                row[arrays['feature'][address]] = value
                boundary.append(row)
    probes = np.concatenate((probes, boundary))
    expected = b.predict(probes, num_threads=1)
    actual = predict(probes, arrays)
    error = float(np.max(np.abs(actual-expected)))
    assert error <= 1e-12, error
    once(D / 'qualification.json', dict(passed=True, lightgbm=lgb.__version__,
        probes=len(probes), trees=b.num_trees(), nodes=len(nodes),
        max_absolute_export_error=error, synthetic_only=True,
        protocol_sha256=sha(D/'protocol.json'), source_seal_sha256=sha(D/'source_seal.json')))
    print(json.dumps(dict(event='qualified', probes=len(probes), max_error=error)), flush=True)


def run(fold):
    q = json.loads((D / 'qualification.json').read_text())
    guard = json.loads((D / 'qualification_guard.json').read_text())
    assert q['passed'] and guard['completed'] and guard['whole_group_retired']
    assert guard['max_swap_growth_bytes'] == 0
    assert q['source_seal_sha256'] == sha(D / 'source_seal.json')
    capture = json.loads((G / 'capture_complete.json').read_text())
    assert sha(G/'capture_complete.json') == cfg['capture_complete_sha256']
    assert capture['all_groups_retired'] and capture['swap_growth_bytes'] == 0
    records = {r['rank']: r for r in capture['rows']}
    test = cfg['folds'][fold]
    train = sorted(set(cfg['ranks']) - set(test))
    assert len(train) == 96 and len(test) == 32
    blocked = set(cfg['excluded_check32'] + cfg['excluded_valid40'])
    assert not blocked.intersection(train + test)
    out = D / f'fold_{fold}'
    out.mkdir()
    t0 = time.monotonic()

    def load(ranks, labels):
        if labels:
            assert ranks == train
        xs, ss, bs, ts = [], [], [], []
        provenance = []
        for rank in ranks:
            assert rank in cfg['ranks'] and rank not in blocked
            row = records[rank]
            packet = G / row['path']
            assert sha(packet) == row['sha256']
            with np.load(packet) as z:
                xs.append(np.concatenate((z['control'], z['graph']), axis=1))
                ss.append(z['sigma'].astype(np.float64))
                bs.append(z['prediction'][15].astype(np.float64))
            if labels:
                truthfile = P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
                with np.load(truthfile) as z:
                    ts.append(z['mc_post'][15].astype(np.float64))
                provenance.append(dict(rank=rank, sha256=sha(truthfile)))
        x, s, b = np.concatenate(xs), np.concatenate(ss), np.concatenate(bs)
        assert x.shape == (len(ranks)*1024, 151) and x.dtype == np.float32
        assert np.isfinite(x).all() and np.isfinite(s).all() and (s > 0).all()
        return x, s, b, np.concatenate(ts) if labels else None, provenance

    x, sigma, base, truth, targets = load(train, True)
    target = (truth-base)/(sigma*cfg['output_scale'])
    weights = sigma*sigma
    weight_mean = float(weights.mean())
    weights = weights/weight_mean
    results = {}
    # Both models are frozen before any test feature is loaded. Test labels
    # are never opened in this process; the scorer runs after all folds.
    for mode, columns in [('control', 94), ('graph', 151)]:
        dataset = lgb.Dataset(np.ascontiguousarray(x[:, :columns]), label=target,
                              weight=weights, free_raw_data=True)
        model = lgb.train(cfg['parameters'], dataset,
                          num_boost_round=cfg['rounds'], callbacks=[])
        payload = model.dump_model()
        arrays, visited = export(payload, cfg['parameters']['max_depth'])
        once(out / f'{mode}_model.json', payload)
        with (out / f'{mode}_model.txt').open('x') as f:
            f.write(model.model_to_string())
        dump_arrays(out / f'{mode}_arrays.npz', arrays)
        results[mode] = dict(model_sha256=sha(out/f'{mode}_model.json'),
            model_text_sha256=sha(out/f'{mode}_model.txt'),
            arrays_sha256=sha(out/f'{mode}_arrays.npz'), num_trees=model.num_trees(),
            nodes=len(visited), feature_columns=columns,
            gain_importance=model.feature_importance('gain').tolist())
        print(json.dumps(dict(event='model_frozen', fold=fold, mode=mode,
                              trees=model.num_trees(), elapsed_s=time.monotonic()-t0)), flush=True)
        del model, dataset, arrays, payload
    once(out/'models_frozen.json', dict(models=results, test_labels_accessed=False,
        train_ranks=train, test_ranks=test, target_files=targets, weight_mean=weight_mean,
        source_seal_sha256=sha(D/'source_seal.json')))
    testx, tests, testbase, _, _ = load(test, False)
    for mode, columns in [('control', 94), ('graph', 151)]:
        with np.load(out/f'{mode}_arrays.npz') as z:
            arrays = {key: z[key] for key in z.files}
        raw = predict(testx[:, :columns], arrays)
        # The independent official predictor reads the terminal model string.
        model = lgb.Booster(model_str=(out/f'{mode}_model.txt').read_text())
        expected = model.predict(testx[:, :columns], num_threads=1)
        error = float(np.max(np.abs(raw-expected)))
        assert error <= 1e-12, error
        final = (testbase + tests*cfg['output_scale']*raw).astype(np.float32).astype(np.float64)
        assert np.isfinite(final).all()
        with (out/f'{mode}_predictions.npy').open('xb') as f:
            np.save(f, final.reshape(len(test), 1024))
        results[mode].update(export_max_absolute_error=error,
            predictions_sha256=sha(out/f'{mode}_predictions.npy'))
    once(out/'result.json', dict(fold=fold, train_ranks=train, test_ranks=test,
        results=results, test_labels_accessed=False, original_check32_accessed=False,
        valid40_accessed=False, source_seal_sha256=sha(D/'source_seal.json'),
        lightgbm=lgb.__version__, numpy=np.__version__,
        elapsed_s=time.monotonic()-t0,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(event='fold_complete', fold=fold,
                          elapsed_s=time.monotonic()-t0)), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'qualify':
        qualify()
    else:
        run(int(sys.argv[1]))
