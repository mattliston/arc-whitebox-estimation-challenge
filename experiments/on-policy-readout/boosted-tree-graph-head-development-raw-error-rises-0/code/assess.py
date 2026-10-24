"""Score frozen network-separated predictions; no CHECK32 or VALID40 access."""
from pathlib import Path
import hashlib
import json
import numpy as np

D = Path(__file__).resolve().parent
G = D.parent / 'native_final_graph_head_v1'
P2 = D.parent.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg = json.loads((D/'protocol.json').read_text())
seal = json.loads((D/'source_seal.json').read_text())
for name, value in seal['files'].items():
    assert sha(D/name) == value, name


def once(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


rows = {r['rank']: r for r in json.loads((G/'capture_complete.json').read_text())['rows']}
predictions = {mode: {} for mode in ['control', 'graph']}
models = []
for fold in range(4):
    guard = json.loads((D/f'fold_{fold}_guard.json').read_text())
    assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes'] == 0
    path = D/f'fold_{fold}'
    result = json.loads((path/'result.json').read_text())
    assert result['test_ranks'] == cfg['folds'][fold]
    assert set(result['train_ranks']).isdisjoint(result['test_ranks'])
    assert len(result['train_ranks']) == 96 and len(result['test_ranks']) == 32
    assert not result['test_labels_accessed'] and not result['original_check32_accessed'] and not result['valid40_accessed']
    assert result['source_seal_sha256'] == sha(D/'source_seal.json')
    models.append(result)
    for mode in predictions:
        for suffix, key in [('model.json', 'model_sha256'), ('model.txt', 'model_text_sha256'),
                            ('arrays.npz', 'arrays_sha256'), ('predictions.npy', 'predictions_sha256')]:
            assert sha(path/f'{mode}_{suffix}') == result['results'][mode][key]
        pred = np.load(path/f'{mode}_predictions.npy')
        assert pred.shape == (32, 1024) and np.isfinite(pred).all()
        predictions[mode].update(zip(result['test_ranks'], pred))
once(D/'preassessment_seal.json', dict(models=models,
    source_seal_sha256=sha(D/'source_seal.json'), assessment_labels_opened=False))
ranks = cfg['ranks']
assert all(set(p) == set(ranks) for p in predictions.values())
base, truth, truth_hashes = [], [], []
for rank in ranks:
    assert rank not in cfg['excluded_check32'] and rank not in cfg['excluded_valid40']
    packet = G / rows[rank]['path']
    assert sha(packet) == rows[rank]['sha256']
    with np.load(packet) as z:
        base.append(z['prediction'][15].astype(np.float64))
    target = P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
    with np.load(target) as z:
        truth.append(z['mc_post'][15].astype(np.float64))
    truth_hashes.append(dict(rank=rank, sha256=sha(target)))
base, truth = np.stack(base), np.stack(truth)
mse = {'bank': ((base-truth)**2).mean(axis=1)}
for mode in predictions:
    mse[mode] = ((np.stack([predictions[mode][r] for r in ranks])-truth)**2).mean(axis=1)
rng = np.random.default_rng(cfg['bootstrap_seed'])
indices = rng.integers(0, len(ranks), size=(50000, len(ranks)))


def compare(a, b):
    ratio = float(a.mean()/b.mean())
    interval = np.quantile(a[indices].mean(1)/b[indices].mean(1), [.025, .975]).tolist()
    return dict(raw_ratio=ratio, change_pct=100*(ratio-1), ci95_ratio=interval,
                ci95_pct=[100*(v-1) for v in interval], wins=int((a < b).sum()))


comparisons = {mode: compare(mse[mode], mse['bank']) for mode in predictions}
contrast = compare(mse['graph'], mse['control'])
gates = {}
fold_scores = {}
for mode in predictions:
    ratios = []
    for fold in cfg['folds']:
        mask = [ranks.index(r) for r in fold]
        ratios.append(float(mse[mode][mask].mean()/mse['bank'][mask].mean()))
    fold_scores[mode] = ratios
    c = comparisons[mode]
    gates[mode] = dict(raw=c['raw_ratio'] <= .97, confidence=c['ci95_ratio'][1] < .98,
                       wins=c['wins'] >= 96, all_folds=all(v < 1 for v in ratios))
gates['graph']['vs_local'] = contrast['raw_ratio'] <= .99
eligible = {m: all(g.values()) for m, g in gates.items()}
survivor = 'graph' if eligible['graph'] else ('control' if eligible['control'] else None)
result = dict(scope='Four-fold out-of-fold development screen on reused FIT128; not independent validation',
    protocol_sha256=sha(D/'protocol.json'), source_seal_sha256=sha(D/'source_seal.json'),
    preassessment_sha256=sha(D/'preassessment_seal.json'), ranks=ranks,
    comparisons_vs_bank=comparisons, graph_vs_control=contrast, fold_raw_ratios=fold_scores,
    gates=gates, eligible=eligible, survivor=survivor,
    per_network=[dict(rank=r, **{k: float(v[i]) for k, v in mse.items()}) for i, r in enumerate(ranks)],
    target_files=truth_hashes, original_check32_accessed=False, valid40_accessed=False,
    native_metering_qualified=False, submission_qualified=False,
    decision='Advance fixed survivor to full FIT128 training and native qualification' if survivor else
        'Close fixed tree screen; no sign, amplitude, round, feature or hyperparameter rescue on these results')
once(D/'assessment.json', result)
print(json.dumps({k: result[k] for k in ['comparisons_vs_bank', 'graph_vs_control', 'fold_raw_ratios',
                                        'gates', 'survivor', 'decision']}, indent=2), flush=True)
