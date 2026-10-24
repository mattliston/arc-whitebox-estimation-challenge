"""Score fixed RGT native development only after sealing own-cold outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT_DIR = HERE.parent / 'lazy_rotation_v1'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal = read(HERE / 'sources_sealed.json')
for item in seal['sources'] + seal['external_pins']:
    assert sha(item['path']) == item['sha256'], item['path']
build = read(HERE / 'build.json')
assert read(HERE / 'null_verdict.json')['passed']
assert read(HERE / 'unit.json')['passed']
assert sha(HERE / 'candidate.py') == build['candidate_sha256']
for name in ('dev_watch.json', 'cold100_watch.json'):
    w = read(HERE / name)
    assert w['completed'] and w['whole_group_retired'] and w['max_swap_growth_bytes'] == 0
dev = read(HERE / 'dev/result.json')
cold = read(HERE / 'cold100/result.json')
assert dev['candidate_sha256'] == cold['candidate_sha256'] == build['candidate_sha256']
assert [r['rank'] for r in dev['records']] == [35, 100, 35]
assert [r['rank'] for r in cold['records']] == [100]
for rec in dev['records'] + cold['records']:
    assert rec['max_rss_bytes'] < 8_000_000_000
    assert rec['weight5_cache']['completed'] and rec['weight5_cache']['numeric_cache_retired']
for name in ('dev_watch.json', 'cold100_watch.json'):
    assert read(HERE / name)['max_rss_bytes'] < 8_000_000_000
assert dev['records'][0]['output_sha256'] == dev['records'][2]['output_sha256']
assert dev['records'][1]['output_sha256'] == cold['records'][0]['output_sha256']
pairs = []
for rank, current, parent_dir in ((35, dev['records'][0], 'dev'),
                                   (100, cold['records'][0], 'cold100')):
    parent_meta = read(PARENT_DIR / parent_dir / 'result.json')
    assert parent_meta['candidate_sha256'] == build['parent_sha256']
    parent = parent_meta['records'][0]
    assert current['includes_initialization'] and parent['includes_initialization']
    assert current['weights_sha256'] == parent['weights_sha256']
    for record in (current, parent):
        assert sha(record['output']) == record['output_sha256']
    a, b = np.load(current['output']), np.load(parent['output'])
    assert a[:2].tobytes() == b[:2].tobytes()
    pairs.append((rank, current, parent, a, b))
with (HERE / 'dev_predictions_sealed.json').open('x') as stream:
    json.dump(dict(candidate_sha256=build['candidate_sha256'], repeat_byte_exact=True,
                   warm_cold100_byte_exact=True,
                   pairs=[dict(rank=r, current=c, parent=p) for r, c, p, a, b in pairs]), stream, indent=2)
rows = []
for rank, current, parent, a, b in pairs:
    truth_path = HERE.parent.parent / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz'
    expected_truth = {35: 'ec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61', 100: '637a63d46ac5480be50cca3f0021a1574eef80fa580ce98f1a8cc69b0b86450c'}
    assert sha(truth_path) == expected_truth[rank]
    truth = np.load(truth_path)['mc_post'][-1].astype(np.float64)
    scores = {}
    for label, record, pred in (('candidate', current, a), ('parent', parent, b)):
        raw = float(np.mean((pred[-1].astype(np.float64) - truth)**2))
        scores[label] = dict(raw=raw, flops=record['flops'], adjusted=raw * record['multiplier'])
    rows.append(dict(rank=rank, truth_sha256=sha(truth_path), **scores,
                     layerwise_delta_mse=np.mean((a.astype(np.float64)-b.astype(np.float64))**2, axis=1).tolist(),
                     raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
                     adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted']))
rec = dict(rows=rows, repeat_byte_exact=True, warm_cold100_byte_exact=True,
           both_adjusted_improve=all(r['adjusted_ratio'] < 1 for r in rows),
           both_cold_flops_lower=all(r['candidate']['flops'] < r['parent']['flops'] for r in rows),
           pooled_adjusted_ratio=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows))
rec['first_two_layers_byte_exact'] = True
rec['both_raw_improve'] = all(r['raw_ratio'] < 1 for r in rows)
rec['development_gate_passed'] = all(r['adjusted_ratio'] <= .95 for r in rows)
rec['both_adjusted_at_most_095'] = rec['development_gate_passed']
rec['fresh_generalization_evidence'] = False
with (HERE / 'dev_score.json').open('x') as stream:
    json.dump(rec, stream, indent=2)
print(json.dumps(rec), flush=True)
