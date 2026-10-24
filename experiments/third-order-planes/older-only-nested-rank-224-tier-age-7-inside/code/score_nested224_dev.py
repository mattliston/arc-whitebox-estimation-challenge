"""Seal both cold predictions before opening the fixed development truths."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
build = read(HERE / 'nested224_build.json')
assert sha(HERE / 'nested224_cost150.py') == build['candidate_sha256']
for name in ('nested224_dev_watch.json', 'nested224_cold100_watch.json'):
    w = read(HERE / name)
    assert w['completed'] and w['whole_group_retired'] and w['max_swap_growth_bytes'] == 0
dev = read(HERE / 'nested224_dev_v1/result.json')['records']
assert dev[0]['output_sha256'] == dev[2]['output_sha256']
cold100 = read(HERE / 'nested224_cold100_v1/result.json')['records'][0]
assert dev[1]['output_sha256'] == cold100['output_sha256']
pairs = []
for rank, current, parent_dir in ((35, dev[0], 'cost_rule150_35'),
                                  (100, cold100, 'cost150_panel_v1/candidate_100')):
    parent_record = read(HERE / parent_dir / 'result.json')
    assert parent_record['candidate_sha256'] == build['parent_sha256']
    parent = parent_record['records'][0]
    assert current['includes_initialization'] and parent['includes_initialization']
    assert current['weights_sha256'] == parent['weights_sha256']
    for record in (current, parent):
        assert sha(record['output']) == record['output_sha256']
    a, b = np.load(current['output']), np.load(parent['output'])
    assert np.array_equal(a[:8], b[:8])
    pairs.append((rank, current, parent, a, b))
with (HERE / 'nested224_dev_predictions_sealed.json').open('x') as f:
    json.dump(dict(candidate_sha256=build['candidate_sha256'], repeat_byte_exact=True,
                   warm_cold100_byte_exact=True, first8_byte_exact=True,
                   pairs=[dict(rank=r, current=c, parent=p) for r,c,p,a,b in pairs]), f, indent=2)
rows = []
for rank, current, parent, a, b in pairs:
    truth_path = HERE.parent / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz'
    truth = np.load(truth_path)['mc_post'][-1].astype(np.float64)
    scores = {}
    for label, record, pred in (('candidate', current, a), ('parent', parent, b)):
        raw = float(np.mean((pred[-1].astype(np.float64) - truth)**2))
        scores[label] = dict(raw=raw, flops=record['flops'], adjusted=raw*record['multiplier'])
    rows.append(dict(rank=rank, truth_sha256=sha(truth_path), **scores,
                     raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
                     adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted']))
record = dict(rows=rows, first8_byte_exact=True, repeat_byte_exact=True,
              both_adjusted_improve=all(r['adjusted_ratio'] < 1 for r in rows),
              pooled_adjusted_ratio=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows))
with (HERE / 'nested224_dev_score.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record), flush=True)
