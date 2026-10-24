"""Offline assessment for the two fixed birth studies; never executes estimators."""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

H2 = Path(__file__).resolve().parent
STUDY = H2 / sys.argv[1]
assert STUDY.name in ['hill_climb_birth_distinct_strength_v2', 'hill_climb_birth_direction_fp32_v1']
BLOCKS = [[35, 100], [36, 45, 52, 58, 64, 70], [3, 12, 40, 41, 43, 44, 57, 68]]
CONTROLS = ['hill_climb_star_projection32_train2_v1', 'hill_climb_star_projection32_train6_v1',
            'hill_climb_star_projection32_extra8_candidate_v1']
N = int(sys.argv[2]); assert N in [2, 16]

def read(p):
    return json.loads(p.read_text())

def pin(p):
    b = p.read_bytes()
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))

build = read(STUDY/'build.json')
for key in ['parent', 'candidate', 'builder', 'protocol']:
    assert build[key] == pin(Path(build[key]['path'])), key
assert build['parent']['sha256'] == 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
evidence = [pin(STUDY/'build.json'), pin(Path(__file__)), build['protocol']]
parity = []; base = []; cand = []; manifests = [[], []]
for i in range(1 if N == 2 else 3):
    guard_path = STUDY/(['train2', 'train6', 'extra8'][i]+'_guard.json')
    g = read(guard_path)
    assert g['completed'] and g['exit_code'] == 0 and g['error'] is None
    assert g['whole_group_retired'] and g['remaining_rows'] == []
    assert 0 < g['max_rss_bytes'] < 7500000000 and g['max_swap_growth_bytes'] == 0
    limits = [150, 300, 360] if STUDY.name == 'hill_climb_birth_distinct_strength_v2' else [180, 330, 400]
    assert g['elapsed_s'] <= limits[i]
    tag = g['command'][-1]
    assert g['command'][-4:-1] == [str(H2/'native_eval.py'), build['candidate']['path'], ','.join(map(str, BLOCKS[i]))]
    evidence.append(pin(guard_path))
    paths = [H2/CONTROLS[i], H2/tag]
    for j, (path, source, target) in enumerate(zip(paths, [build['parent'], build['candidate']], [base, cand])):
        m = read(path/'manifest.json'); manifests[j].append(m)
        assert m['candidate'] == source['path'] and m['sha256'] == source['sha256'] and m['ranks'] == BLOCKS[i]
        rows = [json.loads(x) for x in (path/'rows.jsonl').read_text().splitlines()]
        assert [r['rank'] for r in rows] == BLOCKS[i]
        for r in rows:
            assert all(math.isfinite(r[k]) for k in ['mse', 'adjusted', 'util', 'cpu_s', 'wall_s'])
            assert math.isclose(r['util'], r['flops']/2**41, rel_tol=1e-14)
            assert math.isclose(r['adjusted'], r['mse']*max(.1, r['util']), rel_tol=1e-14)
        target.extend(rows); evidence.extend([pin(path/'manifest.json'), pin(path/'rows.jsonl')])
    for rank in BLOCKS[i]:
        files = [p/f'pred_{rank}.npy' for p in paths]; evidence.extend(map(pin, files))
        b, c = [np.load(p, allow_pickle=False) for p in files]
        assert b.shape == c.shape == (16, 1024) and b.dtype == c.dtype == np.dtype('float64')
        assert np.isfinite(b).all() and np.isfinite(c).all() and b[0].tobytes() == c[0].tobytes()
        parity.append(dict(rank=rank, exact=True))
comp_path = STUDY/f'train{N}_comparison.json'; comp = read(comp_path); evidence.append(pin(comp_path))
assert comp['base_sources'] == manifests[0] and comp['candidate_sources'] == manifests[1]
assert comp['n'] == N and comp['ranks'] == sorted(r['rank'] for r in base)
for metric, field in [('adjusted_pct', 'adjusted'), ('raw_pct', 'mse'), ('mean_util_pct', 'util')]:
    assert math.isclose(comp[metric], 100*(sum(r[field] for r in cand)/sum(r[field] for r in base)-1), abs_tol=1e-10)
wins = sum(c['adjusted'] < b['adjusted'] for b, c in zip(base, cand)); assert comp['adjusted_wins'] == wins
decisions = dict(mean=comp['adjusted_pct'] <= -.1, wins=wins >= (2 if N == 2 else 10))
extra_pct = None
if N == 16:
    admission = read(STUDY/'train2_assessment.json'); assert admission['passed'] and admission['advance']
    extra_pct = 100*(sum(r['adjusted'] for r in cand[2:])/sum(r['adjusted'] for r in base[2:])-1)
    decisions.update(ci=comp['adjusted_ci95_pct'][1] < 0, additional14=extra_pct < 0)
result = dict(passed=True, advance=N == 2 and all(decisions.values()), retained=N == 16 and all(decisions.values()),
              candidate=build['candidate'], parent=build['parent'], decisions=decisions, metrics=comp,
              additional14_pct=extra_pct, evidence=evidence, layer0=parity, release_qualification=False)
with (STUDY/f'train{N}_assessment.json').open('x') as f:
    json.dump(result, f, indent=2); f.write('\n')
print(json.dumps({k: result[k] for k in ['passed', 'advance', 'retained', 'decisions']}))
