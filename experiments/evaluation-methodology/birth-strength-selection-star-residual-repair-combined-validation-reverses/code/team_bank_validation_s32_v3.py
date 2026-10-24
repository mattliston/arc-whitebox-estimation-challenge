"""Offline S32 VALID40 assessment; versions team_bank_validation_s32_v2 (argv-only preflight correction), never launches jobs.

Usage: LIGHT team_bank_validation_s32_v3.py hill_climb_bds_csr_exact_aggregate_v1
The guarded standard comparison owns the unchanged 50,000-pair bootstrap.
This assessor checks receipts and row-derived scores without loading predictions/truth.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

H = Path(__file__).resolve().parent
P2 = H.parent
ROOT = P2.parents[3]
NATIVE = P2 / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
LIGHT = ROOT / '.venv/bin/python'
S32 = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
CANDIDATE = '23a433b250046b7a6dd9b30c279ccb1e5bc7cc9e4dd63b6d68153c1b60480538'
GATES = dict(adjusted_pct_at_most=-1, wins_min=28, ci95_upper_below=0,
             max_per_network_regression_pct=10)
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'), '1')

def read(path):
    return json.loads(Path(path).read_text())

def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)

def once(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')

def verify(obj):
    if isinstance(obj, dict):
        if {'path', 'sha256', 'bytes'} <= obj.keys():
            assert pin(obj['path']) == {k: obj[k] for k in ('path', 'sha256', 'bytes')}, obj['path']
        else:
            for value in obj.values():
                verify(value)
    elif isinstance(obj, list):
        for value in obj:
            verify(value)

def guard(path, command, seconds):
    g = read(path)
    assert g['command'] == command, (path, g['command'], command)
    assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows'], path
    assert g['exit_code'] == 0 and g.get('error') is None, path
    assert g['max_swap_growth_bytes'] == 0 and 0 < g['max_rss_bytes'] < 7_500_000_000, path
    assert g['rss_trigger_bytes'] == 7_500_000_000 and 0 < g['elapsed_s'] <= seconds, path
    return pin(path)

def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-10), (a, b)

def block(tag, source, ranks, guard_path, seconds):
    d = H / tag
    m = read(d / 'manifest.json')
    assert Path(m['candidate']).resolve() == Path(source['path'])
    assert m['sha256'] == source['sha256'] and pin(m['candidate']) == source
    assert m['ranks'] == ranks and len(ranks) == 10 and len(set(ranks)) == 10
    assert Path(m['runtime']).resolve() == NATIVE.resolve() and m['numpy'] == '2.4.6'
    assert m['thread_env'] == THREADS
    flags = [] if source['sha256'] == S32 else ['-B']
    gp = guard(guard_path, [str(NATIVE), *flags, str(H / 'native_eval.py'), source['path'],
                           ','.join(map(str, ranks)), tag], seconds)
    rows = [json.loads(line) for line in (d / 'rows.jsonl').read_text().splitlines()]
    assert [r['rank'] for r in rows] == ranks
    for r in rows:
        assert all(math.isfinite(r[k]) and r[k] > 0 for k in ('mse', 'adjusted', 'util', 'cpu_s', 'wall_s'))
        assert type(r['flops']) is int and 0 < r['flops'] <= 2**41
        close(r['util'], r['flops'] / 2**41)
        # Relative check is needed for these ~1e-9 values (no loose absolute floor).
        assert math.isclose(r['adjusted'], r['mse'] * max(.1, r['util']), rel_tol=2e-14, abs_tol=0)
    return m, rows, dict(manifest=pin(d / 'manifest.json'), rows=pin(d / 'rows.jsonl'),
                        guard=gp, predictions=[pin(d / f'pred_{rank}.npy') for rank in ranks])

def assess(d):
    p = read(d / 'broad_protocol.json')
    verify(p)
    assert p['assessor'] == pin(__file__) and p['score_gates'] == GATES
    candidate, parent = p['candidate'], p['parent']
    assert candidate['sha256'] == CANDIDATE and parent['sha256'] == S32
    assert p['baseline_submission'] == 332738 and p['reused_validation_not_pristine']
    assert p['limits'] == dict(native_seconds=420, baseline_seconds=360, compare_seconds=120,
                              rss_bytes=7_500_000_000, added_swap_bytes=0, serial=True)
    build = read(p['build']['path'])
    assert build['passed'] and build['candidate'] == candidate and build['parent'] == parent
    verify(build)
    train = read(p['training_admission']['path'])
    assert train['passed'] and train['retained'] and train['candidate'] == candidate and train['parent'] == parent
    assert train['metrics']['n'] == 16 and all(train['decisions'].values())
    verify(train['evidence'])
    native_admission = read(p['broad_native_admission']['path'])
    verify(native_admission)
    assert native_admission['candidate'] == candidate and native_admission['training'] == p['training_admission']
    assert native_admission['score_gates'] == GATES and native_admission['blocks'] == p['blocks']
    base_rows, new_rows, base_manifests, new_manifests, evidence = [], [], [], [], []
    for i, b in enumerate(p['blocks'], 1):
        assert b['block'] == i and b['guard_seconds'] == 420 and b['rss_bytes'] == 7_500_000_000 and b['added_swap_bytes'] == 0
        old_tag = f'hill_climb_star32_broad_v1_bank_valid40_b{i}_v1'
        new_tag = f'hill_climb_bds_csr_exact_aggregate_valid40_b{i}_v1'
        assert b['candidate_tag'] == new_tag
        assert b['baseline_manifest'] == pin(H / old_tag / 'manifest.json')
        assert b['baseline_rows'] == pin(H / old_tag / 'rows.jsonl')
        bm, br, be = block(old_tag, parent, b['ranks'], b['baseline_guard']['path'], 360)
        nm, nr, ne = block(new_tag, candidate, b['ranks'], d / f'broad_b{i}_guard.json', 420)
        base_rows.extend(br); new_rows.extend(nr)
        base_manifests.append(bm); new_manifests.append(nm)
        evidence.append(dict(baseline=be, candidate=ne))
    ranks = sorted(r['rank'] for r in base_rows)
    assert len(ranks) == len(set(ranks)) == 40 and len(evidence) == 4
    assert not set(ranks).intersection(train['metrics']['ranks'])
    comp = d / 'broad_comparison.json'
    bg = ','.join(f'hill_climb_star32_broad_v1_bank_valid40_b{i}_v1' for i in range(1, 5))
    cg = ','.join(b['candidate_tag'] for b in p['blocks'])
    command = read(d / 'broad_compare_guard.json')['command']
    assert len(command) == 6 and Path(command[0]).resolve() == LIGHT.resolve() and command[1] == '-B'
    assert Path(command[2]).resolve() == H / 'compare_native.py' and command[3:5] == [bg, cg]
    assert (H / command[5]).resolve() == comp
    compare_guard = guard(d / 'broad_compare_guard.json', command, 120)
    c = read(comp)
    assert c['n'] == 40 and c['ranks'] == ranks
    assert c['base_sources'] == base_manifests and c['candidate_sources'] == new_manifests
    base = {r['rank']: r for r in base_rows}; new = {r['rank']: r for r in new_rows}
    for key, field in [('mse', 'raw_pct'), ('adjusted', 'adjusted_pct'), ('util', 'mean_util_pct'), ('cpu_s', 'mean_cpu_pct')]:
        close(c[field], 100 * (math.fsum(new[r][key] for r in ranks) / math.fsum(base[r][key] for r in ranks) - 1))
    for key, field in [('mse', 'raw_wins'), ('adjusted', 'adjusted_wins')]:
        assert c[field] == sum(new[r][key] < base[r][key] for r in ranks)
    assert math.isclose(c['base_mean_adjusted'], math.fsum(base[r]['adjusted'] for r in ranks) / 40, rel_tol=2e-12, abs_tol=0)
    assert math.isclose(c['candidate_mean_adjusted'], math.fsum(new[r]['adjusted'] for r in ranks) / 40, rel_tol=2e-12, abs_tol=0)
    assert len(c['per_network']) == 40 and [r['rank'] for r in c['per_network']] == ranks
    for r in c['per_network']:
        for key, field in [('mse', 'raw_pct'), ('adjusted', 'adjusted_pct'), ('util', 'util_pct')]:
            close(r[field], 100 * (new[r['rank']][key] / base[r['rank']][key] - 1))
    for name in ('adjusted_ci95_pct', 'raw_ci95_pct'):
        assert len(c[name]) == 2 and all(map(math.isfinite, c[name])) and c[name][0] <= c[name][1]
    maximum = max(r['adjusted_pct'] for r in c['per_network'])
    gates = dict(adjusted=c['adjusted_pct'] <= -1, wins=c['adjusted_wins'] >= 28,
                 ci=c['adjusted_ci95_pct'][1] < 0, max_regression=maximum <= 10)
    return dict(passed=all(gates.values()), receipt_checks_passed=True, stage='VALID40',
                candidate=candidate, parent=parent, baseline_submission=332738,
                policy=p['policy'], protocol=pin(d / 'broad_protocol.json'),
                training_admission=p['training_admission'], broad_native_admission=p['broad_native_admission'],
                comparison=pin(comp), comparison_guard=compare_guard, assessor=pin(__file__),
                gates=gates, validation_score_passed=all(gates.values()), metrics=c,
                max_per_network_adjusted_regression_pct=maximum,
                validation_guards=[e['candidate']['guard'] for e in evidence], evidence=evidence,
                reused_validation_not_pristine=True, prefix_parity_scope='L0 verified in pinned fixed16 admission; no additional prediction decoding here',
                worker_qualified=False, release_qualified=False, submitted=False,
                next_step='Root may admit serving qualification only if all score gates pass; no automatic release')

if __name__ == '__main__':
    d = Path(sys.argv[1]).resolve()
    assert d == H / 'hill_climb_bds_csr_exact_aggregate_v1'
    output = d / 'broad_assessment.json'
    assert not output.exists()
    try:
        result = assess(d)
    except Exception as exc:
        once(output, dict(passed=False, receipt_checks_passed=False, stage='VALID40',
                          error=dict(type=type(exc).__name__, message=str(exc)), assessor=pin(__file__),
                          worker_qualified=False, release_qualified=False, submitted=False))
        raise
    once(output, result)
    print(json.dumps({k: result[k] for k in ('passed', 'receipt_checks_passed', 'gates', 'max_per_network_adjusted_regression_pct')}))
