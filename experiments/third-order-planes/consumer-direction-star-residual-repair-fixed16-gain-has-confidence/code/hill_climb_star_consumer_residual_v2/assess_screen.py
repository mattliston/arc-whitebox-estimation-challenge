"""Offline TRAIN2 receipt/L0 check; no candidate execution or truth decoding."""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
H2 = HERE.parent


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def read(path):
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def guard(path, candidate, tag, ranks, seconds):
    g = read(path)
    check(g['completed'] and g['exit_code'] == 0 and g['error'] is None, 'guard completion')
    check(g['whole_group_retired'] and g['remaining_rows'] == [], 'guard retirement')
    check(0 < g['max_rss_bytes'] < 7500000000 and g['max_swap_growth_bytes'] == 0, 'guard memory/swap')
    check(g['elapsed_s'] <= seconds, 'guard elapsed')
    check(g['command'][-4:] == [str(H2/'native_eval.py'), candidate['path'],
                                ','.join(map(str, ranks)), tag], 'guard command')
    return pin(path)


def load_run(tag, candidate, ranks, evidence):
    path = H2/tag
    manifest = read(path/'manifest.json')
    check(manifest['candidate'] == candidate['path'] and manifest['sha256'] == candidate['sha256'], 'manifest source')
    check(manifest['ranks'] == ranks, 'manifest roster/order')
    rows = [json.loads(line) for line in (path/'rows.jsonl').read_text().splitlines()]
    check([row['rank'] for row in rows] == ranks, 'rows roster/order')
    for row in rows:
        check(all(math.isfinite(row[key]) for key in ['mse','util','adjusted','cpu_s','wall_s']), 'nonfinite row')
        check(row['flops'] > 0 and row['mse'] >= 0 and row['cpu_s'] > 0, 'invalid row')
        check(math.isclose(row['util'], row['flops']/2**41, rel_tol=1e-14), 'FLOP utilization')
        check(math.isclose(row['adjusted'], row['mse']*max(.1,row['util']), rel_tol=1e-14), 'adjusted row')
    evidence += [pin(path/'manifest.json'), pin(path/'rows.jsonl')]
    return manifest, rows, path


def main():
    destination = HERE/'train2_assessment.json'
    check(not destination.exists(), 'exclusive assessment output')
    evidence, failures, parity = [], [], []
    result = dict(passed=False, advance=False, evidence=evidence, failures=failures,
                  layer0=parity, no_estimator_execution=True, no_truth_reads=True,
                  assessor=pin(Path(__file__).resolve()))
    try:
        protocol = read(HERE/'protocol.json')
        build = read(HERE/'build.json')
        for key in ['parent','fragment','candidate']:
            check(protocol[key] == pin(Path(protocol[key]['path'])), 'frozen '+key)
            check(build[key] == protocol[key], 'builder '+key)
        check(build['passed'] and build['exact_inverse'] and build['ast_parent_prefix_unchanged'], 'source build')
        check(build['protocol'] == pin(HERE/'protocol.json'), 'build protocol pin')
        evidence += [pin(HERE/'protocol.json'), pin(HERE/'build.json')]
        result.update(candidate=protocol['candidate'], parent=protocol['parent'], protocol=pin(HERE/'protocol.json'))
        ranks = protocol['train2_ranks']; check(ranks == [35,100], 'fixed TRAIN2')
        cg = read(HERE/'train2_guard.json'); tag = cg['command'][-1]
        check(Path(tag).name == tag, 'native tag')
        evidence.append(guard(HERE/'train2_guard.json', protocol['candidate'], tag, ranks, protocol['limits']['train2_seconds']))
        base_tag = protocol['controls'][0]
        evidence.append(guard(H2/'hill_climb_star_projection32_v1/train2_guard.json', protocol['parent'], base_tag, ranks, 150))
        bm, base, bp = load_run(base_tag, protocol['parent'], ranks, evidence)
        cm, cand, cp = load_run(tag, protocol['candidate'], ranks, evidence)
        comparison_path = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE/'train2_comparison.json'
        comparison = read(comparison_path); evidence.append(pin(comparison_path))
        check(comparison['n'] == 2 and comparison['ranks'] == ranks, 'comparison roster')
        check(comparison['base_sources'] == [bm] and comparison['candidate_sources'] == [cm], 'comparison manifests')
        for field, key in [('raw_pct','mse'),('adjusted_pct','adjusted'),('mean_util_pct','util'),('mean_cpu_pct','cpu_s')]:
            value = 100*(sum(r[key] for r in cand)/sum(r[key] for r in base)-1)
            check(math.isfinite(comparison[field]) and math.isclose(comparison[field],value,abs_tol=1e-10), 'comparison '+field)
        wins = sum(c['adjusted'] < b['adjusted'] for b,c in zip(base,cand))
        check(comparison['adjusted_wins'] == wins, 'adjusted wins')
        for rank in ranks:
            left, right = bp/f'pred_{rank}.npy', cp/f'pred_{rank}.npy'
            evidence += [pin(left),pin(right)]  # Seal evidence before decoding.
            b, c = np.load(left,allow_pickle=False), np.load(right,allow_pickle=False)
            check(b.shape == c.shape == (16,1024) and b.dtype == c.dtype == np.dtype('float64'), 'prediction layout')
            check(np.isfinite(b).all() and np.isfinite(c).all(), 'finite all16 predictions')
            exact = b[0].tobytes() == c[0].tobytes(); parity.append(dict(rank=rank,exact=exact))
            check(exact, 'L0 byte parity')
        gate = protocol['train2_advance']
        result.update(passed=True, comparison=pin(comparison_path), metrics=comparison,
                      advance=bool(comparison['adjusted_pct'] <= gate['adjusted_pct_at_most'] and wins >= gate['wins_min']),
                      gate=gate, candidate_native_tag=tag)
    except Exception as exc:
        failures.append(dict(type=type(exc).__name__,message=str(exc)))
    with destination.open('x') as output:
        json.dump(result,output,indent=2); output.write('\n')
    print(json.dumps({key:result[key] for key in ['passed','advance','failures']}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
