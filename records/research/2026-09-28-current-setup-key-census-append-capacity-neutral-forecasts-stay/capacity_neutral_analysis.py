"""Reproduce a deferred shape-only forecast; never import/build/run an estimator.

Usage: python3 capacity_neutral_analysis.py [new-output.json]
All inputs are pinned by the preserved initial metadata analysis. Output is
exclusive, and the admitted ranking, capacity and forecast gate are fixed.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
PRIOR = H2 / 'teammate_s32_capacity_refresh_v1.json'
PRIOR_SHA = 'd4bebf1198f8e6618d71f9f7a03c4dd0af52ad2e425f089461a983cd366a06d0'
CAPS = {0: 320, 1: 352, 2: 320}
RANKS = [36, 45, 52, 58, 64, 70]


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def keys_from_source(path):
    """Evaluate only the literal tuple grammar used by _PB_SINGLE assignments."""
    old = ()

    def literal(node):
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(literal(x) for x in node.elts)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id == '_PB_SINGLE':
            return old
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'tuple' and len(node.args) == 1
                and not node.keywords):
            return tuple(literal(node.args[0]))
        raise ValueError('Unsupported literal grammar: ' + ast.dump(node))

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == '_PB_SINGLE' for t in node.targets):
            old = literal(node.value)
    assert len(old) == len(set(old))
    return old


def reshape_cost(key):
    side, level, rows, cols = key
    unit = 2 ** level
    assert side in CAPS and level >= 1 and rows % unit == cols % unit == 0
    return (7 ** level + 7 * 4 ** (level - 1)) * (rows // unit) * (cols // unit)


def main():
    assert pin(PRIOR)['sha256'] == PRIOR_SHA
    prior = json.loads(PRIOR.read_text())
    for expected in prior['sources']:
        assert pin(Path(expected['path'])) == expected
    census_path = H2 / 'hill_climb_s32_train6_keys_v1_cache_census.json'
    census = json.loads(census_path.read_text())
    assessment = json.loads((HERE / 'assessment.json').read_text())
    assert census['source']['ranks'] == RANKS
    assert assessment['passed'] and assessment['candidate_eligible'] is False
    assert all(r['all16_bytes_exact'] and r['full_bill_exact'] for r in assessment['parity'])
    old = keys_from_source(H2 / 'hill_climb_star_projection32_v1/candidate.py')
    bundle = keys_from_source(H2 / 'hill_climb_exact_bundle_v1/candidate.py')
    old_set, bundle_set = set(old), set(bundle)
    fixed = tuple(k for k in bundle if k not in old_set)
    assert len(old) == 600 and len(bundle) == 636 and len(fixed) == 36
    assert [list(k) for k in fixed] == prior['fixed_keys']

    ranks, calls, misses = defaultdict(set), Counter(), Counter()
    seen = set()
    for row in census['rows']:
        key, rank = tuple(row['key']), row['rank']
        assert rank in RANKS and (rank, key) not in seen
        seen.add((rank, key))
        assert row['calls'] > 0 and 0 <= row['misses'] <= row['calls']
        ranks[key].add(rank)
        calls[key] += row['calls']
        misses[key] += row['misses']
    pool = (set(ranks) | old_set) - set(fixed)
    # Twice the original score avoids floating-point ties without changing order.
    ordered = sorted(pool, key=lambda k: (-(2 * len(ranks[k]) + int(k in old_set))
                                         * reshape_cost(k), k))
    counts = Counter(k[0] for k in fixed)
    selected, skipped = [], []
    for key in ordered:
        if len(selected) == 600:
            break
        if counts[key[0]] >= CAPS[key[0]]:
            skipped.append(key)
            continue
        selected.append(key)
        counts[key[0]] += 1
    selected636 = fixed + tuple(selected)
    assert len(selected636) == len(set(selected636)) == 636
    assert [list(k) for k in selected] == prior['selected_remaining600']
    added = sorted(set(selected636) - bundle_set)
    dropped = sorted(bundle_set - set(selected636))

    def detail(key):
        return dict(key=list(key), incidence=len(ranks[key]), calls=calls[key],
                    misses=misses[key], reshape_cost=reshape_cost(key), ranks=sorted(ranks[key]))

    add_rows, drop_rows = [detail(k) for k in added], [detail(k) for k in dropped]
    assert add_rows == prior['added'] and drop_rows == prior['dropped']
    assert len(added) == len(dropped) == 48
    assert all(misses[k] == 1 for k in added)
    used_drops = [dict(detail(k), first_use_debit_flops=reshape_cost(k))
                  for k in dropped if calls[k]]
    assert len(used_drops) == 10 and all(misses[tuple(r['key'])] == 0 for r in used_drops)
    gross = sum(misses[k] * reshape_cost(k) for k in added)
    debit = sum(r['first_use_debit_flops'] for r in used_drops)
    total = sum(r['flops'] for r in assessment['parity'])
    net = gross - debit
    assert (gross, debit, net, total) == (1694468377, 93068157, 1601400220, 1645845615881)
    out = dict(prior)
    out.pop('passed')
    out.update(
        status='DEFERRED_UNEXECUTED_POSITIVE_FORECAST_BELOW_PLANNING_GATE',
        measured_candidate_result=False, planning_gate_passed=False,
        forecast_not_measured_incremental_gain=True,
        sources=prior['sources'] + [pin(PRIOR), pin(Path(__file__).resolve())],
        selected636=[list(k) for k in selected636], protected36=[list(k) for k in fixed],
        used_drop_charges=used_drops, selection_pool_keys=len(pool),
        skipped_for_caps=[list(k) for k in skipped],
        counts_by_side={str(k): counts[k] for k in sorted(CAPS)},
        gross_first_miss_savings_flops=gross, dropped_used_first_bind_cost_flops=debit,
        net_firstuse_forecast_flops=net, complete_six_flops=total,
        net_forecast_pct=100 * net / total,
        disposition='Preserved for possible future aggregate consideration; no standalone qualification admitted.')
    assert out['net_forecast_pct'] < out['required_net_forecast_pct'] == 0.1
    destination = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else HERE / 'capacity_neutral_analysis.json'
    assert len(sys.argv) <= 2
    with destination.open('x') as handle:
        json.dump(out, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(output=pin(destination), status=out['status'], net_forecast_pct=out['net_forecast_pct'])))


if __name__ == '__main__':
    main()
