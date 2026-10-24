"""Stdlib-only chronology replay of a previously frozen prebind key swap.

No estimator import, array operations, target reads, or candidate construction.
Estimates assume setup binds and fuses every key before its wall guard.
"""
import ast
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
H = D.parent
SOURCE = H / 'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
FROZEN = H / 'hill_climb_s32_train6_key_census_v1/capacity_neutral_analysis.json'
CENSUS = D / 'key_census.jsonl'
CAPS = (320, 352, 320)


def pin(p):
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def source_keys():
    old = ()

    def literal(n):
        if isinstance(n, (ast.Tuple, ast.List)):
            return tuple(literal(x) for x in n.elts)
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.Name) and n.id == '_PB_SINGLE':
            return old
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return literal(n.left) + literal(n.right)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'tuple':
            assert len(n.args) == 1 and not n.keywords
            return tuple(literal(n.args[0]))
        raise ValueError(ast.dump(n))

    for n in ast.parse(SOURCE.read_text()).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_PB_SINGLE' for t in n.targets):
            old = literal(n.value)
    assert len(old) == len(set(old)) == 649
    return old


def bind_cost(key, fused=False):
    """Count public FNP requests from _ipt_side6989 and _lvf_ops6964.

    _ipt_view: getitem+reshape; chunks: (L-2)getitem+(L-1)unstack;
    level1: getitem+reshape+transpose+unstack, plus A pair getitem+transpose
    or B pair getitem; basis: 2(L-1)getitems. Each unstack mints seven
    handles, so handles=requests+6L. Fusion views mint one handle/request.
    Only the two reshape requests are billed, by output element count.
    """
    side, level, rows, cols = key
    assert level >= 2 and rows % 2**level == cols % 2**level == 0
    plain = 4*level + 1 + (2 if side == 0 else 1 if side == 1 else 0)
    fuse = (3*level-2, 3*level, 5*level-6)[side] if fused else 0
    return dict(requests=plain+fuse, minted_handles=plain+fuse+6*level,
                reshape_flops=(7**level+7*4**(level-1))*(rows//2**level)*(cols//2**level))


def arithmetic_saving(key, caller):
    side, level, _, _ = key
    name, first, line = caller
    expected = {
        7020: ('_ipt_forward_a_ab', 7021, 0),
        7036: ('_ipt_forward_b_ab', 7037, 1),
        7052: ('_ipt_leaf_unwind_ab', 7053, 2),
        8053: ('_ipt_forward_a_ab', 8057, 0),
        7335: ('_ks_product', 7348, 2),
    }
    assert expected[first] == (name, line, side)
    if first == 8053:
        return 0  # _TR_LAST reuses a tree without executing its forward.
    # _ks_product7335 passes its returned ops to _ks_unwind once, so that
    # indirect C transform DOES receive the same fusion credit as7052.
    return level if side < 2 else 2*(level-1)


def caches_from_keys(keys):
    # _pb_bind6532 touches reversed(_PB_SINGLE) after constructing entries;
    # phase2 fusion preserves order via dict.__setitem__ on existing keys.
    return [OrderedDict((k[1:], True) for k in reversed(keys) if k[0] == side)
            for side in range(3)]


def snapshot(caches):
    return [[list(k)+[v] for k, v in cache.items()] for cache in caches]


def replay(row, caches, verify):
    if verify:
        assert snapshot(caches) == row['initial_cache']
    total = Counter()
    observed_counts = Counter()
    key_rows = {}
    evictions = []
    for index, (key_id, caller_id, hit0, before0, fused0) in enumerate(row['events']):
        key = tuple(row['keys'][key_id])
        caller = row['callers'][caller_id]
        cache = caches[key[0]]
        local = key[1:]
        hit = local in cache
        before = hit and cache[local]
        if hit:
            fused = cache.pop(local)
        else:
            fused = False  # _LVF_EAGER=False during predict.
            while len(cache) >= CAPS[key[0]]:
                removed, removed_fused = cache.popitem(last=False)
                evictions.append(dict(event=index, key=[key[0], *removed], fused=removed_fused))
        cache[local] = fused
        assert len(cache) <= CAPS[key[0]]
        if verify:
            assert (int(hit), int(before), int(fused)) == (hit0, before0, fused0), (row['rank'], index)
        observed_counts[(key_id, caller_id, bool(hit0), bool(before0), bool(fused0))] += 1
        price = bind_cost(key)
        detail = key_rows.setdefault(key, Counter())
        for meter in (total, detail):
            meter['calls'] += 1
            meter['misses'] += int(not hit)
            meter['binding_requests'] += int(not hit)*price['requests']
            meter['binding_minted_handles'] += int(not hit)*price['minted_handles']
            meter['binding_flops'] += int(not hit)*price['reshape_flops']
            meter['fusion_requests_saved'] += int(fused)*arithmetic_saving(key, caller)
            meter['plain_arithmetic_calls'] += int(not fused and arithmetic_saving(key, caller) > 0)
            meter['cache_only_calls'] += int(caller[1] == 8053)
    assert observed_counts == Counter({(r['key_id'], r['caller_id'], r['hit'], r['before_fused'], r['returned_fused']): r['calls'] for r in row['counts']})
    if verify:
        assert snapshot(caches) == row['final_cache']
    return dict(rank=row['rank'], totals=dict(total), evictions=evictions,
                cache_sizes=[len(c) for c in caches], fused_keys=[sum(c.values()) for c in caches],
                by_key=[dict(key=list(k), **v) for k, v in sorted(key_rows.items())])


def main():
    assert pin(SOURCE)['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    assert pin(FROZEN)['sha256'] == '91b1d9b6ccf951b9f81093fe41390525499397eac1dc1af3cf53504013642361'
    assert pin(CENSUS)['sha256'] == '01e285f4e046319c66e35c72f815d2c44e4a6ea6b8fc62d49c7ab91550744d43'
    fixed = json.loads(FROZEN.read_text())
    original = source_keys()
    selected = tuple(map(tuple, fixed['selected636']))
    adds = {tuple(x['key']) for x in fixed['added']}
    drops = {tuple(x['key']) for x in fixed['dropped']}
    bundle_set = set(selected)-adds | drops
    extra13 = tuple(k for k in original if k not in bundle_set)
    assert len(extra13) == 13 and not (set(extra13) & (adds | drops))
    proposed = selected + extra13
    assert len(proposed) == len(set(proposed)) == len(original)
    assert set(proposed)-set(original) == adds and set(original)-set(proposed) == drops
    assert len(adds) == len(drops) == 48
    assert set(map(tuple, fixed['protected36'])) <= set(proposed)
    rows = list(map(json.loads, CENSUS.read_text().splitlines()))
    assert [r['rank'] for r in rows] == [35, 100]
    base_caches = caches_from_keys(original)
    child_caches = caches_from_keys(proposed)
    results = []
    for row in rows:
        base = replay(row, base_caches, True)
        child = replay(row, child_caches, False)
        bt, ct = base['totals'], child['totals']
        removed_bind = bt['binding_requests'] - ct['binding_requests']
        additional_fusion = ct['fusion_requests_saved'] - bt['fusion_requests_saved']
        results.append(dict(rank=row['rank'], baseline=base, prospective=child,
            removed_binding_requests=removed_bind, additional_fusion_request_saving=additional_fusion,
            net_explicit_request_saving=removed_bind+additional_fusion,
            net_billed_flop_saving=bt['binding_flops']-ct['binding_flops'],
            diagnostic_baseline_flops=row['flops']))
    setup_delta = {k: sum(bind_cost(x, True)[k] for x in adds)-sum(bind_cost(x, True)[k] for x in drops)
                   for k in ('requests', 'minted_handles', 'reshape_flops')}
    assert setup_delta == dict(requests=-8, minted_handles=-14, reshape_flops=532842163)
    mean_saving = sum(r['net_explicit_request_saving'] for r in results)/2
    result = dict(schema='frozen48_chronology_replay_v1', measured_candidate=False,
        scope='Stdlib-only replay; no candidate built or model execution.',
        sources=[pin(p) for p in (SOURCE, FROZEN, CENSUS, Path(__file__).resolve())],
        original_setup_counts=dict(Counter(k[0] for k in original)),
        proposed_setup_counts=dict(Counter(k[0] for k in proposed)),
        unchanged_binding_count=682, original_order_validated=True,
        all_original_hit_fusion_events_and_initial_final_orders_validated=True,
        selection_frozen_before_current_D5_diagnostic=True,
        proposed_single_keys=[list(k) for k in proposed], preserved_later13=[list(k) for k in extra13],
        setup_delta=setup_delta, results=results, mean_explicit_request_saving=mean_saving,
        mean_billed_flop_saving=sum(r['net_billed_flop_saving'] for r in results)/2,
        aggregate_bill_saving_pct=100*sum(r['net_billed_flop_saving'] for r in results)/sum(r['flops'] for r in rows),
        planning_request_floor_exceeded=mean_saving > 500,
        assumptions=['Setup completes all649 single keys and their fusion before unchanged4.0s guard.',
                     'Shape chronology remains equal, since the key table only chooses preconstructed views and existing exact fused kernels.',
                     'No claim of byte parity for prospective fused arithmetic until root primitive/model validation.',
                     'No conversion from request count to residual seconds or runtime eligibility.',
                     'Rank100 retains caches after35, matching this actual diagnostic; this is not an independent cold-worker forecast.'],
        correction='Caller7335 is an indirect arithmetic use: _ks_product passes ops to _ks_unwind once. Caller8053 alone is cache-only.',
        status='ELIGIBLE_FOR_ROOT_BUILD_CONSIDERATION' if mean_saving > 500 else 'INSUFFICIENT_REQUEST_REDUCTION_FOR_ROOT_PLANNING_FLOOR')
    out = D / 'replay_frozen48_v1.json'
    with out.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(output=pin(out), rows=[{k:v for k,v in r.items() if k not in ('baseline','prospective')} for r in results],
        setup_delta=setup_delta, mean_request_saving=mean_saving, status=result['status'])))


if __name__ == '__main__':
    main()
