"""Independent small exhaustive search, depth certificate, and old/new comparison."""
import ast
from fractions import Fraction
from functools import lru_cache
import resource
import textwrap
from common import *

seal_check(); done('reprice'); done('verify_paths')
data = read(HERE / 'reprice_result.json')
assert data['maximum_stages'] == 7 and data['minimum_leaf_dimension'] == 8
replay = read(HERE / 'paths_verified.json')
assert replay['passed'] and replay['result_sha256'] == sha(HERE / 'reprice_result.json')
old = read(D / 'full_lower_cost_frontier_v1/reprice_result.json')
assert old['maximum_stages'] == 5
inputs = read(HERE / 'reprice_inputs.json')
actual = {r['rank']: read(r['path']) for r in inputs['censuses']}

# A size bound independent of the producer's recurrence and selected schedules.
shape_checks = []
for rank, row in actual.items():
    for call in row['outermost_mm_calls']:
        shape = (call['left'][0], call['left'][1], call['right'][1])
        start = min(shape)
        assert start <= 1024
        chain = [start]
        for _ in range(8):
            chain.append((chain[-1] + 1) // 2)
        assert chain[8] < 8
        shape_checks.append(dict(rank=rank, shape=shape, upper_minimum_after_eight=chain[8]))
for family in data['families']:
    assert all(min(p['shape']) >= 2 for p in family['programs'])

# Extend the exact rational identity to depths omitted by the old six fixtures.
for depth in range(8):
    m, k, n = 128*3, 128*5, 128*7
    a, b, c = m//2**depth, k//2**depth, n//2**depth
    cost = 2*a*b*c-a*c
    for _ in range(depth):
        cost = 7*cost + 7*(a*b+b*c+a*c)
        a, b, c = a*2, b*2, c*2
    q = Fraction(7, 4)**depth
    expected = 2*m*k*n*Fraction(7, 8)**depth-m*n*q+Fraction(7, 3)*(m*k+k*n+m*n)*(q-1)
    assert cost == expected

# Load only producer functions, never its top-level search/reporting code.
source = (HERE / 'reprice.py').read_text(); tree = ast.parse(source)
parts = []
for name in ['gemm', 'fringe_parts', 'pareto']:
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    parts.append(ast.get_source_segment(source, node))
calculate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate')
frontier = next(n for n in calculate.body if isinstance(n, ast.FunctionDef) and n.name == 'frontier')
frontier_source = '@lru_cache(None)\n' + textwrap.dedent(ast.get_source_segment(source, frontier))

fixtures = [(9, 256, 64), (15, 256, 64), (31, 255, 33), (64, 15, 256),
            (64, 256, 15), (16, 64, 64), (32, 96, 64), (65, 33, 129)]
limits = [None, 65_536, 1_048_576, 1_168_900_096]
small_checks = []
padding_witnesses = []
for family in data['families']:
    programs = family['programs']
    ns = dict(lru_cache=lru_cache, programs=programs, MIN_LEAF=8)
    exec('\n\n'.join(parts + [frontier_source]), ns)

    @lru_cache(None)
    def exhaustive(m, k, n, stages):
        # Keep EVERY schedule: no producer Pareto rule or dynamic pruning.
        options = [(2*m*k*n-m*n, 4*(m*k+k*n+m*n))]
        if not stages:
            return tuple(options)
        for p in programs:
            for peel in [False, True]:
                dims = tuple(x//d if peel else (x+d-1)//d for x,d in zip((m,k,n),p['shape']))
                if min(dims) < 8:
                    continue
                a,b,c = dims
                cm,ck,cn = [x*d for x,d in zip(dims,p['shape'])]
                if peel and (cm,ck,cn) == (m,k,n):
                    continue
                extra_cost = (2*m*k*n-m*n)-(2*cm*ck*cn-cm*cn) if peel else 0
                fringe = 0
                if peel:
                    for u,v,w in [(cm,k-ck,cn),(cm,k,n-cn),(m-cm,k,n)]:
                        if u > 0 and v > 0 and w > 0:
                            fringe = max(fringe, 4*(u*v+v*w+u*w))
                ca,cb,cc = p['arithmetic']
                linear = ca*a*b+cb*b*c+cc*a*c
                for cost,memory in exhaustive(a,b,c,stages-1):
                    options.append((p['rank']*cost+linear+extra_cost,
                                    max(p['rank']*memory,fringe)))
        return tuple(options)

    for shape in fixtures:
        brute = exhaustive(*shape, 2)
        front = ns['frontier'](*shape, 2)
        for limit in limits:
            available = [cost for cost,memory in brute if limit is None or memory <= limit]
            actual_front = [cost for cost,memory,path in front if limit is None or memory <= limit]
            assert (min(available) if available else None) == (min(actual_front) if actual_front else None)
            small_checks.append(dict(family=family['name'], shape=shape, stages=2,
                leaf_limit=limit, schedules_enumerated=len(brute),
                minimum=min(available) if available else None))
        if min(shape) < 16 and len(brute) > 1:
            padding_witnesses.append(dict(family=family['name'], shape=shape,
                classical_cost=brute[0][0], minimum_cost=min(c for c,m in brute),
                schedules=len(brute)))
    ns['frontier'].cache_clear(); exhaustive.cache_clear()
assert padding_witnesses

comparison = []
changed_calls = []
for before, after in zip(old['families'], data['families']):
    assert before['name'] == after['name'] and before['programs'] == after['programs']
    for b, a in zip(before['records'], after['records']):
        assert b['rank'] == a['rank'] and b['complete_cold_flops'] == a['complete_cold_flops']
        by_index = {r['index']: r for r in b['details']}
        for entry in a['details']:
            prev = by_index[entry['index']]
            assert prev['shape'] == entry['shape'] and prev['actual_call_flops'] == entry['actual_call_flops']
            for field in ['optimistic_saved', 'leaf_storage_limited_saved']:
                assert entry[field] >= prev[field]
            if any(entry[k] != prev[k] for k in ['optimistic_saved', 'leaf_storage_limited_saved']):
                changed_calls.append(dict(family=after['name'], rank=a['rank'], index=entry['index'],
                    shape=entry['shape'], caller=entry['caller'],
                    old_optimistic_saved=prev['optimistic_saved'], new_optimistic_saved=entry['optimistic_saved'],
                    old_limited_saved=prev['leaf_storage_limited_saved'], new_limited_saved=entry['leaf_storage_limited_saved'],
                    new_optimistic_schedule=entry['optimistic_best'], new_limited_schedule=entry['leaf_storage_limited_best']))
        comparison.append(dict(family=after['name'], rank=a['rank'],
            old_fraction=b['optimistic_fraction'], new_fraction=a['optimistic_fraction'],
            old_leaf_limited_fraction=b['leaf_storage_limited_fraction'], new_leaf_limited_fraction=a['leaf_storage_limited_fraction'],
            additional_optimistic_saved=a['optimistic_saved']-b['optimistic_saved'],
            additional_leaf_limited_saved=a['leaf_storage_limited_saved']-b['leaf_storage_limited_saved']))
write(HERE / 'audit_result.json', dict(passed=True, all_old_call_optima_nonworsening=True,
    maximum_depth_seven_complete_for_specified_family=True, current_shape_bound_checks=len(shape_checks),
    maximum_starting_minimum=1024, maximum_possible_minimum_after_eight=4,
    winograd_depth_identities=8, exhaustive_small_checks=small_checks,
    admitted_small_parent_padding_witnesses=padding_witnesses,
    comparison=comparison, changed_calls=changed_calls,
    producer=pin(HERE / 'reprice.py'), result=pin(HERE / 'reprice_result.json'),
    path_replay=pin(HERE / 'paths_verified.json'), old_result=pin(D / 'full_lower_cost_frontier_v1/reprice_result.json'),
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True, small_checks=len(small_checks), changed_calls=len(changed_calls), comparison=comparison), flush=True)

