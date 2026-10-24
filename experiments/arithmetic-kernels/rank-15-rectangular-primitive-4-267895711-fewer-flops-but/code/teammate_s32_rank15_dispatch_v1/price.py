"""Integer/Fraction pricing of ONE stored rank-15 algorithm; no arrays/model."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import hashlib
import json

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
ASTRA = H2.parent / 'astra_phase2_reexamination_v1'
CP = H2 / 'hill_climb_s32_bill_census_v1/census.json'
CERT = ASTRA / 'catalog_executable_cse_certificate_v1.json'
PILOT = H2 / 'teammate_mixedradix_pilot_v1/native_read6016.json'


def read(p):
    return json.loads(p.read_text())


def pin(p):
    data = p.read_bytes()
    return dict(path=str(p.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def binary(d, m, k, n):
    if not d:
        return (0, 0, 0, m*n*(2*k-1))
    growth = Fraction(7, 4)**d
    load = growth + Fraction(d+3, 4)
    unwind = 2*growth - Fraction(3, 4) + Fraction(7*(d-1), 16)
    return tuple(map(int, (load*m*k, load*k*n, unwind*m*n,
                          7**d*(m//2**d)*(n//2**d)*(2*(k//2**d)-1))))


def alternative(p, d, m, k, n):
    u = 2**d
    a, b, c = ((x+s*u-1)//(s*u)*u for x, s in zip((m, k, n), p['shape']))
    if min(a, b, c)//u < 4:
        return None
    M, K, N = [x*s for x, s in zip((a, b, c), p['shape'])]
    padding = ((m*k if (M, K) != (m, k) else 0)
               + (k*n if (K, N) != (k, n) else 0)
               + (m*n if (M, N) != (m, n) else 0))
    la, lb, lc = p['packed_linear_operations']
    parts = binary(d, a, b, c)
    tree = 4*15*7**d*((a//u)*(b//u)+(b//u)*(c//u)+(a//u)*(c//u))
    maps = 4*15*(a*b+b*c+a*c)
    return dict(name=p['name'], depth=d, child=[a, b, c], padded=[M, K, N],
                leaf=[a//u, b//u, c//u], leaves=15*7**d,
                flops=15*sum(parts)+la*a*b+lb*b*c+lc*a*c+padding,
                binary_child_flops=15*sum(parts), linear_map_flops=la*a*b+lb*b*c+lc*a*c,
                padding_copy_flops=padding, tree_bytes=tree, packed_maps_bytes=maps)


def group(r):
    return (r['layer'], r['line'], tuple(map(tuple, r['outer'])))


census = read(CP)
programs = [r for r in read(CERT)['records'] if r['rank'] == 15]
assert len(programs) == 6 and all(r['passed'] for r in programs)
products = []
actual_warm = Counter()
actual_cold = Counter()
kernel_names = {'_ipt_leaf_unwind_ab', '_ks_product', '_ks_unwind',
                '_ipt_forward_a_ab', '_ipt_forward_b_ab',
                '_rs_load_a', '_rs_load_b', '_rs_out_quadrants'}
for r in census['rows']:
    if r['inner'][0] in kernel_names:
        actual_warm[group(r)] += r['flops']
    if r['inner'][0] in {'_ipt_view', '_ipt_bits', '_lvf_rs_level1'}:
        actual_cold[group(r)] += r['flops']
    if r['op'] != 'matmul' or len(r['shapes']) != 2 or any(t != 'float32' for t in r['dtypes']):
        continue
    aa, bb = r['shapes']
    pre = aa[:-2]
    if not pre or pre != bb[:-2] or any(x != 7 for x in pre):
        continue
    d = len(pre)
    shape = aa[-2]*2**d, aa[-1]*2**d, bb[-1]*2**d
    parts = binary(d, *shape)
    assert parts[-1]*r['ops'] == r['flops']
    products.append(dict(shape=shape, depth=d, parts=parts, calls=r['ops'],
                         layer=r['layer'], outer=r['outer'], inner=r['inner'], key=group(r)))


def price_family(pp):
    positive = []
    model_group = Counter()
    proposed_group = Counter()
    selected_groups = set()
    keys = set()
    for row in products:
        m, k, n = row['shape']
        choices = [q for p in pp for d in range(7) if (q := alternative(p, d, m, k, n))]
        best = min(choices, key=lambda q: (q['flops'], q['name'], q['depth']))
        old = sum(row['parts'])
        saving = max(0, old-best['flops'])*row['calls']
        model_group[row['key']] += old*row['calls']
        proposed_group[row['key']] += min(old, best['flops'])*row['calls']
        if saving:
            positive.append(dict(**{k: v for k, v in row.items() if k != 'key'},
                                 savings=saving, alternative=best))
            selected_groups.add(row['key'])
            d = best['depth']
            a, b, c = best['child']
            for side, x, y in ((0, a, b), (1, b, c), (2, a, c)):
                keys.add((side, d, x, y))
    gross = sum(r['savings'] for r in positive)
    adjustment = sum(actual_warm[k]-model_group[k] for k in selected_groups)
    # Existing batched _ipt_side_p's TWO cold reshapes: whole tree, then
    # the level1 four-valued subgrid. This excludes additional fused binds.
    cold = sum(15*(7**d+7*4**(d-1))*(x//2**d)*(y//2**d)
               for side, d, x, y in keys if d)
    cold_tree_only = sum(15*7**d*(x//2**d)*(y//2**d) for side, d, x, y in keys if d)
    by_family = Counter()
    by_depth = Counter()
    for r in positive:
        by_family[r['inner'][0]] += r['savings']
        by_depth[r['alternative']['depth']] += r['calls']
    old_cold = sum(actual_cold[k] for k in selected_groups)
    return dict(names=[p['name'] for p in pp], gross=gross, gross_pct=100*gross/census['bill'],
                positive_calls=sum(r['calls'] for r in positive), positive=positive,
                selected_groups=len(selected_groups), grouped_actual_warm_adjustment=adjustment,
                grouped_warm_saving=gross+adjustment,
                grouped_warm_pct=100*(gross+adjustment)/census['bill'],
                old_selected_group_cold=old_cold,
                new_unique_side_keys=len(keys), new_cold_tree_reshape_lower=cold_tree_only,
                new_cold_existing_batched_two_reshape=cold,
                grouped_cold_no_prebind_saving=gross+adjustment+old_cold-cold,
                new_leaf_products=sum(r['alternative']['leaves']*r['calls'] for r in positive),
                old_leaf_products=sum(7**r['depth']*r['calls'] for r in positive),
                max_tree_plus_packed_maps_bytes=max(r['alternative']['tree_bytes']+r['alternative']['packed_maps_bytes'] for r in positive),
                by_family=dict(by_family), by_depth=dict(by_depth))


single = [price_family([p]) for p in programs]
family = price_family(programs)
pilot = read(PILOT)
native_parent, native_smirnov = pilot['records'][:2]
matmul = native_smirnov['context']['operations']['matmul']
leaf_seconds = matmul['flopscope_backend_time_s'] / (40*7**5)
result = dict(metadata_only=True, estimator_built=False, numerical_jobs=0,
              actual_total_bill=census['bill'], matched_binary_products=sum(r['calls'] for r in products),
              matched_leaf_flops=sum(r['parts'][-1]*r['calls'] for r in products),
              single_orientation=[{k: v for k, v in x.items() if k != 'positive'} for x in single],
              one_algorithm_shape_dispatch=family,
              historical_runtime=dict(smirnov_leaf=[7, 32, 9], smirnov_leaf_count=40*7**5,
                                      smirnov_leaf_backend_s=matmul['flopscope_backend_time_s'],
                                      smirnov_total_wall_s=native_smirnov['wall_s'],
                                      parent_total_wall_s=native_parent['wall_s'],
                                      seconds_per_leaf_descriptive=leaf_seconds,
                                      new_leaf_seconds_at_same_rate=family['new_leaf_products']*leaf_seconds,
                                      caveat='This is a reference-rate scenario, NOT a runtime bound; leaf shapes/layout/dispatch differ.'),
              primary_url='https://fmm.univ-lille.fr/2x3x3.html',
              pins=[pin(p) for p in (CP, CERT, PILOT, Path(__file__),
                                    ASTRA/'public_rectangular_catalog_data_v1/2x3x3_raw.mpl.bz2',
                                    H2/'hill_climb_star_projection32_v1/candidate.py')],
              limitations=[
                  'The optimistic per-leaf baseline assigns a full A/B transform and unwind to each product; grouped actual warm adjustment subtracts observed reuse/KS advantages.',
                  'Grouped comparison is a planning bound, not a completed replacement: shared-A scope and KS leaf accumulation require explicit ownership/routing.',
                  'Existing packed input copies, linear DAG arithmetic, binary transforms and padding copies are priced. No new vectorized kernel exists.',
                  'New fused metadata binds, linear-DAG workspace views, output accumulation and runtime are not certified. Cold side-key charges must not be silently prebound for free.',
                  'No threshold changes or empirical performance claims. D8 unexecuted; D7 inner primitive was the2.335x failure.',
              ])
with (HERE/'result.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps({k: v for k, v in family.items() if k != 'positive'}))
print(json.dumps(result['historical_runtime']))
