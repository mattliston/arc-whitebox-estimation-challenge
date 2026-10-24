"""Fixed isolated HK15 qualification. No network inputs, labels, or dispatch changes."""
from pathlib import Path
from fractions import Fraction
import gc
import hashlib
import json
import os
import resource
import statistics
import sys
import time
import types
import numpy as np
import flopscope as flops

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
PARENT = H2 / 'hill_climb_star_projection32_v1/candidate.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
array_sha = lambda a: hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def write(name, data):
    with (HERE/name).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def load_module(name, path):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_text(), str(path), 'exec'), mod.__dict__)
    return mod


def summary(ctx):
    s = ctx.summary_dict()
    ops = s['operations']
    return dict(flops=int(s['flops_used']),
                backend_s=sum(float(v.get('flopscope_backend_time_s', 0)) for v in ops.values()),
                operations=sum(int(v.get('calls', v.get('count', 0))) for v in ops.values()))


def array_views(value):
    if isinstance(value, (tuple, list)):
        for v in value: yield from array_views(v)
    elif isinstance(value, dict):
        for v in value.values(): yield from array_views(v)
    elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
        yield value


def layout(product, external=()):
    """Harness metadata checks only; no estimator value or kernel computation."""
    arena = np.asarray(product.arena)
    scratch = np.asarray(product.scratch)
    checked = 0
    for side in product.sides:
        for a in array_views([side.T, side.chunks, side.pair, side.basis, side.ops]):
            assert np.shares_memory(np.asarray(a), arena)
            checked += 1
    for a in array_views(product.work):
        assert np.shares_memory(np.asarray(a), scratch)
        checked += 1
    allocations = [product.arena, product.scratch, product.left, product.right, product.products]
    allocations += [a for a in (product.pad_a, product.pad_b, product.pad_c) if a is not None]
    arrays = [np.asarray(a) for a in allocations]
    assert sum(a.nbytes for a in arrays) == product.workspace_bytes
    arrays += [np.asarray(a) for a in external]
    for i, a in enumerate(arrays):
        for b in arrays[:i]: assert not np.shares_memory(a, b)
    return dict(checked_shared_views=checked, allocations_disjoint=True,
                workspace_bytes=product.workspace_bytes)


def binary_bill(depth, shape):
    m, k, n = shape
    u = 1 << depth
    g = Fraction(7, 4)**depth
    load = g + Fraction(depth+3, 4)
    unwind = 2*g - Fraction(3, 4) + Fraction(7*(depth-1), 16)
    return int(load*(m*k+k*n) + unwind*m*n + 7**depth*(m//u)*(n//u)*(2*(k//u)-1))


def main():
    started = time.monotonic()
    seal = json.loads((HERE/'source_seal.json').read_text())
    for name, digest in seal['files'].items(): assert sha(HERE/name) == digest, name
    assert sha(PARENT) == seal['parent_sha256']
    assert all(os.environ.get(name) == '1' for name in
               ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'))
    programs = json.loads((HERE/'programs.json').read_text())
    descriptor = json.loads((HERE/'representative.json').read_text())
    assert programs['passed'] and descriptor['passed']
    assert sha(HERE/'programs.json') == descriptor['programs']['sha256']
    kernel = load_module('hk15_isolated', HERE/'kernel.py')
    fnp = kernel.fnp
    rng = np.random.default_rng(2026092815)
    fixtures = []
    # D3, leaf2: each certified orientation, both aligned and one-less padding.
    for rec in programs['records']:
        for padded in (False, True):
            shape = tuple(q*16-int(padded) for q in rec['shape'])
            m, k, n = shape
            a = rng.integers(-2, 3, size=(m,k)).astype(np.float32)
            b = rng.integers(-2, 3, size=(k,n)).astype(np.float32)
            input_hashes = (array_sha(a), array_sha(b))
            truth = a.astype(np.float64) @ b.astype(np.float64)
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                product = kernel.Product(rec, 3, shape)
            bind = summary(ctx)
            assert bind['flops'] == product.bind_flops
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                aa, bb = fnp.asarray(a), fnp.asarray(b)
                out = fnp.empty((m,n), dtype=fnp.float32)
            input_bill = summary(ctx)
            geometry = layout(product, (aa, bb, out))
            bills = []
            for repetition in range(2):
                with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                    returned = product(aa, bb, out)
                assert returned is out
                assert np.array_equal(np.asarray(out, dtype=np.float64), truth)
                assert (array_sha(aa), array_sha(bb)) == input_hashes
                bills.append(summary(ctx))
                assert bills[-1]['flops'] == kernel.warm_flops(rec, 3, shape)
            fixtures.append(dict(name=rec['name'], shape=shape, padded=padded, exact=True,
                                 bind=bind, input_conversion=input_bill, runs=bills, layout=geometry))
            del product, aa, bb, out, returned, a, b, truth
            gc.collect()
    write('fixtures.json', dict(passed=True, fixtures=fixtures,
                               source_seal_sha256=sha(HERE/'source_seal.json')))
    print(json.dumps(dict(event='fixtures_passed', count=len(fixtures))), flush=True)

    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    parent = load_module('s32_primitive_control', PARENT)
    est = parent.Estimator()
    t0 = time.monotonic()
    est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
              api_version='2', scratch_dir=None, submission_dir=None, seed=0))
    setup_s = time.monotonic()-t0
    chosen = descriptor['representative']
    olddepth, m, k, n = chosen['key']
    shape = (m,k,n)
    alt = chosen['alternative']
    rec = next(r for r in programs['records'] if r['name'] == alt['name'])
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        product = kernel.Product(rec, alt['depth'], shape)
    candidate_bind = summary(ctx)
    assert candidate_bind['flops'] == product.bind_flops == 651273255
    assert product.workspace_bytes == chosen['workspace_bytes'] == 2533228700
    assert kernel.warm_flops(rec, alt['depth'], shape) == alt['flops'] == 3501135335
    # Force and separately charge full incumbent binding, then hold the fused entries.
    dims = ((m,k), (k,n), (m,n))
    old_eager = parent._LVF_EAGER[0]
    for cache, (x,y) in zip((parent._IPT_A, parent._IPT_B, parent._IPT_C), dims):
        cache.pop((olddepth,x,y), None)
    parent._LVF_EAGER[0] = True
    try:
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            base_sides = [parent._ipt_side(j, olddepth, x, y) for j,(x,y) in enumerate(dims)]
    finally:
        parent._LVF_EAGER[0] = old_eager
    base_bind = summary(ctx)
    assert all(side[3] is not None for side in base_sides)
    u = 1 << olddepth
    predicted_base_bind = sum((7**olddepth + 7*4**(olddepth-1))*(x//u)*(y//u) for x,y in dims)
    assert base_bind['flops'] == predicted_base_bind
    a = rng.standard_normal((m,k)).astype(np.float32)
    b = rng.standard_normal((k,n)).astype(np.float32)
    input_hashes = (array_sha(a), array_sha(b))
    truth = a.astype(np.float64) @ b.astype(np.float64)
    norm = float(np.linalg.norm(truth))
    assert norm > 0 and np.isfinite(truth).all()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        aa, bb = fnp.asarray(a), fnp.asarray(b)
        out = fnp.empty((m,n), dtype=fnp.float32)
    input_bill = summary(ctx)
    geometry = layout(product, (aa,bb,out,parent._SCRATCH[0]))
    runs = dict(base=[], candidate=[])
    first, first_hashes, warmup = {}, {}, {}
    deterministic = True
    for rep in (-1,0,1,2):
        for arm in (('base','candidate') if rep % 2 == 0 else ('candidate','base')):
            parent._TR_LAST[0] = None
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                t0, c0 = time.monotonic(), time.process_time()
                returned = parent._ipt_product_ab(aa,bb,out,olddepth) if arm == 'base' else product(aa,bb,out)
                cpu, wall = time.process_time()-c0, time.monotonic()-t0
            assert returned is out
            value = np.asarray(out, dtype=np.float64)
            value_hash = array_sha(out)
            finite = bool(np.isfinite(value).all())
            deterministic &= arm not in first_hashes or first_hashes[arm] == value_hash
            if arm not in first:
                first[arm] = value.copy()
                first_hashes[arm] = value_hash
            row = summary(ctx)
            row.update(cpu_s=cpu, wall_s=wall, finite=finite, output_sha256=value_hash,
                       relative_error=float(np.linalg.norm(value-truth)/norm) if finite else None)
            row['expected_flops'] = binary_bill(olddepth, shape) if arm == 'base' else alt['flops']
            row['bill_exact'] = row['flops'] == row['expected_flops']
            if rep < 0: warmup[arm] = row
            else: runs[arm].append(row)
            print(json.dumps(dict(event='run', arm=arm, rep=rep, **row)), flush=True)
            del value, returned
    aggregate = {arm:{key:statistics.median(r[key] for r in rows) if all(r[key] is not None for r in rows) else None
                      for key in ('flops','cpu_s','wall_s','backend_s','operations','relative_error')}
                 for arm,rows in runs.items()}
    ratios = {key:aggregate['candidate'][key]/aggregate['base'][key] if aggregate['candidate'][key] is not None and aggregate['base'][key] else None
              for key in ('flops','cpu_s','wall_s','backend_s','relative_error')}
    allrows = list(warmup.values()) + [r for rows in runs.values() for r in rows]
    gates = dict(fixtures_exact=len(fixtures)==12, all_finite=all(r['finite'] for r in allrows),
                 repeated_outputs_exact=bool(deterministic), complete_bills_exact=all(r['bill_exact'] for r in allrows),
                 candidate_relative_error=all(r['relative_error'] is not None and r['relative_error'] <= 2e-4
                                              for r in [warmup['candidate']] + runs['candidate']),
                 candidate_bill_lower=aggregate['candidate']['flops'] < aggregate['base']['flops'],
                 inputs_unchanged=(array_sha(aa),array_sha(bb)) == input_hashes)
    result = dict(scope='Isolated fixed maximum-workspace primitive; no rollout, residual or bank qualification',
                  source_seal_sha256=sha(HERE/'source_seal.json'), passed_internal_gates=all(gates.values()),
                  guard_required=True, gates=gates, representative=chosen, input_conversion=input_bill,
                  candidate_bind=candidate_bind, base_bind=base_bind, layout=geometry,
                  setup_s=setup_s, warmup=warmup, runs=runs, aggregate=aggregate, ratios=ratios,
                  candidate_vs_base_relative_difference=float(np.linalg.norm(first['candidate']-first['base'])/norm) if all(r['finite'] for r in allrows) else None,
                  base_arena_bytes=int(parent._SCRATCH[0].nbytes),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  elapsed_s=time.monotonic()-started, targets_accessed=False,
                  timing_is_diagnostic=True, independent_arenas=True)
    write('result.json', result)
    print(json.dumps(dict(event='complete', gates=gates, ratios=ratios)), flush=True)


if __name__ == '__main__': main()
