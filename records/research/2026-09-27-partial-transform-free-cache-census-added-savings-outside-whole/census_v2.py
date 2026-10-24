"""Observe ordered partial arithmetic reuse; never return cached arithmetic."""
from pathlib import Path
import ctypes
import functools
import gc
import hashlib
import json
import resource
import struct
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent
H = D.parent
P2 = H.parent
BANK = H / 'candidate_old_both_tail_trace_bounded_v6.py'
BLOCK = 4096
EXPECTED = '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def once(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def operand_keys(op, x, y, block=BLOCK):
    a, b = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    header = (op + ':' + a.dtype.str + ':' + b.dtype.str + ':').encode()
    aa, bb = a.ravel(order='C'), b.ravel(order='C')
    keys = []
    for start in range(0, a.size, block):
        end = min(start + block, a.size)
        h = hashlib.sha256(header + struct.pack('<I', end - start))
        h.update(memoryview(aa[start:end]).cast('B'))
        h.update(memoryview(bb[start:end]).cast('B'))
        keys.append((end - start, h.digest()))
    assert sum(n for n, _ in keys) == a.size
    return keys, a.shape


def output_keys(value, block=BLOCK):
    a = np.asarray(value).ravel(order='C')
    return [(min(block, a.size - start), hashlib.sha256(memoryview(a[start:start + block]).cast('B')).digest())
            for start in range(0, a.size, block)]


def qualify():
    a = np.arange(73 * 67, dtype=np.float32).reshape(73, 67)
    cases = [(a, a / 3), (a.T, a.T / 3), (a[::-1, ::-2], a[::-1, ::-2] / 3),
             (a, np.array(2, np.float32)), (a[:0], a[:0]),
             (np.array([0., -0., 1.], np.float32), np.array([-0., 0., 2.], np.float32))]
    witnesses = []
    for op in ['add', 'subtract']:
        for x, y in cases:
            before = [np.asarray(v).tobytes() for v in (x, y)]
            keys, shape = operand_keys(op, x, y)
            aa, bb = np.broadcast_arrays(x, y)
            flat_a, flat_b = aa.ravel(order='C'), bb.ravel(order='C')
            header = (op + ':' + aa.dtype.str + ':' + bb.dtype.str + ':').encode()
            direct = []
            for start in range(0, aa.size, BLOCK):
                stop = min(start + BLOCK, aa.size)
                xa = b''.join(flat_a[i:i + 1].tobytes() for i in range(start, stop))
                xb = b''.join(flat_b[i:i + 1].tobytes() for i in range(start, stop))
                direct.append((stop - start, hashlib.sha256(header + struct.pack('<I', stop - start) + xa + xb).digest()))
            assert keys == direct and tuple(shape) == aa.shape
            assert before == [np.asarray(v).tobytes() for v in (x, y)]
            z = getattr(np, op)(x, y)
            outputs = output_keys(z)
            assert [n for n, _ in outputs] == [n for n, _ in keys]
            assert outputs == [(min(BLOCK, z.size - s), hashlib.sha256(z.ravel(order='C')[s:s + BLOCK].tobytes()).digest())
                               for s in range(0, z.size, BLOCK)]
            witnesses.extend((op, n, key, out) for (n, key), (_, out) in zip(keys, outputs))
    copied = list(witnesses)
    witnesses.extend(copied)
    seen = {}
    saved = 0
    for op, n, key, out in witnesses:
        if key in seen:
            assert seen[key] == out
            saved += n
        else:
            seen[key] = out
    exhaustive = sum(v[1] for i, v in enumerate(witnesses)
                     if any(w[2] == v[2] for w in witnesses[:i]))
    assert saved == exhaustive and saved > 0
    p = np.array([0.], np.float32)
    q = np.array([-0.], np.float32)
    assert operand_keys('add', p, p)[0] != operand_keys('add', q, p)[0]
    assert operand_keys('add', p, p)[0] != operand_keys('subtract', p, p)[0]
    x, y = a.copy(), (a / 3).copy()
    old_keys = operand_keys('add', x, y)[0]
    expected = x + y
    np.add(x, y, out=x)
    assert np.array_equal(x, expected) and old_keys != operand_keys('add', x, y)[0]
    return dict(passed=True, scalar_byte_cases=12, exhaustive_repeat_count=True,
                signed_zero_distinct=True, operation_distinct=True, in_place_observation=True)


assert sha(BANK) == EXPECTED
assert not (D / 'result.json').exists()
once(D / 'launch_seal_v2.json', {'files': {str(p): sha(p) for p in [BANK, D / 'PROTOCOL.md', Path(__file__)]},
                            'targets_accessed': False})
qualification = qualify()
once(D / 'qualification.json', qualification)
current, active, catalog = {}, {}, {}
operations, transforms, examples = [], [], []
original_ops = {name: getattr(fnp, name) for name in ['add', 'subtract']}


def arithmetic(name, original):
    @functools.wraps(original)
    def call(x, y, *args, **kwargs):
        if not active:
            return original(x, y, *args, **kwargs)
        keys, shape = operand_keys(name, x, y)
        before = int(current['ctx'].flops_used)
        value = original(x, y, *args, **kwargs)
        paid = int(current['ctx'].flops_used) - before
        out = np.asarray(value)
        assert out.shape == shape and out.dtype == np.float32
        count = int(out.size)
        assert paid == count, (name, shape, paid, count)
        op_index = len(operations)
        saving, repeats = 0, 0
        outs = output_keys(out)
        assert len(keys) == len(outs)
        for j, ((n, key), (on, output_hash)) in enumerate(zip(keys, outs)):
            assert n == on
            earlier = catalog.get(key)
            if earlier is None:
                catalog[key] = (output_hash, op_index, j)
            else:
                assert earlier[0] == output_hash, (op_index, j, earlier[1:])
                saving += n
                repeats += 1
                if len(examples) < 100:
                    examples.append(dict(operation=op_index, chunk=j, first_operation=earlier[1],
                                         first_chunk=earlier[2], key_sha256=key.hex(), output_sha256=output_hash.hex(), count=n))
        assert len(catalog) <= 5_000_000
        operations.append(dict(index=op_index, transform=active['index'], op=name, shape=list(shape),
            paid=paid, chunks=len(keys), repeated_chunks=repeats, removable_arithmetic=saving))
        return value
    return call


# Patch before source import/setup so eager operation tuples receive observers.
for name, original in original_ops.items():
    setattr(fnp, name, arithmetic(name, original))
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('partial_transform_observed_bank')
mod.__file__ = str(BANK)
sys.modules[mod.__name__] = mod
exec(compile(BANK.read_text(), str(BANK), 'exec'), mod.__dict__)
original_transforms = {n: getattr(mod, n) for n in ['_ipt_forward_a_ab', '_ipt_forward_b_ab']}


def transform(side, original):
    def call(a, depth, rows, cols):
        if not current:
            return original(a, depth, rows, cols)
        assert not active
        start = len(operations)
        active.update(index=len(transforms))
        before = int(current['ctx'].flops_used)
        try:
            result = original(a, depth, rows, cols)
        finally:
            active.clear()
        paid = int(current['ctx'].flops_used) - before
        ops = operations[start:]
        arithmetic_paid = sum(r['paid'] for r in ops)
        assert 0 <= arithmetic_paid <= paid
        transforms.append(dict(index=len(transforms), side=side, depth=depth, dims=[rows, cols],
            flops=paid, arithmetic_flops=arithmetic_paid, operation_count=len(ops),
            removable_arithmetic=sum(r['removable_arithmetic'] for r in ops)))
        return result
    return call


for name, original in original_transforms.items():
    setattr(mod, name, transform('A' if name.endswith('a_ab') else 'B', original))
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
          api_version='2', scratch_dir=None, submission_dir=None, seed=0))
store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
weights_rows = json.loads((store / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
controls = {r['rank']: r for r in map(json.loads, (H / 'both_tail_train2_v6/rows.jsonl').read_text().splitlines())}
relief = ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
relief.restype = ctypes.c_size_t
network_rows = []
begin = time.monotonic()
try:
    for rank in [35, 100]:
        assert not current and not active and not catalog
        wp = store / Path(weights_rows[rank]['path']).parent / 'weights.npy'
        weights = np.load(wp)
        mlp = types.SimpleNamespace(width=1024, depth=16,
              weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
        t0 = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            current.update(rank=rank, ctx=ctx)
            y = np.asarray(est.predict(mlp, 2**41), np.float64)
            current.clear()
        bill = int(ctx.summary_dict()['flops_used'])
        base = np.load(H / 'both_tail_train2_v6' / f'pred_{rank}.npy')
        assert y.shape == base.shape and y.dtype == base.dtype and y.tobytes() == base.tobytes()
        assert bill == controls[rank]['flops']
        with (D / f'pred_{rank}.npy').open('xb') as f:
            np.save(f, y)
        assert sha(D / f'pred_{rank}.npy') == sha(H / 'both_tail_train2_v6' / f'pred_{rank}.npy')
        old = json.loads((H / 'factor_transform_reuse_v1' / f'transforms_{rank}.json').read_text())
        assert len(transforms) == len(old)
        assert [(r['side'], r['depth'], r['dims'], r['flops']) for r in transforms] == [(r['side'], r['depth'], r['dims'], r['flops']) for r in old]
        saved = sum(r['removable_arithmetic'] for r in operations)
        row = dict(rank=rank, flops=bill, all16_prediction_file_exact=True, complete_bill_exact=True,
            prior_transform_trace_exact=True, transforms=len(transforms), operations=len(operations),
            distinct_block_keys=len(catalog), total_chunks=sum(r['chunks'] for r in operations),
            repeated_chunks=sum(r['repeated_chunks'] for r in operations),
            input_transform_flops=sum(r['flops'] for r in transforms),
            observed_transform_arithmetic=sum(r['paid'] for r in operations),
            optimistic_removable_flops=saved, optimistic_saving_pct=100 * saved / bill,
            weights_sha256=sha(wp), elapsed_s=time.monotonic() - t0)
        network_rows.append(row)
        once(D / f'operations_{rank}.json', operations)
        once(D / f'transforms_{rank}.json', transforms)
        once(D / f'examples_{rank}.json', examples)
        once(D / f'network_{rank}.json', row)
        print(json.dumps(dict(event='network_complete', **row)), flush=True)
        catalog.clear(); operations.clear(); transforms.clear(); examples.clear()
        del weights, mlp, y, base, old
        gc.collect()
        relief(None, 0)
finally:
    current.clear(); active.clear()
    for name, original in original_ops.items():
        setattr(fnp, name, original)
    for name, original in original_transforms.items():
        setattr(mod, name, original)

result = dict(networks=network_rows, passes_ceiling_gate=all(r['optimistic_saving_pct'] >= 1.5 for r in network_rows),
    parent_sha256=EXPECTED, protocol_sha256=sha(D / 'PROTOCOL.md'), census_sha256=sha(Path(__file__)),
    qualification=qualification, targets_accessed=False, valid40_accessed=False, fits_run=False,
    candidate_built=False, submitted=False, block_size=BLOCK,
    scope='Free unlimited reuse of identical ordered 4096-scalar arithmetic blocks; not deployed savings or a bound on other partitions',
    elapsed_s=time.monotonic() - begin, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
once(D / 'result.json', result)
assert sum(p.stat().st_size for p in D.glob('*') if p.is_file()) < 50_000_000
print(json.dumps(dict(event='complete', passes=result['passes_ceiling_gate'],
                     elapsed_s=result['elapsed_s'], peak_rss_bytes=result['peak_rss_bytes'])), flush=True)
