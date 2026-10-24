"""Independent full-D6 parity, exact bills, and every packet liveness interval."""
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
OUT = HERE / 'unit_v1'
OUT.mkdir(exist_ok=False)
assert all(x['num_threads'] == 1 for x in threadpool_info())
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
spec = importlib.util.spec_from_file_location('read6_unit_candidate', HERE / 'candidate.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod


def meter(function):
    started = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        value = function()
    return value, dict(flops=ctx.summary_dict()['flops_used'], wall_s=time.monotonic() - started)


_, import_bill = meter(lambda: spec.loader.exec_module(mod))


def initialize():
    arena = fnp.empty((mod._PACKED_ELEMENTS,), dtype=fnp.float32)
    views = mod._cp3_bind_arena_views(arena, mod._PACKED_READ)
    return arena, views


(arena, views), binding_bill = meter(initialize)


def independent_frame(levels, m, k, n):
    """Separate per-frame allocations; original full-D6 loop supplies arithmetic."""
    forward = []
    old_a = old_b = None
    elements = 0

    def empty(shape):
        nonlocal elements
        count = int(np.prod(shape))
        assert count * 4 < 4294967296
        elements += count
        return fnp.empty(shape, dtype=fnp.float32)

    for level in range(1, levels + 1):
        r, t, c = m // 2**level, k // 2**level, n // 2**level
        a = empty((7,) * level + (r, t))
        b = empty((7,) * level + (t, c))
        aq = ((old_a[..., :r, :t], old_a[..., :r, t:],
               old_a[..., r:, :t], old_a[..., r:, t:]) if old_a is not None else None)
        bq = ((old_b[..., :t, :c], old_b[..., :t, c:],
               old_b[..., t:, :c], old_b[..., t:, c:]) if old_b is not None else None)
        forward.append((aq, bq, fnp.unstack(a, axis=0), fnp.unstack(b, axis=0), a[:2], b[:3]))
        old_a, old_b = a, b
    leaf = empty((7,) * levels + (m // 2**levels, n // 2**levels))
    current = leaf
    backward = []
    for level in range(levels, 0, -1):
        r, c = m // 2**level, n // 2**level
        scratch = empty((3,) + (7,) * (level - 1) + (r, c))
        corners = None
        if level > 1:
            destination = empty((7,) * (level - 1) + (2 * r, 2 * c))
            corners = (destination[..., :r, :c], destination[..., :r, c:],
                       destination[..., r:, :c], destination[..., r:, c:])
        backward.append((fnp.unstack(current, axis=0), fnp.unstack(scratch, axis=0), corners))
        if level > 1:
            current = destination
    return (forward, old_a, old_b, leaf, backward), elements * 4


def disjoint(intervals):
    ordered = sorted(intervals)
    assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), ordered
    assert all(0 <= a < b <= mod._PACKED_ELEMENTS for a, b in ordered)


def exact(specification, r, c):
    start, stop, shape = specification
    count = int(np.prod(shape[:-2])) * r * c
    assert start + count <= stop
    return start, start + count


def interval_certificate(m, k, n):
    pa = exact(mod._PACKED_READ[6], m // 16, k // 16)
    pb = exact(mod._PACKED_READ[7], k // 16, n // 16)
    pc = exact(mod._PACKED_READ[16], m // 16, n // 16)
    stages = []
    for q in (1, 2):
        a5 = (157351936, 157351936 + 7 * q * 343 * (m // 32) * (k // 32))
        b5 = (196689920, 196689920 + 7 * q * 343 * (k // 32) * (n // 32))
        a6 = (236027904, 236027904 + 49 * q * 343 * (m // 64) * (k // 64))
        b6 = (304869376, 304869376 + 49 * q * 343 * (k // 64) * (n // 64))
        c6 = (373710848, 373710848 + 49 * q * 343 * (m // 64) * (n // 64))
        c5 = (157351936, 157351936 + 7 * q * 343 * (m // 32) * (n // 32))
        s6 = (196689920, 196689920 + 21 * q * 343 * (m // 64) * (n // 64))
        s5 = (196689920, 196689920 + 3 * q * 343 * (m // 32) * (n // 32))
        group = dict(forward5=[pa, pb, pc, a5, b5],
                     forward6=[pa, pb, pc, a5, b5, a6, b6],
                     multiply6=[pa, pb, pc, a6, b6, c6],
                     backward6=[pa, pb, pc, c6, c5, s6],
                     backward5=[pa, pb, pc, c5, s5])
        for intervals in group.values():
            disjoint(intervals)
        stages.append(dict(q=q, **group))
    current = pc
    for unwind, depth in enumerate(range(4, 0, -1)):
        r, c = m // 2**depth, n // 2**depth
        scratch = exact(mod._PACKED_READ[20 + unwind], r, c)
        live = [current, scratch]
        if depth > 1:
            destination = exact(mod._PACKED_READ[17 + unwind], 2 * r, 2 * c)
            live.append(destination)
        disjoint(live)
        stages.append(dict(backward_depth=depth, live_intervals=live))
        if depth > 1:
            current = destination
    return stages


def formula(m, k, n, depth):
    g = Fraction(7, 4)**depth
    price = 2 * m * k * n * Fraction(7, 8)**depth - m * n * g
    price += Fraction(7, 3) * (m * k + k * n + m * n) * (g - 1)
    assert price.denominator == 1
    return int(price)


shapes = [(1024, 8192, 1024), (832, 8192, 768), (64, 1088, 64)]
records = []
first_hash = None
for index in range(4):
    which = index if index < 3 else 0
    m, k, n = shapes[which]
    rng = np.random.default_rng(190919600 + which)
    a_host = rng.standard_normal((m, k), dtype=np.float32) / np.float32(k**.5)
    b_host = rng.standard_normal((k, n), dtype=np.float32)
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        a, b = fnp.asarray(a_host), fnp.asarray(b_host)
        out = fnp.empty((m, n), dtype=fnp.float32)
    _, cold = meter(lambda: mod._packed_read6(a, b, out, views))
    predicted = np.asarray(out).copy()
    digest = hashlib.sha256(predicted.tobytes()).hexdigest()
    assert np.isfinite(predicted).all()
    if index == 3:
        assert digest == first_hash
        records.append(dict(shape=[m, k, n], repeat_after_interleave_exact=True, output_sha256=digest, warm=cold))
        break
    if index == 0:
        first_hash = digest
    _, warm = meter(lambda: mod._packed_read6(a, b, out, views))
    assert np.array_equal(predicted, np.asarray(out))
    assert warm['flops'] == formula(m, k, n, 6)
    (reference, reference_bytes), ref_setup = meter(lambda: independent_frame(6, m, k, n))
    original_frame = mod._packed_frame

    def reference_frame(levels, mm, kk, nn, vv):
        assert (levels, mm, kk, nn) == (6, m, k, n) and vv is views
        return reference

    mod._packed_frame = reference_frame
    try:
        _, ref_bill = meter(lambda: mod._packed_product_cached(a, b, out, views, 6))
    finally:
        mod._packed_frame = original_frame
    assert np.array_equal(predicted, np.asarray(out)), (m, k, n, 'Full D6 mismatch')
    assert ref_bill['flops'] == warm['flops']
    del reference
    _, parent_first = meter(lambda: mod._packed_read5(a, b, out, views))
    _, parent_warm = meter(lambda: mod._packed_read5(a, b, out, views))
    assert parent_warm['flops'] == formula(m, k, n, 5)
    if index < 2:
        assert warm['flops'] < parent_warm['flops']
        assert mod._rule_depth('read', m, k, n) == 6
    direct = a_host.astype(np.float64) @ b_host.astype(np.float64)
    relative_rms = float(np.sqrt(np.mean((predicted.astype(np.float64) - direct)**2) / np.mean(direct**2)))
    assert relative_rms <= 3e-4
    row = dict(shape=[m, k, n], cold=cold, warm=warm, reference_setup=ref_setup,
               reference_bill=ref_bill, reference_bytes=reference_bytes,
               parent5_first=parent_first, parent5_warm=parent_warm,
               full6_byte_exact=True, direct64_relative_rms=relative_rms,
               warm_bill_ratio=warm['flops'] / parent_warm['flops'],
               selected_depth=mod._rule_depth('read', m, k, n), output_sha256=digest,
               interval_certificate=interval_certificate(m, k, n))
    records.append(row)
    with (OUT / f'case_{index}.json').open('x') as stream:
        json.dump(row, stream, indent=2)
    print(json.dumps({key: value for key, value in row.items() if key != 'interval_certificate'}), flush=True)
    del predicted, direct, a_host, b_host, a, b, out

receipt = dict(candidate_sha256=hashlib.sha256((HERE / 'candidate.py').read_bytes()).hexdigest(),
               unit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               import_bill=import_bill, binding_bill=binding_bill, records=records,
               arena_bytes=mod._PACKED_ELEMENTS * 4, passed=True, truth_accessed=False)
with (OUT / 'result.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(dict(passed=True, arena_bytes=mod._PACKED_ELEMENTS * 4)), flush=True)
