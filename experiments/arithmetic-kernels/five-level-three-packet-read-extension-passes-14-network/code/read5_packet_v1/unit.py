"""Truth-free native arithmetic, billing, liveness and interleave certificate."""
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
spec = importlib.util.spec_from_file_location('read5_unit_candidate', HERE / 'candidate.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod


def meter(function):
    began = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        result = function()
    return result, dict(flops=ctx.summary_dict()['flops_used'], wall_s=time.monotonic() - began)


_, import_bill = meter(lambda: spec.loader.exec_module(mod))


def separated_specs(levels, m, k, n):
    specs = []
    cursor = 0

    def add(shape):
        nonlocal cursor
        size = int(np.prod(shape))
        specs.append((cursor, cursor + size, shape))
        cursor += size

    for depth in range(1, levels + 1):
        add((7,) * depth + (m // 2**depth, k // 2**depth))
        add((7,) * depth + (k // 2**depth, n // 2**depth))
    for _ in range(2 * levels):
        add((1, 1))  # Unused legacy slots; all live arrays get disjoint storage.
    add((7,) * levels + (m // 2**levels, n // 2**levels))
    for depth in range(levels - 1, 0, -1):
        add((7,) * depth + (m // 2**depth, n // 2**depth))
    for depth in range(levels, 0, -1):
        add((3,) + (7,) * (depth - 1) + (m // 2**depth, n // 2**depth))
    assert len(specs) == 6 * levels
    return tuple(specs), cursor


reference_specs, reference_size = separated_specs(5, 1024, 8192, 1024)


def initialize():
    arena = fnp.empty((mod._PACKED_ELEMENTS,), dtype=fnp.float32)
    views = mod._cp3_bind_arena_views(arena, mod._PACKED_READ)
    ref_arena = fnp.empty((reference_size,), dtype=fnp.float32)
    ref_views = mod._cp3_bind_arena_views(ref_arena, reference_specs)
    return arena, views, ref_arena, ref_views


(arena, views, ref_arena, ref_views), binding_bill = meter(initialize)


def disjoint(intervals):
    ordered = sorted(intervals)
    assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), ordered
    assert all(0 <= a < b <= mod._PACKED_ELEMENTS for a, b in intervals)


def exact_interval(specification, rows, columns):
    start, stop, shape = specification
    count = int(np.prod(shape[:-2])) * rows * columns
    assert start + count <= stop
    return start, start + count


def certificate(m, k, n):
    input_a = exact_interval(mod._PACKED_READ[6], m // 16, k // 16)
    input_b = exact_interval(mod._PACKED_READ[7], k // 16, n // 16)
    output = exact_interval(mod._PACKED_READ[16], m // 16, n // 16)
    stages = []
    for q in (2, 3):
        ca = (157351936, 157351936 + 7 * q * 343 * (m // 32) * (k // 32))
        cb = (216358912, 216358912 + 7 * q * 343 * (k // 32) * (n // 32))
        cc = (275365888, 275365888 + 7 * q * 343 * (m // 32) * (n // 32))
        ss = (157351936, 157351936 + 3 * q * 343 * (m // 32) * (n // 32))
        disjoint([input_a, input_b, ca, cb, cc, output])
        disjoint([input_a, input_b, cc, ss, output])
        stages.append(dict(packet=q, transforms_and_matmul=[input_a, input_b, ca, cb, cc, output],
                           reconstruction=[input_a, input_b, cc, ss, output]))
    current = output
    for unwind, depth in enumerate(range(4, 0, -1)):
        rows, columns = m // 2**depth, n // 2**depth
        scratch = exact_interval(mod._PACKED_READ[20 + unwind], rows, columns)
        live = [current, scratch]
        if depth > 1:
            dest = exact_interval(mod._PACKED_READ[17 + unwind], 2 * rows, 2 * columns)
            live.append(dest)
        disjoint(live)
        stages.append(dict(backward_depth=depth, live_intervals=live))
        if depth > 1:
            current = dest
    return stages


shapes = [(1024, 8192, 1024), (832, 8192, 768), (32, 1056, 32)]
records = []
first_hash = None
for index in range(4):
    shape_index = index if index < 3 else 0
    m, k, n = shapes[shape_index]
    rng = np.random.default_rng(190919000 + shape_index)
    a_host = rng.standard_normal((m, k), dtype=np.float32) / np.float32(k**.5)
    b_host = rng.standard_normal((k, n), dtype=np.float32)
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        a, b = fnp.asarray(a_host), fnp.asarray(b_host)
        output = fnp.empty((m, n), dtype=fnp.float32)
    _, d5_first = meter(lambda: mod._packed_read5(a, b, output, views))
    predicted = np.asarray(output).copy()
    output_hash = hashlib.sha256(predicted.tobytes()).hexdigest()
    assert np.isfinite(predicted).all()
    if index == 3:
        assert output_hash == first_hash
        records.append(dict(shape=[m, k, n], interleaved_repeat_exact=True, output_sha256=output_hash,
                            packet5_warm=d5_first))
        break
    if index == 0:
        first_hash = output_hash
    _, d5_warm = meter(lambda: mod._packed_read5(a, b, output, views))
    assert np.array_equal(predicted, np.asarray(output))
    _, ref_bill = meter(lambda: mod._packed_product_cached(a, b, output, ref_views, 5))
    exact = np.array_equal(predicted, np.asarray(output))
    assert exact, (m, k, n, float(np.max(np.abs(predicted - np.asarray(output)))))
    _, d4_warm = meter(lambda: mod._packed_product_cached(a, b, output, views, 4))
    direct = a_host.astype(np.float64) @ b_host.astype(np.float64)
    relative_rms = float(np.sqrt(np.mean((predicted.astype(np.float64) - direct)**2) / np.mean(direct**2)))
    assert relative_rms <= 2e-4
    if index < 2:
        assert d5_warm['flops'] < d4_warm['flops']
    record = dict(shape=[m, k, n], packet5_first=d5_first, packet5_warm=d5_warm,
                  full5_first=ref_bill, depth4_warm=d4_warm, full5_byte_exact=exact,
                  direct64_relative_rms=relative_rms, output_sha256=output_hash,
                  bill_ratio=d5_warm['flops'] / d4_warm['flops'],
                  selected_depth=mod._rule_depth('read', m, k, n),
                  interval_certificate=certificate(m, k, n))
    records.append(record)
    with (OUT / f'case_{index}.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k != 'interval_certificate'}), flush=True)
    del predicted, direct, a_host, b_host, a, b, output

result = dict(candidate_sha256=hashlib.sha256((HERE / 'candidate.py').read_bytes()).hexdigest(),
              unit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              truth_accessed=False, import_bill=import_bill, binding_bill=binding_bill,
              candidate_arena_bytes=mod._PACKED_ELEMENTS * 4,
              reference_arena_bytes=reference_size * 4, records=records, passed=True)
with (OUT / 'result.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(dict(passed=True, candidate_arena_bytes=mod._PACKED_ELEMENTS * 4,
                      reference_arena_bytes=reference_size * 4)), flush=True)
