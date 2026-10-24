"""Actual metered D5 cache parity, saved-work and physical-owner checks."""
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
spec = importlib.util.spec_from_file_location('weight5_unit_candidate', HERE / 'candidate.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod


def meter(fn):
    started = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        value = fn()
    return value, dict(flops=ctx.summary_dict()['flops_used'], wall_s=time.monotonic() - started)


_, import_bill = meter(lambda: spec.loader.exec_module(mod))


def initialize():
    arena = fnp.empty((mod._PACKED_ELEMENTS,), dtype=fnp.float32)
    mod._SCRATCH = [arena, mod._cp3_bind_arena_views(arena, mod._PACKED_PUSH),
                    mod._cp3_bind_arena_views(arena, mod._PACKED_READ), None, None, None, None]
    mod._bind_push6(arena)


_, setup_bill = meter(initialize)
rng = np.random.default_rng(190919500)
records = []
network = object()
for layer, k in enumerate((992, 768)):
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        weight = fnp.asarray(rng.standard_normal((1024, 1024), dtype=np.float32) / np.float32(32))
        previous = fnp.arange(k)
        different_owner = fnp.arange(k)
        left = weight[previous, :].T
        rights = {n: fnp.asarray(rng.standard_normal((k, n), dtype=np.float32)) for n in (1024, 384, 32)}
        foreign = fnp.asarray(rng.standard_normal((1024, k), dtype=np.float32))
    assert mod._rule_depth('sq', 1024, k, 1024) == mod._rule_depth('sq', 1024, k, 384) == 5
    assert mod._rule_depth('sq', 1024, k, 32) == 0
    baseline = {}
    for n, right in rights.items():
        first, _ = meter(lambda: mod._mm(left, right))
        again, bill = meter(lambda: mod._mm(left, right))
        assert np.array_equal(np.asarray(first), np.asarray(again))
        baseline[n] = (np.asarray(again).copy(), bill)
        del first, again

    scope = mod._w5_port_new(network)
    mod._W5_PORT_SCOPE = scope
    mod._w5_port_begin(scope, network, layer, weight, previous)

    def call(n, role):
        function = mod._shared_old384_mm if role == 'shared_basis' else mod._mm
        return mod._w5_port_call(scope, weight, previous, role, function, left, rights[n])

    result, creator_bill = meter(lambda: call(1024, 'cov_outer'))
    assert np.array_equal(baseline[1024][0], np.asarray(result))
    copied = 16807 * (1024 // 32) * (k // 32)
    omitted = 36827 * 1024 * k // 1024
    assert creator_bill['flops'] - baseline[1024][1]['flops'] == copied
    owner = scope['cache']
    assert owner._leaf is owner._backing and tuple(owner._leaf.shape) == (7, 7, 7, 7, 7, 32, k // 32)

    # This ordinary product overwrites the same original arena, not the owned copy.
    _, foreign_bill = meter(lambda: mod._mm(foreign, rights[1024]))
    result, hit_bill = meter(lambda: call(384, 'shared_basis'))
    assert np.array_equal(baseline[384][0], np.asarray(result))
    assert baseline[384][1]['flops'] - hit_bill['flops'] == omitted
    result, fallback_bill = meter(lambda: call(32, 'newborn_common'))
    assert np.array_equal(baseline[32][0], np.asarray(result))
    assert baseline[32][1]['flops'] == fallback_bill['flops']
    result, second_hit_bill = meter(lambda: call(1024, 'star_push'))
    assert np.array_equal(baseline[1024][0], np.asarray(result))
    assert baseline[1024][1]['flops'] - second_hit_bill['flops'] == omitted

    rejected = False
    try:
        meter(lambda: mod._w5_port_call(scope, weight, different_owner, 'cov_outer', mod._mm, left, rights[1024]))
    except RuntimeError as error:
        assert 'provenance' in str(error)
        rejected = True
    assert rejected
    mod._w5_port_end_layer(scope)
    mod._w5_port_finish(scope)
    mod._W5_PORT_SCOPE = None
    assert owner._leaf is None and owner._backing is None and owner._state == 'closed'
    assert all(scope[key] is None for key in ('network', 'generation', 'weight', 'previous', 'grant', 'cache'))
    assert mod._W5_PORT_LAST['creators'] == 1 and mod._W5_PORT_LAST['hits'] == 2
    assert mod._W5_PORT_LAST['copied_flops'] == copied
    assert mod._W5_PORT_LAST['removed_flops'] == 2 * omitted
    row = dict(layer=layer, left_shape=[1024, k], baseline={n: b for n, (_, b) in baseline.items()},
               creator=creator_bill, hit=hit_bill, second_hit=second_hit_bill,
               fallback=fallback_bill, foreign_overwrite=foreign_bill,
               expected_copy_flops=copied, expected_omitted_per_hit=omitted,
               net_saved_flops=2 * omitted - copied, metadata=mod._W5_PORT_LAST,
               all_byte_exact=True, different_index_owner_rejected=True, all_numeric_owners_retired=True)
    records.append(row)
    print(json.dumps(row), flush=True)
    del result, weight, previous, different_owner, left, rights, foreign, baseline, owner, scope

receipt = dict(candidate_sha256=hashlib.sha256((HERE / 'candidate.py').read_bytes()).hexdigest(),
               unit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               truth_accessed=False, import_bill=import_bill, setup_bill=setup_bill,
               records=records, passed=True)
with (OUT / 'result.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
