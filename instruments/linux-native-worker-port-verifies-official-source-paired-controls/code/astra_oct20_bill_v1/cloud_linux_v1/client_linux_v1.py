"""NumPy-free Linux production-boundary science worker. Original group order is retained.

Fixture load, layer views, predict, fetch, finite checks and handle release remain
inside one BudgetContext, matching boundary_client_product_census_v5.py. Passive
Darwin hardware/GC-counter wrappers are omitted; official timing is untouched.
"""
import argparse
import array
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import types
from common_linux_v1 import check_sdk, delta, load_plan, no_governor, own_usage, sha, utc, verified, write_new

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--plan', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--candidate-sha', required=True)
p.add_argument('--ranks', required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--setup', choices=['paced', 'real'], default='paced')
p.add_argument('--warmup-each', action='store_true', help='selected-fixture fresh screens only; not science32')
a = p.parse_args()
assert importlib.util.find_spec('numpy') is None, 'client must be NumPy-free'
plan = load_plan(a.root, a.plan)
environment = check_sdk(a.root, plan, 'client')
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _connection

assert sha(a.candidate) == a.candidate_sha
no_governor(a.candidate.read_text())
ranks = [int(x) for x in a.ranks.split(',')]
assert len(ranks) == len(set(ranks))
fixtures = {rank: verified(a.root, plan['rows'][str(rank)]['weights']) for rank in ranks}
clock_path = verified(a.root, plan['setup_clock'])
spec = importlib.util.spec_from_file_location('arc_linux_setup_clock', clock_path)
setup_clock = importlib.util.module_from_spec(spec); spec.loader.exec_module(setup_clock)
out = a.out.resolve(); server_pid = int(os.environ['RELEASE_SERVER_PID'])
marks = Path(os.environ['RELEASE_SERVER_MARKS'])
count = dict(requests=0, free_trips=0, freed=0)
_send = _connection.Connection.send_recv; _drain = _connection.drain_pending


def counted_send(self, raw):
    count['requests'] += 1
    return _send(self, raw)


def counted_drain():
    got = _drain()
    if got:
        count['free_trips'] += 1; count['freed'] += len(got)
    return got


def mark():
    before = marks.stat().st_size if marks.exists() else 0
    os.kill(server_pid, signal.SIGUSR1)
    for _ in range(4000):
        if marks.exists() and marks.stat().st_size > before:
            return json.loads(marks.read_text().splitlines()[-1])
        time.sleep(.0005)
    raise RuntimeError('server did not answer mark signal')


def stored(after, before):
    return {k: v - before['put_bytes'].get(k, 0) for k, v in after['put_bytes'].items()
            if v != before['put_bytes'].get(k, 0)}


_connection.Connection.send_recv = counted_send
_connection.drain_pending = counted_drain
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
start = time.monotonic()
mod = types.ModuleType('release_candidate'); mod.__file__ = str(a.candidate)
sys.modules[mod.__name__] = mod
exec(compile(a.candidate.read_text(), str(a.candidate), 'exec'), mod.__dict__)
import_s = time.monotonic() - start
assert mod._FAST_RESCUE_FRAC == {}

# Same metadata-only product/prebind census used by the frozen Mac reference.
product_events = []; product_depth = [0]


def product_wrapper(parent, label):
    def observe(x, y):
        depth = product_depth[0]
        if len(x.shape) == len(y.shape) == 2:
            caller = sys._getframe(1)
            product_events.append((label, depth, caller.f_code.co_name, caller.f_lineno,
                                   int(x.shape[0]), int(x.shape[1]), int(y.shape[1]), str(x.dtype), str(y.dtype)))
            del caller
        product_depth[0] += 1
        try:
            return parent(x, y)
        finally:
            product_depth[0] -= 1
    observe.__name__ = getattr(parent, '__name__', label)
    observe.__wrapped__ = parent
    return observe


mod._mm = product_wrapper(mod._mm, 'mm')
mod._shared_old384_mm = product_wrapper(mod._shared_old384_mm, 'shared_old384')
keys = []; binds = []; seen = set()
old_side = mod._ipt_side; old_view = mod._ipt_view; old_acc = mod._ks_acc


def side(kind, level, rows, cols):
    if getattr(mod, '_R7_A', None) is None or mod._IPT_A is not mod._R7_A:
        keys.append((int(kind), int(level), int(rows), int(cols)))
    return old_side(kind, level, rows, cols)


def view(arena, start, shape):
    key = (int(start), tuple(int(x) for x in shape))
    binds.append((key[0], key[1], key in seen)); seen.add(key)
    return old_view(arena, start, shape)


def acc(rows, cols):
    keys.append((3, 6, int(rows), int(cols)))
    return old_acc(rows, cols)


mod._ipt_side = side; mod._ipt_view = view; mod._ks_acc = acc


def cache_snapshot():
    result = {}
    for label, env in [('peel', getattr(mod, '_SMX_ENV', None)), ('read', getattr(mod, '_ASTRA_SM40_ENV', None))]:
        for attr in ('_sm40_frames', '_sm40_cache', '_sm40_io_cache', '_sm40_outer_cache', '_sm40_group_cache', '_sm40_padding_grid', '_sm40_padding_cache'):
            result[label + '.' + attr] = sorted(map(repr, getattr(env, attr, {})))
    for attr in ('_IPT_A', '_IPT_B', '_IPT_C'):
        cache = getattr(mod, attr, {})
        result[attr] = sorted(map(repr, cache))
        result[attr + '.fused'] = sorted(repr(k) for k, v in cache.items() if v[3] is not None)
        result[attr + '.finished'] = sorted(repr(k) for k, v in cache.items() if v[2][1] is not None)
    return result


class MLP:
    def __init__(self, weights, rank):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f'release-{rank}'


slowdown = 1.0 if a.setup == 'paced' else None
setup_log, clock = setup_clock.install(mod, lambda: count['requests'], slowdown)
est = mod.Estimator()
m0, u0, start, r0 = mark(), own_usage(), time.monotonic(), count['requests']
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2',
                               scratch_dir=None, submission_dir=None, seed=0))
setup_s = time.monotonic() - start; m1 = mark()
entries = [(kind, key, entry) for kind, cache in enumerate((mod._IPT_A, mod._IPT_B, mod._IPT_C))
           for key, entry in cache.items() if (kind,) + key in mod._PB_SINGLE]
setup = dict(import_s=import_s, setup_s=setup_s, requests=count['requests'],
             actual_coverage=dict(bound=len(entries), fused=sum(e[3] is not None for _, _, e in entries),
                                  finished=sum(e[2][1] is not None for _, _, e in entries)),
             client=delta(own_usage(), u0), puts=m1['puts'] - m0['puts'], put_bytes=stored(m1, m0),
             key_uses=[list(k) + [v] for k, v in sorted(collections.Counter(keys).items())],
             bind_list=[[x, list(y), z] for x, y, z in binds], cache_keys=cache_snapshot(),
             **setup_clock.record(mod, setup_log, clock, start, r0))
assert mod._FAST_RESCUE_FRAC == {}
assert not setup['failed'], 'setup reported restored-after-failure'
write_new(out / 'manifest.json', dict(utc=utc(), candidate=str(a.candidate), sha256=a.candidate_sha,
          ranks=ranks, setup=setup, environment=environment, plan_sha256=sha(a.plan),
          harness_sha256=sha(__file__), setup_mode=a.setup, short_setup=a.setup == 'paced',
          warmup_each=a.warmup_each, hardware_counters_available=False,
          setup_is_release_gate=False, fixture_hashes={str(r): plan['rows'][str(r)]['weights']['sha256'] for r in ranks}))
print(json.dumps(dict(event='setup', setup_s=setup_s, requests=count['requests'])), flush=True)


def evaluate(rank, phase):
    count.update(requests=0, free_trips=0, freed=0)
    keys.clear(); binds.clear(); product_events.clear(); assert product_depth[0] == 0
    caches_before = cache_snapshot(); m0 = mark()
    with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f'release-{rank}') as ctx:
        blob = fnp.load(str(fixtures[rank]))
        weights = [blob[layer] for layer in range(16)]
        u0, start = own_usage(), time.monotonic()
        prediction = est.predict(MLP(weights, rank), 2**41)
        data, shape, dtype = prediction._fetch_data()
        wall = time.monotonic() - start
        cpu = delta(own_usage(), u0); observed = dict(count)
        assert tuple(shape) == (16, 1024), shape
        assert str(dtype) in ('float32', '<f4', '=f4', 'f4'), dtype
        assert len(data) == 65536
        decoded = array.array('f'); decoded.frombytes(data)
        nonfinite = sum(1 for x in decoded if not math.isfinite(x))
        del blob, weights, prediction
        flops_used = int(ctx.summary_dict()['flops_used'])
    summary = ctx.summary_dict(); m1 = mark(); assert product_depth[0] == 0
    assert nonfinite == 0 and len(decoded) == 16384
    assert mod._FAST_RESCUE_FRAC == {} and not getattr(mod, '_FAST_RESCUE_ON', False)
    row = dict(rank=rank, phase=phase, flops=flops_used, util=flops_used / 2**41, wall_s=wall,
               backend_s=summary['flopscope_backend_time_s'], overhead_s=summary['flopscope_overhead_time_s'],
               residual_s=summary['residual_wall_time_s'], context_wall_s=summary['wall_time_s'],
               requests=observed['requests'], free_trips=observed['free_trips'], freed_handles=observed['freed'],
               client=cpu, server=m1['process'], peak_handles=m1['peak_handles'],
               puts=m1['puts'] - m0['puts'], put_bytes=stored(m1, m0),
               comms=[x - y for x, y in zip(m1['comms'], m0['comms'])],
               operations={k: [v['calls'], v['flop_cost']] for k, v in summary['operations'].items()},
               key_uses=[list(k) + [v] for k, v in sorted(collections.Counter(keys).items())],
               bind_list=[[x, list(y), z] for x, y, z in binds],
               product_geometry_counts=[list(k) + [v] for k, v in sorted(collections.Counter(product_events).items())],
               caches_before=caches_before, caches_after=cache_snapshot(), n_out=len(decoded), nonfinite=nonfinite,
               dtype=str(dtype), prediction_sha256=hashlib.sha256(data).hexdigest(), hardware_counters_available=False,
               residual_cycles=None, residual_instructions=None,
               fast_rescue_stats=dict(getattr(mod, '_FAST_RESCUE_STATS', {})),
               fast_rescue_control=dict(getattr(mod, '_FAST_RESCUE_CONTROL', {})))
    for name in ('_SM40_COMPILE_STATS', '_MRS_STATS', '_ASTRA_SM40_STATS', '_O20_SMDIRECT_STATS', '_GCF_STATS'):
        row[name] = dict(getattr(mod, name, {}))
    if phase == 'scored':
        with (out / f'pred_{rank}.bin').open('xb') as f:
            f.write(data)
    del decoded, data
    return row


with (out / 'rows.jsonl').open('x') as stream:
    for rank in ranks:
        if a.warmup_each:
            warm = evaluate(rank, 'warmup')
            with (out / 'warmup.jsonl').open('a') as f:
                f.write(json.dumps(warm) + '\n')
        row = evaluate(rank, 'scored')
        stream.write(json.dumps(row) + '\n'); stream.flush()
        print(json.dumps({k:row[k] for k in ('rank', 'flops', 'requests', 'wall_s', 'prediction_sha256')}), flush=True)
print('ALL_NETWORKS_COMPLETE', flush=True)
