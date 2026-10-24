"""TAG opus-release: numpy-free worker at the production boundary (flopscope client -> flopscope server), as a platform worker runs it.

setup() once (timed, requests counted), then the networks one after the other in ONE process (first = cold, later = warm). No
resets between networks. Per network: bill, wall of predict + fetch, the client's own timing split (backend = server numpy time,
overhead = flopscope machinery + wire, residual = participant Python: the platform's three telemetry fields), requests, hidden free
round trips, the server's and the client's kernel counters (user/system seconds, cycles, instructions,
peak footprint; rusage.py), tree-cache sizes, non-finite outputs. Predictions are saved as raw bytes for parity checks.
Census fields for fast_path.py (2026-10-02): operations = billed [calls, flops] per operation family (the context's own summary),
puts and put_bytes = arrays the server stored and their logical bytes by number type (also recorded for the setup).
Derived from fable_porter_v1/census_boundary_setup_v2.py and H/handle_client_v2.py.
--short-setup runs the estimator's setup under the platform's clock (setup_clock.py: a normal platform worker);
--setup-slowdown X the same for a worker X times slower in setup (0 = no guard: everything bound and fused).
--sites records every request's call site in the candidate and its operation name (rows: by_func, by_site; request_sites.py).
--profile runs every prediction under cProfile and writes profile_<rank>.pstats (where the client's own Python time goes:
residual_report.py); the session's timings and counters are then inflated and must not be used for a projection.
usage (client python; FLOPSCOPE_SERVER_URL, RELEASE_SERVER_PID, RELEASE_SERVER_MARKS set by boundary_run.py):
       boundary_client.py <candidate.py> <ranks> <outdir> [--sites] [--profile] [--short-setup | --setup-slowdown X]"""
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

sys.path.insert(0,'/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release')
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _connection

import rusage
import setup_clock

assert importlib.util.find_spec('numpy') is None, 'must run numpy-free'
P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate = Path(sys.argv[1]).resolve()
ranks = [int(x) for x in sys.argv[2].split(',')]
out = Path(sys.argv[3]).resolve()
SITES = '--sites' in sys.argv[4:]
SLOWDOWN = (float(sys.argv[sys.argv.index('--setup-slowdown') + 1]) if '--setup-slowdown' in sys.argv[4:]
            else 1.0 if '--short-setup' in sys.argv[4:] else None)      # None: the candidate's own wall clock
PROFILE = '--profile' in sys.argv[4:]
if PROFILE:
    import cProfile
if SITES:
    import msgpack
SERVER_PID = int(os.environ['RELEASE_SERVER_PID'])
name = str(candidate)
MARKS = Path(os.environ['RELEASE_SERVER_MARKS'])


def stored(m1, m0):
    """Logical bytes by number type of the arrays the server stored between two marks."""
    return {k: v - m0['put_bytes'].get(k, 0) for k, v in m1['put_bytes'].items() if v - m0['put_bytes'].get(k, 0)}


def mark():
    """One line of the server's cumulative counters (handles, kernel seconds/cycles/instructions by operation)."""
    before = MARKS.stat().st_size if MARKS.exists() else 0
    os.kill(SERVER_PID, signal.SIGUSR1)
    for _ in range(4000):
        if MARKS.exists() and MARKS.stat().st_size > before:
            break
        time.sleep(.0005)
    else:
        raise RuntimeError('server did not answer the mark signal')
    return json.loads(MARKS.read_text().splitlines()[-1])


count = dict(requests=0, free_trips=0, freed=0)
by_func = collections.Counter()
by_site = collections.Counter()
_send = _connection.Connection.send_recv
_drain = _connection.drain_pending


def counted_send(self, raw):
    count['requests'] += 1
    if SITES:
        frame = sys._getframe(1)
        while frame is not None and frame.f_code.co_filename != name:
            frame = frame.f_back
        if frame is not None:
            op = msgpack.unpackb(raw, raw=False).get('op', '?')          # the request's operation name (diagnostic runs only)
            by_func[frame.f_code.co_name] += 1
            by_site[f'{frame.f_lineno} {frame.f_code.co_name} {op}'] += 1
    return _send(self, raw)


def counted_drain():
    got = _drain()
    if got:
        count['free_trips'] += 1
        count['freed'] += len(got)
    return got


_connection.Connection.send_recv = counted_send
_connection.drain_pending = counted_drain

# Residual in load-independent units. The platform bills residual = wall - dispatch (the participant's Python outside flopscope's
# dispatch spans). Its local wall reading swings +-50% with machine load, so the same split is also taken in CPU cycles and
# instructions: the two counter reads sit INSIDE the outermost dispatch span (their cost is overhead, not residual). The wrapper
# is defined in flopscope._dispatch's own namespace because the span refuses callers from outside the package.
from flopscope import _dispatch
_dispatch.__dict__.update(_release_raw=rusage.raw, _release_depth=[0], _release_acc=[0, 0],
                          _release_span=_dispatch._counted_span)
exec('''
@contextmanager
def _counted_span():
    with _release_span():
        top = _release_depth[0] == 0
        if top:
            c0, i0 = _release_raw()
        _release_depth[0] += 1
        try:
            yield
        finally:
            _release_depth[0] -= 1
            if top:
                c1, i1 = _release_raw()
                _release_acc[0] += c1 - c0
                _release_acc[1] += i1 - i0
''', _dispatch.__dict__)
DISPATCH = _dispatch._release_acc      # [cycles, instructions] spent inside dispatch spans so far

stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
t_import = time.monotonic()
mod = types.ModuleType('release_candidate')
mod.__file__ = name
sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), name, 'exec'), mod.__dict__)
import_s = time.monotonic() - t_import
log, clock = setup_clock.install(mod, lambda: count['requests'], SLOWDOWN)
rows = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']


class MLP:
    def __init__(self, weights, rank):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f'release-{rank}'


est = mod.Estimator()
setup = dict(import_s=import_s, setup_s=0.0, requests=0, pb_log=[], defined=hasattr(est, 'setup'))
if setup['defined']:
    m0, own0, start, r0 = mark(), rusage.read(), time.monotonic(), count['requests']
    setup_dispatch0 = list(DISPATCH)
    est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None,
                                    submission_dir=None, seed=0))
    elapsed = time.monotonic() - start
    m1 = mark()
    setup.update(setup_s=elapsed, requests=count['requests'], pb_log=[list(x) for x in getattr(mod, '_PB_LOG', [])],
                 kernel_s=m1['kernel_s'] - m0['kernel_s'], kernel_cycles=m1['kernel_cycles'] - m0['kernel_cycles'],
                 server_cycles=m1['cycles'] - m0['cycles'], server_instructions=m1['instructions'] - m0['instructions'],
                 client=rusage.delta(rusage.read(), own0), handles=m1['peak_handles'],
                 puts=m1['puts'] - m0['puts'], put_bytes=stored(m1, m0),
                 comms=[a - b for a, b in zip(m1['comms'], m0['comms'])],
                 comms_work=[a - b for a, b in zip(m1['comms_work'], m0['comms_work'])],
                 slowdown=SLOWDOWN, **setup_clock.record(mod, log, clock, start, r0))
if setup['defined']:
    setup['residual_instructions'] = setup['client']['instructions'] - (DISPATCH[1] - setup_dispatch0[1])
    entries = [(kind,key,entry) for kind,cache in enumerate((mod._IPT_A,mod._IPT_B,mod._IPT_C)) for key,entry in cache.items() if (kind,)+key in mod._PB_SINGLE]
    setup['actual_coverage'] = dict(bound=len(entries), fused=sum(entry[3] is not None for _,_,entry in entries), finished=sum(entry[2][1] is not None for _,_,entry in entries))
    setup['smx_stats'] = dict(getattr(mod, '_SMX_STATS', {}))
    setup['smx_bound'] = mod._SMX_ENV._sm40_owner is mod._SCRATCH[0]
manifest = dict(candidate=name, sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), ranks=ranks, setup=setup,
                client_flopscope=flops.__version__, sites=SITES, short_setup=SLOWDOWN is not None, setup_slowdown=SLOWDOWN,
                profiled=PROFILE)
(out/'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
print(json.dumps(dict(event='setup', **setup)), flush=True)

with (out/'rows.jsonl').open('x') as log:
    for rank in ranks:
        fixture = STORE/Path(rows[rank]['path']).parent/'weights.npy'
        count.update(requests=0, free_trips=0, freed=0)
        by_func.clear()
        by_site.clear()
        m0 = mark()
        with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f'release-{rank}') as ctx:
            blob = fnp.load(str(fixture))
            weights = [blob[layer] for layer in range(16)]
            cpu0, own0, d0, start = rusage.read(SERVER_PID), rusage.read(), list(DISPATCH), time.monotonic()
            sm24_before = dict(getattr(mod, '_SM24_STATS', {}).get('shapes', {}))
            sm12_before = getattr(mod, '_SM12_STATS', {}).get('routed', 0)
            peel_before = {k: getattr(mod, '_PEEL_STATS', {}).get(k, 0) for k in ('push', 'read')}
            w3_before = getattr(mod, '_W3R_STATS', {}).get('calls', 0)
            if PROFILE:
                profile = cProfile.Profile()
                profile.enable()
            prediction = est.predict(MLP(weights, rank), 2**41)
            if PROFILE:
                profile.disable()
                profile.dump_stats(str(out/f'profile_{rank}.pstats'))
            data, shape, dtype = prediction._fetch_data()
            wall = time.monotonic() - start
            server, own = rusage.delta(rusage.read(SERVER_PID), cpu0), rusage.delta(rusage.read(), own0)
            residual_cycles = own['cycles'] - (DISPATCH[0] - d0[0])
            residual_instructions = own['instructions'] - (DISPATCH[1] - d0[1])
            seen = dict(count)
            assert tuple(shape) == (16, 1024), shape
            decoded = array.array('f' if '32' in str(dtype) else 'd')
            decoded.frombytes(data)
            nonfinite = sum(1 for x in decoded if not math.isfinite(x))
            del blob, weights, prediction   # harness-owned arrays go before the session closes
            flops_used = int(ctx.summary_dict()['flops_used'])
        s = ctx.summary_dict()
        m1 = mark()
        kernel_ops = {k: [a - b for a, b in zip(v, m0['ops'].get(k, [0, 0, 0, 0]))] for k, v in m1['ops'].items()}
        numpy_ops = {k: [a - b for a, b in zip(v, m0.get('numpy_ops', {}).get(k, [0, 0, 0, 0]))]
                     for k, v in m1.get('numpy_ops', {}).items()}
        row = dict(rank=rank, flops=flops_used, util=flops_used/2**41, wall_s=wall,
                   backend_s=s['flopscope_backend_time_s'], overhead_s=s['flopscope_overhead_time_s'],
                   residual_s=s['residual_wall_time_s'], context_wall_s=s['wall_time_s'],
                   residual_cycles=residual_cycles, residual_instructions=residual_instructions,
                   requests=seen['requests'], free_trips=seen['free_trips'], freed_handles=seen['freed'],
                   server=server, client=own, peak_handles=m1['peak_handles'], server_footprint_max=m1['footprint_max'],
                   kernel_calls=m1['kernel_calls'] - m0['kernel_calls'], kernel_s=m1['kernel_s'] - m0['kernel_s'],
                   kernel_cycles=m1['kernel_cycles'] - m0['kernel_cycles'],
                   kernel_instructions=m1['kernel_instructions'] - m0['kernel_instructions'],
                   comms=[a - b for a, b in zip(m1['comms'], m0['comms'])],            # requests, ns, bytes in, bytes out
                   comms_work=[a - b for a, b in zip(m1['comms_work'], m0['comms_work'])],   # calls, cycles, instructions
                   operations={k: [v['calls'], v['flop_cost']] for k, v in s['operations'].items()},
                   puts=m1['puts'] - m0['puts'], put_bytes=stored(m1, m0),
                   kernel_ops={k: v for k, v in kernel_ops.items() if v[0]},
                   numpy_ops={k: v for k, v in numpy_ops.items() if v[0]},
                   server_minor_faults=m1['minor_faults'] - m0['minor_faults'],
                   n_out=len(decoded), nonfinite=nonfinite, dtype=str(dtype),
                   index_cache_entries=[len(getattr(mod, k)) for k in ('_IPT_A', '_IPT_B', '_IPT_C') if hasattr(mod, k)])
        row.update(smx_stats=dict(getattr(mod, '_SMX_STATS', {})),smallsm_stats=dict(getattr(mod, '_SMALLSM_STATS', {})),pushsm_stats=dict(getattr(mod, '_PUSHSM_STATS', {})),latebox_stats=dict(getattr(mod, '_LB_STATS', {})),smx_cache_counts={k:len(getattr(mod._SMX_ENV,k,{}))for k in ('_sm40_frames','_sm40_cache','_sm40_io_cache','_sm40_outer_cache','_sm40_group_cache')})
        row.update(peel_stats=dict(getattr(mod, '_PEEL_STATS', {})), peel_delta={k: getattr(mod, '_PEEL_STATS', {}).get(k, 0)-v for k,v in peel_before.items()}, peel_budget=dict(getattr(mod, '_PEEL_CALL_BUDGET', {})), w3_stats=dict(getattr(mod, '_W3R_STATS', {})), w3_delta=getattr(mod, '_W3R_STATS', {}).get('calls', 0)-w3_before)
        row.update(sm12_stats=dict(getattr(mod, '_SM12_STATS', {})), sm12_delta=getattr(mod, '_SM12_STATS', {}).get('routed', 0)-sm12_before)
        row.update(sm24_shapes={k:v-sm24_before.get(k,0) for k,v in getattr(mod, '_SM24_STATS', {}).get('shapes',{}).items() if v!=sm24_before.get(k,0)})
        if SITES:
            row.update(by_func=dict(by_func.most_common()), by_site=dict(by_site.most_common()))
        (out/f'pred_{rank}.bin').write_bytes(data)
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(json.dumps({k: v for k, v in row.items()
                          if k not in ('by_func', 'by_site', 'kernel_ops', 'numpy_ops', 'operations', 'put_bytes')}), flush=True)
        assert nonfinite == 0 and len(decoded) == 16384, (rank, nonfinite, len(decoded))
        del decoded, data
print('ALL_NETWORKS_COMPLETE', flush=True)
