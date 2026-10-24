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
sys.path.insert(0, '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3')
from thread_counts_v3 import raw as thread_raw, assert_owner as thread_assert_owner, selfcheck as thread_selfcheck
THREAD_COUNTER = thread_selfcheck()

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
from sync_observer_v1 import SyncObserver
sync_observer = SyncObserver(name)
sync_observer.install()
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
                          _release_span=_dispatch._counted_span,
                          _release_thread_raw=thread_raw, _release_thread_acc=[0, 0], _release_span_count=[0])
exec('''
@contextmanager
def _counted_span():
    with _release_span():
        top = _release_depth[0] == 0
        if top:
            tc0, ti0 = _release_thread_raw()
            c0, i0 = _release_raw()
            _release_span_count[0] += 1
        _release_depth[0] += 1
        try:
            yield
        finally:
            _release_depth[0] -= 1
            if top:
                c1, i1 = _release_raw()
                tc1, ti1 = _release_thread_raw()
                _release_thread_acc[0] += tc1 - tc0
                _release_thread_acc[1] += ti1 - ti0
                _release_acc[0] += c1 - c0
                _release_acc[1] += i1 - i0
''', _dispatch.__dict__)
DISPATCH = _dispatch._release_acc      # [cycles, instructions] spent inside dispatch spans so far
THREAD_DISPATCH = _dispatch._release_thread_acc
DISPATCH_SPANS = _dispatch._release_span_count

stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
sync_observer.phase('candidate_import')
t_import = time.monotonic()
mod = types.ModuleType('release_candidate')
mod.__file__ = name
sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), name, 'exec'), mod.__dict__)
import_s = time.monotonic() - t_import
sync_observer.phase('harness')
SM_SHAPES=collections.Counter();OLD_MM_SHAPES=collections.Counter()
# Harness-only facade lifecycle probe. No wrapper is added to predict operations.
import weakref
import gc
_ash_refs = []
_ash_init = mod._SetupAliasHold.__init__
def _ash_probe_init(self, namespace):
    _ash_init(self, namespace)
    _ash_refs.append(weakref.ref(self))
mod._SetupAliasHold.__init__ = _ash_probe_init
def ash_lifetime():
    alive = [ref() for ref in _ash_refs if ref() is not None]
    return dict(created=len(_ash_refs), live=len(alive),
                calls=[x.calls for x in alive], held=[len(x.held) for x in alive],
                setup_stats=dict(mod._ASH_STATS),
                flush_shape=list(mod._ASH_FLUSH[0].shape) if mod._ASH_FLUSH[0] is not None else None)


# Harness diagnostics only: observe natural GC, never schedule or suppress it.
_GC_PHASE = ['harness']
_GC_EVENTS = []
_GC_BEGIN = {}
_GC_LOG = (out/'gc_events.jsonl').open('x')
def _observe_gc(phase, info):
    generation = info['generation']
    if phase == 'start':
        _GC_BEGIN[generation] = (time.monotonic_ns(), _dispatch._release_depth[0], _GC_PHASE[0], count['requests'], rusage.raw())
    else:
        started = _GC_BEGIN.pop(generation, None)
        if started is not None:
            event = dict(generation=generation, duration_s=(time.monotonic_ns()-started[0])/1e9,
                         dispatch_depth=started[1], phase=started[2], request=started[3],
                         collected=info['collected'], uncollectable=info['uncollectable'], cpu_delta=[a-b for a,b in zip(rusage.raw(), started[4])])
            _GC_EVENTS.append(event)
gc.callbacks.append(_observe_gc)

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
    _GC_PHASE[0] = 'setup'
    sync_observer.phase('setup')
    est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None,
                                    submission_dir=None, seed=0))
    elapsed = time.monotonic() - start
    sync_observer.phase('harness')
    sync_observer.write(out/'sync_diagnostic.json')
    _GC_PHASE[0] = 'harness'
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
    setup['smx_stats'] = dict(getattr(mod,'_SMX_STATS',{}))
    setup['compile_stats'] = dict(getattr(mod,'_SM40_COMPILE_STATS',{}))
    setup['mrp_stats'] = dict(getattr(mod,'_MRP_STATS',{}))
    setup['finish160_stats'] = dict(getattr(mod,'_SETUP_FINISH160_STATS',{}))
    setup['smx_bound'] = getattr(getattr(mod,'_SMX_ENV',None),'_sm40_owner',None) is mod._SCRATCH[0]

def cache_snapshot():
    result = {}
    for label, env in (("peel",getattr(mod,"_SMX_ENV",None)),("read",getattr(mod,"_ASTRA_SM40_ENV",None))):
        for attr in ('_sm40_frames','_sm40_cache','_sm40_io_cache','_sm40_outer_cache','_sm40_group_cache','_sm40_padding_grid','_sm40_padding_cache'):
            result[label+'.'+attr] = set(map(repr,getattr(env,attr,{})))
        for family, (_,grid,dest) in getattr(env,'_smx_padding',{}).items():
            result[label+'.padding_grid.'+family]=set(map(repr,grid))
            result[label+'.padding_dest.'+family]=set(map(repr,dest))
    for attr in ('_IPT_A','_IPT_B','_IPT_C'):
        cache=getattr(mod,attr,{})
        result[attr]=set(map(repr,cache))
        result[attr+'.fused']={repr(k) for k,v in cache.items() if v[3] is not None}
        result[attr+'.finished']={repr(k) for k,v in cache.items() if v[2][1] is not None}
    return result

def cache_delta(before,after):
    return {k:dict(before=len(before.get(k,())),after=len(after.get(k,())),added=sorted(after.get(k,set())-before.get(k,set())),removed=sorted(before.get(k,set())-after.get(k,set()))) for k in before.keys()|after.keys()}

setup['cache_counts']={k:len(v) for k,v in cache_snapshot().items()}
setup['alias_hold_lifetime']=ash_lifetime()
setup['gc_state']=dict(frozen_objects=gc.get_freeze_count(), enabled=gc.isenabled(), threshold=gc.get_threshold(), count=gc.get_count(), stats=gc.get_stats(), source_stats=dict(getattr(mod, '_GCF_STATS', {})))
assert not any(setup['alias_hold_lifetime']['held']), setup['alias_hold_lifetime']
manifest = dict(candidate=name, sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), ranks=ranks, setup=setup,
                client_flopscope=flops.__version__, python=sys.version, sites=SITES, short_setup=SLOWDOWN is not None, setup_slowdown=SLOWDOWN,
                profiled=PROFILE, dual_counter=THREAD_COUNTER, sync_diagnostic=True,
                diagnostic_harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                thread_helper_sha256=hashlib.sha256(Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/thread_counts_v3.py').read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
print(json.dumps(dict(event='setup', **setup)), flush=True)

with (out/'rows.jsonl').open('x') as log:
    for rank in ranks:
        _GC_PHASE[0] = 'fixture-' + str(rank)
        fixture = STORE/Path(rows[rank]['path']).parent/'weights.npy'
        count.update(requests=0, free_trips=0, freed=0)
        by_func.clear()
        by_site.clear()
        caches_before=cache_snapshot()
        SM_SHAPES.clear();OLD_MM_SHAPES.clear()
        m0 = mark()
        with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f'release-{rank}') as ctx:
            blob = fnp.load(str(fixture))
            weights = [blob[layer] for layer in range(16)]
            thread_assert_owner()
            thread0 = thread_raw()
            cpu0, own0, d0, start = rusage.read(SERVER_PID), rusage.read(), list(DISPATCH), time.monotonic()
            thread_dispatch0, dispatch_spans0 = tuple(THREAD_DISPATCH), DISPATCH_SPANS[0]
            if PROFILE:
                profile = cProfile.Profile()
                profile.enable()
            gc_event_start = len(_GC_EVENTS)
            gc_count_before = gc.get_count()
            gc_stats_before = gc.get_stats()
            _GC_PHASE[0] = 'predict-' + str(rank)
            sync_observer.phase('predict', rank)
            prediction = est.predict(MLP(weights, rank), 2**41)
            sync_observer.phase('final_fetch', rank)
            if PROFILE:
                profile.disable()
                profile.dump_stats(str(out/f'profile_{rank}.pstats'))
            data, shape, dtype = prediction._fetch_data()
            wall = time.monotonic() - start
            sync_observer.phase('harness')
            _GC_PHASE[0] = 'harness-' + str(rank)
            gc_count_after = gc.get_count()
            gc_stats_after = gc.get_stats()
            gc_events_predict = _GC_EVENTS[gc_event_start:]
            server, own = rusage.delta(rusage.read(SERVER_PID), cpu0), rusage.delta(rusage.read(), own0)
            thread1 = thread_raw()
            dispatch_spans = DISPATCH_SPANS[0]-dispatch_spans0
            thread_total = [a-b for a,b in zip(thread1, thread0)]
            thread_dispatch = [a-b for a,b in zip(THREAD_DISPATCH, thread_dispatch0)]
            thread_residual = [a-b for a,b in zip(thread_total, thread_dispatch)]
            thread_assert_owner()
            assert all(v >= 0 for v in thread_total + thread_dispatch + thread_residual), (thread_total, thread_dispatch, thread_residual)
            residual_cycles = own['cycles'] - (DISPATCH[0] - d0[0])
            residual_instructions = own['instructions'] - (DISPATCH[1] - d0[1])
            seen = dict(count)
            assert tuple(shape) == (16, 1024), shape
            decoded = array.array('f' if '32' in str(dtype) else 'd')
            decoded.frombytes(data)
            nonfinite = sum(1 for x in decoded if not math.isfinite(x))
            del blob, weights, prediction   # harness-owned arrays go before the session closes
            flops_used = int(ctx.summary_dict()['flops_used'])
        caches_after=cache_snapshot()
        s = ctx.summary_dict()
        m1 = mark()
        kernel_ops = {k: [a - b for a, b in zip(v, m0['ops'].get(k, [0, 0, 0, 0]))] for k, v in m1['ops'].items()}
        numpy_ops = {k: [a - b for a, b in zip(v, m0.get('numpy_ops', {}).get(k, [0, 0, 0, 0]))]
                     for k, v in m1.get('numpy_ops', {}).items()}
        row = dict(rank=rank, flops=flops_used, util=flops_used/2**41, wall_s=wall,
                   backend_s=s['flopscope_backend_time_s'], overhead_s=s['flopscope_overhead_time_s'],
                   residual_s=s['residual_wall_time_s'], context_wall_s=s['wall_time_s'],
                   residual_cycles=residual_cycles, residual_instructions=residual_instructions,
                   top_level_dispatch_spans=dispatch_spans,
                   thread_total_cycles=thread_total[0], thread_total_instructions=thread_total[1],
                   thread_dispatch_cycles=thread_dispatch[0], thread_dispatch_instructions=thread_dispatch[1],
                   thread_residual_cycles=thread_residual[0], thread_residual_instructions=thread_residual[1],
                   process_minus_thread_residual_instructions=residual_instructions-thread_residual[1],
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
        row.update(smx_stats=dict(getattr(mod,'_SMX_STATS',{})),peel_stats=dict(getattr(mod,'_PEEL_STATS',{})),w3_stats=dict(getattr(mod,'_W3R_STATS',{})),cap_stats=dict(getattr(mod,'_PEEL_CALL_BUDGET',{})),cache_delta=cache_delta(caches_before,caches_after))
        row.update(sm12_stats=dict(getattr(mod,'_SM12_STATS',{})),rqs_stats=dict(getattr(mod,'_RQS_STATS',{})),rqx2_stats=dict(getattr(mod,'_RQX2_STATS',{})),ot_stats=dict(getattr(mod,'_OT_STATS',{})),owned_quad_cache_entries=len(getattr(mod,'_OT_QV',{})))
        row['smallsm_shapes']=[list(k)+[v]for k,v in sorted(SM_SHAPES.items())]
        row['oldmm_eligible_shapes']=[list(k)+[v]for k,v in sorted(OLD_MM_SHAPES.items())]
        row['owned_pool_stats']=dict(getattr(mod,'_OWP_STATS',{}))
        pool=getattr(mod,'_OWP',{})
        row['owned_pool_retained_bytes']=pool.get('bytes',0)
        row['lazybind_stats']=dict(getattr(mod,'_LB_STATS',{}))
        row['compute_stats']=dict(getattr(mod,'_ASTRA_COMPUTE_STATS',{}))
        row['post_first_stats'] = dict(getattr(mod,'_MR_POST_STATS',{}))
        row['compile_stats'] = dict(getattr(mod,'_SM40_COMPILE_STATS',{}))
        row['mrp_stats'] = dict(getattr(mod,'_MRP_STATS',{}))
        row['finish160_stats'] = dict(getattr(mod,'_SETUP_FINISH160_STATS',{}))
        row['natural_gc'] = dict(count_before=gc_count_before, count_after=gc_count_after, stats_before=gc_stats_before, stats_after=gc_stats_after, events=gc_events_predict, frozen_objects=gc.get_freeze_count(), enabled=gc.isenabled())
        row['hardware_repair_stats']=dict(getattr(mod,'_HW_REPAIR_STATS_V1',{}))
        row['mainread_setup_stats']=dict(getattr(mod,'_MRS_STATS',{}))
        row['mainread_stats']=dict(getattr(mod,'_ASTRA_SM40_STATS',{}))
        row['fast_rescue_stats'] = dict(getattr(mod, '_FAST_RESCUE_STATS', {}))
        row['fast_rescue_control'] = dict(getattr(mod, '_FAST_RESCUE_CONTROL', {}))
        row['fast_rescue_final_state'] = dict(on=getattr(mod, '_FAST_RESCUE_ON', None), layer=getattr(mod, '_FAST_RESCUE_LAYER', None))
        row['alias_hold_lifetime']=ash_lifetime()
        assert not any(row['alias_hold_lifetime']['held']), row['alias_hold_lifetime']
        if SITES:
            row.update(by_func=dict(by_func.most_common()), by_site=dict(by_site.most_common()))
        sync_observer.write(out/'sync_diagnostic.json')
        row['sync_diagnostic'] = True
        (out/f'pred_{rank}.bin').write_bytes(data)
        for gc_event in gc_events_predict:
            _GC_LOG.write(json.dumps(gc_event)+'\n')
        _GC_LOG.flush()
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(json.dumps({k: v for k, v in row.items()
                          if k not in ('by_func', 'by_site', 'kernel_ops', 'numpy_ops', 'operations', 'put_bytes')}), flush=True)
        assert nonfinite == 0 and len(decoded) == 16384, (rank, nonfinite, len(decoded))
        del decoded, data
# Separately labeled stress only, after every measured prediction is complete.
# No automatic-GC policy change, and no score/qualification timing is rewritten.
_GC_PHASE[0] = 'end_stress'
_stress_before = rusage.read()
_stress_start = time.monotonic()
_stress_collected = gc.collect()
_stress_wall = time.monotonic() - _stress_start
_stress_cpu = rusage.delta(rusage.read(), _stress_before)
_stress = dict(label='diagnostic_full_generation_scan_after_sequence', wall_s=_stress_wall,
               client=_stress_cpu, collected=_stress_collected, frozen_objects=gc.get_freeze_count(),
               automatic_gc_enabled=gc.isenabled(), threshold=list(gc.get_threshold()),
               gc_stats=gc.get_stats(), gc_count=gc.get_count(),
               setup_freeze_stats=dict(getattr(mod, '_GCF_STATS', {})),
               excluded_from_prediction_metrics=True)
(out/'end_gc_stress.json').write_text(json.dumps(_stress,indent=1)+'\n')
print(json.dumps(dict(event='end_gc_stress', **_stress)), flush=True)
_GC_LOG.flush()
sync_observer.write(out/'sync_diagnostic.json')
sync_observer.uninstall()
print('ALL_NETWORKS_COMPLETE', flush=True)
