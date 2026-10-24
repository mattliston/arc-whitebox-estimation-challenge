"""No model/SDK runtime/NumPy: allocation, scope, ownership and span checks."""
import ast
from array import array
import contextlib
import dis
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OBSERVER = HERE.parents[1] / 'astra_oct20_bill_v1/profile_outside_dispatch_v2.py'
SDK_SOURCE = HERE.parent / 'sync_exact4685_v1/sdk_snapshot_v1/_dispatch.py'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_observer():
    spec = importlib.util.spec_from_file_location('metadata_observer_v2', OBSERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_arm(mode, cycles):
    module = load_observer()
    sdk = dict(__name__='flopscope._dispatch', contextmanager=contextlib.contextmanager,
        _now_ns=time.perf_counter_ns, _caller_is_internal=lambda **_: True,
        _count_participant_span=lambda: None, _total_dispatch_ns=0)
    node = next(node for node in ast.parse(SDK_SOURCE.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == '_counted_span')
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'isolated_official_span_AST', 'exec'), sdk)
    filename = 'candidate_metadata_fixture_v2_only'
    queue = array('q', [0, 0])
    batches = array('q', [0]) * 20000
    position = array('q', [0])
    collections = array('q', [0, 0, 0])
    env = dict(queue=queue, cycles=cycles)

    def drain():
        batches[position[0]] = queue[0]
        position[0] += 1
        queue[0] = 0

    def sdk_step():
        with sdk['_counted_span']():
            env['inside_sdk']()
            with sdk['_counted_span']():
                env['inside_sdk']()
            drain()

    def gc_event(phase, info):
        if phase == 'stop':
            collections[info['generation']] += 1

    env['sdk_step'] = sdk_step
    code = '''class Token:
    def __init__(self, i):
        self.i = i
        if cycles and i % 37 == 0:
            self.peer = self
    def __del__(self):
        queue[0] += 1
        queue[1] += 1
class Helpers:
    @staticmethod
    def number(x):
        return abs(x) + 1
    @classmethod
    def twice(cls, x):
        return 2 * cls.number(x)
    @property
    def value(self):
        return 3
def inside_sdk():
    return sum(range(5))
def child(i):
    token = Token(i)
    value = Helpers.twice(i)
    return value
def make_closure():
    def inner(x):
        return x + 1
    def outer(x):
        return inner(x)
    return outer
closure = make_closure()
def generator():
    yield 2
    yield 3
def raises():
    raise ValueError('fixture exception')
def parent(n):
    total = 0
    for i in range(n):
        total += child(i)
        sdk_step()
    total += closure(3) + Helpers().value + sum(generator())
    try:
        raises()
    except ValueError:
        total += 1
    return total
'''
    exec(compile(code, filename, 'exec'), env)
    profile = module.OutsideDispatchProfile(filename, env,
        lambda: sdk['_total_dispatch_ns'], sdk['_counted_span'].__wrapped__.__code__)
    # Both arms construct the identical observer before the fixture. This
    # explicit fixture initialization does not alter GC thresholds/policy.
    thresholds = gc.get_threshold()
    gc.callbacks.append(gc_event)
    gc.collect()
    collections[0] = collections[1] = collections[2] = 0
    before = gc.get_count()
    if mode == 'observe':
        profile.start()
    answer = env['parent'](10000)
    if mode == 'observe':
        profile.stop()
    after = gc.get_count()
    gc.callbacks.remove(gc_event)
    assert gc.get_threshold() == thresholds and gc.isenabled()
    assert sys.getprofile() is None and profile.depth == profile.sdk_depth == 0
    report = profile.report()
    if mode == 'observe':
        assert report['root_dispatch_ns'] == sdk['_total_dispatch_ns'] > 0
        assert report['excluded_candidate_events_in_sdk'] == 40000
        assert not any(row['function'] == 'inside_sdk' for row in report['rows'])
        assert sum(row['outside_exclusive_ns'] for row in report['rows']) == report['root_outside_ns']
        assert report['events']['call'] == report['events']['return_event']
        assert {row['function'] for row in report['rows']} >= {'inner','outer','number','twice','value','generator','raises','__del__'}
    else:
        assert not report['started'] and not report['rows'] and report['root_outside_ns'] == 0
    return dict(mode=mode, cycles=cycles, answer=answer,
        gc_counts_delta=[after[i]-before[i] for i in range(3)],
        gc_collections=list(collections), finalizers=int(queue[1]), pending=int(queue[0]),
        drains=int(position[0]), free_batch_sha256=hashlib.sha256(batches.tobytes()).hexdigest(),
        profile_calls=profile.calls, registered_codes=len(profile.codes),
        max_depth=profile.max_depth, max_sdk_depth=profile.max_sdk_depth,
        threshold_unchanged=True, profile_restored=True, exact_dispatch_subtraction=True)


def main():
    module = load_observer()
    forbidden = {'BUILD_LIST','BUILD_TUPLE','BUILD_MAP','BUILD_SET','BUILD_CONST_KEY_MAP',
                 'LIST_APPEND','MAP_ADD','SET_ADD','LIST_EXTEND','SET_UPDATE','DICT_UPDATE','DICT_MERGE'}
    instructions = list(dis.get_instructions(module.OutsideDispatchProfile.__call__))
    assert not [item.opname for item in instructions if item.opname in forbidden]
    arms = []
    for cycles in (False, True):
        pair = []
        for mode in ('control', 'observe'):
            raw = subprocess.check_output([sys.executable, '-B', str(__file__), '--arm', mode,
                                           '1' if cycles else '0'], text=True)
            pair.append(json.loads(raw))
        for key in ('answer','gc_counts_delta','gc_collections','finalizers','pending','drains',
                    'free_batch_sha256','registered_codes','threshold_unchanged','profile_restored'):
            assert pair[0][key] == pair[1][key], (cycles, key, pair)
        arms.extend(pair)
    # The actual adapter persists a list-owned observer populated by evaluate.
    # Verify the formerly broken local-vs-module scope with no model/SDK work.
    scope = {'PROFILE_STATE': [None]}
    exec('def evaluate():\n    outside_profile = object()\n    PROFILE_STATE[0] = outside_profile\n    return 7\nanswer=evaluate()\nassert PROFILE_STATE[0] is not None\n', scope)
    report = dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
        successful=True, python=sys.version, executable=sys.executable, metadata_only=True,
        model_count=0, sdk_runtime_imported=False, observer_sha256=sha(OBSERVER),
        checker_sha256=sha(__file__), official_span_ast_sha256=sha(SDK_SOURCE),
        no_per_event_tracked_container_build_opcodes=True, scope_check_pass=True,
        same_preallocated_state_both_arms=True, same_gc_counts_and_collection_schedule=True,
        exact_finalizer_and_free_batch_sequence=True, arms=arms,
        limitation='Synthetic object-lifetime parity is necessary but not a full SDK/model guarantee. Preserve all exact-control native gates; no timing gain is inferred.')
    out = HERE / ('METADATA_CHECK_py%d%d_v1.json' % sys.version_info[:2])
    with out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(json.dumps(one_arm(sys.argv[2], sys.argv[3] == '1')))
    else:
        main()
