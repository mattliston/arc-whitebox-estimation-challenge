"""Fixed partial-D6/P1 native40 endpoint; inert without an explicit --run.

Full lexical build admission runs in a process retired before the numerical
worker. The worker retains the original official-input/paid/reset loop and
observes only existing small bytes, identities, shapes and public counters.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW/'birth_direction_fit_v1'
TL = NEW/'current_d50_terminal_linear_v2'
CP = NEW/'current_p1_node_composition_v1'
P1 = NEW/'current_d50_native_objective_p1_v1'
RC = NEW/'current_d50_response10_candidate_v1/native.py'
OFFPY = P2/'data_gate/.venv/bin/python'
HELPER = TL/'watch_fit.py'
HELPER_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
CENSUS = NEW/'current_d50_response10_census_v1/watch.py'
CENSUS_SHA = '7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf'
CP_SHA = 'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01'
WATCH_PARENT_SHA = '8ffc9ae68388065d49c9e39286893f5da8f5eeabe000db625643fd340442cd28'
RC_SHA = 'cccd8b50d18e6d093a48a3e3675aa20cbe2ef31cb45c2f167f153ddc842fe09c'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
OBSERVER_SHA = '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe'
COMMON_SHA = '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a'
PROTOCOL_SHA = 'b4fc1edd8cabec7b59d11ede87de42ce3f3c142aa0b142582bcccac8bdcdaff9'
BUILDER_SHA = '850de8855cfd50a0ff98e5bce072d401aafd0e3d87282c79b17256dfa1397f08'
BUILD_SHA = '43773c4204d869732b40ca345115e272782fe1d17b64da343bcaf98137b0fab4'
BUILD_END_SHA = '6f259844d30eeebd4114d2751d6a0adc8f0888662b86c335da47053d6c728664'
CANDIDATE_SHA = 'eaeaac6ebe75aa272b712b4dfed85f220e876a4236227e77e47933e0d363f1d0'
CANDIDATES = {'terminal':CANDIDATE_SHA}
P1_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
REFERENCE_SHA = '7a1f37d80a7ed32850b10047408d85eed050220b4a6271dde90bdd21c07062e3'
FAIL_SHA = '86ce7e8b1ea3d9dcc81c42bad6cb78828b520f38aed5f4918acf6052a6cf8660'
PROBE_SHA = 'bf79b160f8952db1f433093c245c5497d2d25a623db6888cafa13f0ae48aaeb2'
MONITOR_DIR = NEW/'current_p1_native_objective_p2_monitor_v2'
MONITOR_SHA = 'd4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873'
MONITOR_TEST_SHA = '1438ca2f4b12b53628cfbb2be4f997eea8949bdf5a70a8c44c74738e83529fb9'
MONITOR_TEST_RESULT_SHA = 'c5a3f5b8a3dc99ca7e3550fb9ccb474b76968a49b96e00ed7e5606597476bf0e'
MONITOR_TEST_END_SHA = 'bfb69013c603f0578a56c5e268e3654221b1ab50e00a8bd10159ef074b9e846f'
INVENTORY_SHA = '4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
ADMISSION, OUTPUT, WATCH_OUT = HERE/'admission_v1',HERE/'native_v1',HERE/'native_watch_v1'
CONTEXT_SCHEMA = 'current_p1_partial_d6_context_v1'
ADMIT_SCHEMA = 'current_p1_partial_d6_admission_v1'
NATIVE_SCHEMA = 'current_p1_partial_d6_native_v1'
WATCH_SCHEMA = 'current_p1_partial_d6_native_watch_v1'
RETIRED_SCHEMA = 'current_p1_partial_d6_admission_watch_v1'
BUDGET, RSS_LIMIT, CONTEXT_LIMIT = 2**41,8_000_000_000,2*1024**2
EXTRA_BINDINGS, PRODUCT_FLOPS = 561056000,7062260240
SHAPE = (1024,832,8192)
EXTRA_LAYOUTS = {'_P1_D6_832_PUSH_SPECS':278566656,'_P1_D6_832_READ_SPECS':282489344}
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
_BASE = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'Target arrays forbidden')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
            if tick: tick()
    return digest.hexdigest()


def read(path, bound=None):
    path = Path(path)
    require(not path.is_symlink() and (bound is None or path.stat().st_size <= bound), 'Bounded regular metadata')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def bind(items, tick=None):
    require(isinstance(items,dict), 'Flat immutable hash map')
    for path,digest in items.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest or '')
                and sha(path,tick) == digest, 'Immutable dependency: '+str(path))


def load(path, name, digest):
    require(sha(path) == digest, 'Reviewed source owner')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def base():
    global _BASE
    if _BASE is None:
        _BASE = load(CP/'native.py','partial_d6_frozen_native_metadata',CP_SHA)
    return _BASE


def old():
    # Compatibility with the source-inverse retired-process watchdog only.
    return sys.modules[__name__]


def environment():
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.flags.optimize == 0
            and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'OFFPY/nice10/BLAS1/assertions/no fallback')
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope')), 'Pre-numerical stdlib admission')


def arguments(role):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256',dest='watcher_sha256' if role == 'watcher' else 'worker_sha256')
    parser.add_argument('--worker-sha256' if role == 'watcher' else '--watcher-sha256')
    for name in ('builder','build','candidate','protocol','context','admission-watch'):
        parser.add_argument('--'+name+'-sha256')
    if role == 'watcher':
        parser.add_argument('--prior-verdict-reviewed',action='store_true')
        parser.add_argument('--numerical-slot-clear',action='store_true')
    else:
        parser.add_argument('--stage',choices=('admit','worker'))
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return None
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in
                ('watcher','worker','builder','build','candidate','protocol')), 'Explicit reviewed source/build hashes')
    require((args.builder_sha256,args.build_sha256,args.candidate_sha256,args.protocol_sha256) ==
            (BUILDER_SHA,BUILD_SHA,CANDIDATE_SHA,PROTOCOL_SHA), 'Only the frozen exact-shape candidate')
    worker = role != 'watcher' and args.stage == 'worker'
    require(role == 'watcher' or args.stage in ('admit','worker'), 'Explicit nonautomatic stage')
    require((args.context_sha256 is not None) is worker and (args.admission_watch_sha256 is not None) is worker,
            'Only the numerical worker receives sealed retired-admission bindings')
    if worker:
        require(all(re.fullmatch('[0-9a-f]{64}',x) for x in (args.context_sha256,args.admission_watch_sha256)),
                'Parent-supplied context and retirement digests')
    return args


def owners(args, tick=None):
    result = {str(HERE/'native.py'):args.worker_sha256,str(HERE/'watch_native.py'):args.watcher_sha256,
        str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(HERE/'build_v2.py'):BUILDER_SHA,
        str(HERE/'SOURCE_BUILD_V2_RESULT.md'):'01f358ce200a5fcac9cfab2b5cf044f6f78a3a97b012da982f0c1c6bdb16c3a2',
        str(CP/'native.py'):CP_SHA,str(CP/'watch_native_v2.py'):WATCH_PARENT_SHA,
        str(RC):RC_SHA,str(BD/'native_v2.py'):NATIVE_SHA,str(TL/'native.py'):OBSERVER_SHA,
        str(TL/'common.py'):COMMON_SHA,str(HELPER):HELPER_SHA,str(CENSUS):CENSUS_SHA,
        str(MONITOR_DIR/'monitor.py'):MONITOR_SHA,str(MONITOR_DIR/'monitor_tests.py'):MONITOR_TEST_SHA,
        str(MONITOR_DIR/'PROTOCOL.md'):'968910dc63ba1c090665322c945b4e7db3b2978b1a059baa10bab13ea2dc3af7',
        str(P1/'watch.py'):INVENTORY_SHA}
    bind(result,tick)
    return result


def transport_admission(tick=None):
    """Mock-certified stdlib transport only, not MPS/model/training lineage."""
    folder = MONITOR_DIR/'monitor_tests_v1'
    inputs = {str(folder/'result.json'):MONITOR_TEST_RESULT_SHA,str(folder/'completion.json'):MONITOR_TEST_END_SHA}
    bind(inputs,tick); record,end = read(folder/'result.json'),read(folder/'completion.json')
    require(record['schema'] == 'current_p1_native_objective_p2_monitor_tests_v2'
            and all(record[k] is True for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
            and end['passed'] and end['completed'] and end['result_sha256'] == MONITOR_TEST_RESULT_SHA
            and not any(record[k] for k in ('numerical_imports','arrays_accessed','targets_accessed','model_accessed'))
            and record['real_subprocesses_started'] == record['real_signals_sent'] == 0
            and record['elapsed_s'] < 5 and record['peak_rss_bytes'] < 256_000_000
            and record['checks'] == ['concurrent_delayed_within_actual_1s','timeout_no_retry_and_owned_reap',
                'late_finalization_is_failure','nonzero_command_is_failure','single_snapshot_tree_and_foreign_inventory',
                'malformed_duplicate_process_table_rejected','owned_group_cleanup_and_exit_race',
                'exited_leader_live_group_is_retired'], 'Completed fixed target-free monitor mock certificate')
    require(record['source_bindings'] == {
        str(MONITOR_DIR/'monitor.py'):MONITOR_SHA,str(MONITOR_DIR/'monitor_tests.py'):MONITOR_TEST_SHA,
        str(MONITOR_DIR/'PROTOCOL.md'):'968910dc63ba1c090665322c945b4e7db3b2978b1a059baa10bab13ea2dc3af7',
        str(P1/'watch.py'):INVENTORY_SHA}, 'Exact tested transport/parser/classifier sources')
    bind(record['source_bindings'],tick)
    return inputs


def command(args, role, context_sha=None, admission_watch_sha=None):
    require(role in ('admitter','worker'), 'Only two serial process roles')
    cmd = [str(OFFPY),'-B','-u',str(HERE/'native.py'),'--run','--stage',
           'admit' if role == 'admitter' else 'worker','--source-sha256',args.worker_sha256,
           '--watcher-sha256',args.watcher_sha256]
    for key in ('builder','build','candidate','protocol'):
        cmd += ['--'+key+'-sha256',getattr(args,key+'_sha256')]
    if context_sha: cmd += ['--context-sha256',context_sha]
    if admission_watch_sha: cmd += ['--admission-watch-sha256',admission_watch_sha]
    return cmd


def splice(source, edits):
    changed = source
    for before,after in edits:
        require(changed.count(before) == 1, 'Unique source adapter span')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique inverse source span')
        inverse = inverse.replace(after,before)
    require(inverse == source, 'Full source inverse')
    return changed


def prepare():
    """Small harness sources only; no estimator AST/literal in the worker."""
    n = base(); n.load_diagnostics()
    r = load(RC,'partial_d6_startup_owner',RC_SHA); r.SMALL = n.SMALL
    text = (TL/'native.py').read_text(); tree = ast.parse(text)
    node = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name == 'Observer')
    original = ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    begin = original[original.index('    def __init__'):original.index('    @staticmethod')]
    unused = original[original.index('        elif code is self.gnn_code'):original.index('\n    def export_bytes')]
    export = original[original.index('    def export_bytes'):original.index('\n    def clear')]
    clear = original[original.index('    def clear'):]
    filtered = splice(original,[
        ('class Observer:', 'class DomainObserver:'),
        (begin, "    def __init__(self, module):\n        self.gaussian_code = module.gaussian_raw.__code__\n"
         "        self.select_code = module.select.__code__\n        self.variances, self.selectors = [], []\n\n"),
        (unused,'        # No terminal-head, parameter, or raw-mean capture in this endpoint.\n'),
        (export,"    def export_bytes(self):\n        c.require(len(self.variances) == 16 and len(self.selectors) == 47,\n"
         "                  'Complete unchanged domain/selector packet')\n"
         "        packet = dict(variances=tuple(self.variances),selectors=tuple(self.selectors))\n"
         "        self.clear()\n        return packet\n"),
        (clear,"    def clear(self):\n        self.variances.clear()\n        self.selectors.clear()\n")])
    namespace = dict(vars(n.SMALL)); exec(compile(filtered,str(__file__)+':domain_observer','exec'),namespace)
    Domain = namespace['DomainObserver']
    source,original_sha = n.derived_main()
    proof = dict(native_original_sha256=NATIVE_SHA,native_original_main_sha256=original_sha,
        derived_paid_main_sha256=hashlib.sha256(source.encode()).hexdigest(),paid_loop_inverse_verified=True,
        original_observer_sha256=hashlib.sha256(original.encode()).hexdigest(),
        domain_only_observer_sha256=hashlib.sha256(filtered.encode()).hexdigest(),domain_observer_inverse_verified=True,
        omitted_passive_captures=['terminal_hidden','terminal_head_parameters','all_raw_means'],
        original_TL_domain_diagnostics_unchanged=True,interested_code_count=9)

    class Observer(r.Observer):
        def __init__(self,module):
            super().__init__(module)
            self.names.update({id(getattr(module,k)):k for k in EXTRA_LAYOUTS})
            self.extra_codes = {k:getattr(module,k).__code__ for k in
                ('_p1_d6_832_bind','_p1_d6_832_product','_packed_product_cached')}
            self.dispatch_code = module._packed_product_4.__code__
            self.interested = frozenset(id(x) for x in self.codes.values()) | frozenset(
                id(x) for x in self.extra_codes.values()) | frozenset((id(self.small.gaussian_code),id(self.small.select_code)))
            require(len(self.interested) == 9 and all(code.co_filename == module.__file__ for code in
                    (*self.codes.values(),*self.extra_codes.values(),self.dispatch_code)), 'Exact nine observed code owners')

        def begin(self):
            super().begin()
            self.small = Domain(self.module)
            self.arena_id = None if self.module._SCRATCH is None else id(self.module._SCRATCH[0])
            self.extra_empty_before = self.module._P1_D6_832_BINDINGS is None
            self.extra_regions,self.dispatches = [],[]
            self.extra_pending,self.serial,self.child_pending = None,None,None

        def __call__(self,frame,event,answer):
            if event not in ('call','return') or id(frame.f_code) not in self.interested:
                return
            code = frame.f_code
            if code is self.extra_codes['_p1_d6_832_bind']:
                if event == 'call':
                    require(self.extra_pending is None and self.module._P1_D6_832_BINDINGS is None
                            and frame.f_locals['arena'] is self.module._SCRATCH[0], 'One same-arena extra binding entry')
                    self.extra_pending = dict(start=int(self.ctx.flops_used),ops=len(self.ctx.op_log))
                else:
                    item = self.extra_pending; require(item is not None, 'Matched extra binding return')
                    item.update(flops=int(self.ctx.flops_used)-item.pop('start'),
                                operation_count=len(self.ctx.op_log)-item.pop('ops'))
                    require(item['flops'] == EXTRA_BINDINGS and item['operation_count'] == 68
                            and all(b.arena is self.module._SCRATCH[0] and not b.busy for b in
                                    self.module._P1_D6_832_BINDINGS), 'Both paid serial layouts, same original arena')
                    item.update(same_original_arena=True,nested_binding_flops_not_added_twice=True)
                    self.extra_regions.append(item); self.extra_pending = None
                return
            if code is self.extra_codes['_p1_d6_832_product']:
                if event == 'call':
                    a,b,out,binding = (frame.f_locals[k] for k in ('a','b','out','binding'))
                    require(self.serial is None and frame.f_back.f_code is self.dispatch_code
                            and frame.f_back.f_locals['arena_views'] is self.module._SCRATCH[1]
                            and binding is self.module._P1_D6_832_BINDINGS[0]
                            and binding.arena is self.module._SCRATCH[0] and not binding.busy
                            and (tuple(a.shape),tuple(b.shape),tuple(out.shape)) == ((1024,832),(832,8192),(1024,8192))
                            and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32', 'Only fixed existing push dispatch')
                    self.serial = dict(layer=len(self.small.variances),shape=list(SHAPE),dtype='float32',
                        start=int(self.ctx.flops_used),ops=len(self.ctx.op_log),children=[])
                else:
                    item = self.serial; require(item is not None and self.child_pending is None, 'Matched serial return')
                    item.update(flops=int(self.ctx.flops_used)-item.pop('start'),
                                operation_count=len(self.ctx.op_log)-item.pop('ops'))
                    require(item['flops'] == PRODUCT_FLOPS and len(item['children']) == 7
                            and answer is frame.f_locals['out'] and not frame.f_locals['binding'].busy,
                            'Complete serial tariff, seven direct children, no product copy')
                    item.update(same_original_arena=True,returned_original_out=True,child_bypasses_dispatcher=True)
                    self.dispatches.append(item); self.serial = None
                return
            if code is self.extra_codes['_packed_product_cached']:
                if self.serial is None: return
                if event == 'call':
                    binding = self.module._P1_D6_832_BINDINGS[0]
                    j = len(self.serial['children'])
                    require(self.child_pending is None and frame.f_back.f_code is self.extra_codes['_p1_d6_832_product']
                            and frame.f_back.f_locals['j'] == j and 0 <= j < 7 and frame.f_locals['levels'] == 5
                            and frame.f_locals['views'] is binding.child_views
                            and frame.f_locals['out'] is frame.f_back.f_locals['products'][j]
                            and tuple(frame.f_locals['a'].shape) == (512,416)
                            and tuple(frame.f_locals['b'].shape) == (416,4096), 'Direct unchanged D5 child owner')
                    self.child_pending = dict(child=j,start=int(self.ctx.flops_used),depth=5)
                else:
                    item = self.child_pending; require(item is not None, 'Matched D5 child return')
                    item['flops'] = int(self.ctx.flops_used)-item.pop('start')
                    require(item['flops'] > 0 and answer is frame.f_locals['out'], 'Owned direct-child output and bill')
                    self.serial['children'].append(item); self.child_pending = None
                return
            if code is self.codes['_cp3_bind_arena_views'] and event == 'call':
                arena = frame.f_locals['arena']
                if self.arena_id is None: self.arena_id = id(arena)
                require(id(arena) == self.arena_id and tuple(arena.shape) == (292225024,)
                        and str(arena.dtype) == 'float32', 'Every layout uses the original backing arena')
            super().__call__(frame,event,answer)
            if code is self.codes['_cp3_bind_arena_views'] and event == 'call':
                self.pending['binding'].update(same_original_arena=True,
                    reshape_calls=len(frame.f_locals['view_specs']))

        def finish(self):
            require(self.extra_pending is None and self.serial is None and self.child_pending is None,
                    'No retained partial observer region')
            packet,startup = super().finish()
            startup.update(extra_empty_before=self.extra_empty_before,
                extra_ready_after=self.module._P1_D6_832_BINDINGS is not None,
                extra_binding_regions=list(self.extra_regions),extra_binding_flops=sum(x['flops'] for x in self.extra_regions),
                extra_regions_are_nested_in_bindings=True,original_arena_bytes=1168900096,
                no_second_backing_arena=True,frame_cache_scope='Shape-only original-arena views; no caller matrices retained')
            return packet,startup,list(self.dispatches)

    return Observer,source,proof


def authenticate(args):
    sources = owners(args); n = base()
    builder = load(HERE/'build_v2.py','partial_d6_build_admission',BUILDER_SHA)
    context = builder.verify_build(BUILD_SHA,CANDIDATE_SHA,BUILDER_SHA,PROTOCOL_SHA)
    require(context['input_bindings'][str(HERE/'build_v2/completion.json')] == BUILD_END_SHA,
            'Pinned successful source-build completion')
    sources.update(context['source_bindings'])
    sources.update({v['path']:v['sha256'] for v in context['candidates'].values()})
    inputs = dict(context['input_bindings'])
    inputs.update(transport_admission())
    refs,rpins = n.reference('terminal'); inputs.update(rpins)
    control = read(P1/'native2_terminal_v1/result.json')
    require(control['ranks'] == list(VALID) and len(control['trials']) == len(control['input_preparation']) == 41,
            'P1 exact complete40/repeat control')
    bind(control['source_bindings']); inputs.update(control['source_bindings'])
    legacy = n.legacy()
    sources.update({str(p):v for p,v in legacy.PINS.items()})
    inputs[str(legacy.STORE/'manifest.json')] = n.STORE_SHA
    manifest = read(legacy.STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    for rank in VALID:
        item = next(x for x in manifest if x['role_rank'] == rank)
        path = legacy.STORE/item['path']; require(sha(path) == item['sha256'], 'Canonical row manifest')
        row = read(path); weight = path.parent/'weights.npy'
        require(row['files']['weights.npy']['sha256'] == refs[rank]['weight_file_sha256'], 'Same P1 official inputs')
        inputs.update({str(path):item['sha256'],str(weight):row['files']['weights.npy']['sha256']})
    _observer,_source,proof = prepare()
    bind(sources); bind(inputs); environment()
    return dict(schema=CONTEXT_SCHEMA,role='terminal',ranks=list(VALID),candidate_sha256=CANDIDATE_SHA,
        build_sha256=BUILD_SHA,prerequisites=dict(build_sha256=BUILD_SHA,P1_native_result_sha256=REFERENCE_SHA,
            push_probe_result_sha256=PROBE_SHA),source_bindings=sources,input_bindings=inputs,
        reviewed_owners=owners(args),reference_by_rank=refs,build_proof=context['proof'],proof=proof,
        full_frozen_admission_passed=True,observer_inverse_verified=True,paid_loop_inverse_verified=True,
        estimator_executed=False,numerical_imports=False,truth_accessed=False,
        old_max_read_timing_gate_still_failed=True)


def admission_main(args):
    began = time.monotonic(); environment(); expected = owners(args)
    require(not ADMISSION.exists() and not ADMISSION.is_symlink(), 'Exclusive retired admission')
    ADMISSION.mkdir()
    record = dict(schema=ADMIT_SCHEMA,passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
        reviewed_owners=expected,source_bindings=expected,input_bindings={},artifacts={},
        estimator_executed=False,numerical_imports=False,truth_accessed=False)
    def guard(*unused):
        require(time.monotonic()-began < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
                'Retired stdlib admission60s/2GB')
    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        context = authenticate(args)
        require(len(json.dumps(context,allow_nan=False).encode()) <= CONTEXT_LIMIT, 'No large source/array context')
        bind(context['source_bindings'],guard); bind(context['input_bindings'],guard); guard()
        write(ADMISSION/'context.json',context)
        record.update(source_bindings=context['source_bindings'],input_bindings=context['input_bindings'],
            context_sha256=sha(ADMISSION/'context.json'),full_frozen_admission_passed=True,proof=context['proof'])
        record['artifacts'][str(ADMISSION/'context.json')] = record['context_sha256']
        guard(); record.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        record.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        record.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if record['elapsed_s'] >= 60 or record['peak_rss_bytes'] >= 2_000_000_000:
            record.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False)
        write(ADMISSION/'result.json',record)
        write(ADMISSION/'completion.json',dict(passed=record['passed'],completed=record['completed'],result_sha256=sha(ADMISSION/'result.json')))
    return 0 if record['passed'] else 1


def read_context(args, digest, tick=None):
    expected = owners(args,tick)
    rp,cp,jp = ADMISSION/'result.json',ADMISSION/'completion.json',ADMISSION/'context.json'
    require(sha(jp,tick) == digest, 'Exact caller-supplied admission context')
    value,record,end = read(jp,CONTEXT_LIMIT),read(rp,CONTEXT_LIMIT),read(cp,CONTEXT_LIMIT)
    require(value['schema'] == CONTEXT_SCHEMA and record['schema'] == ADMIT_SCHEMA
            and all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
            and end['passed'] and end['completed'] and end['result_sha256'] == sha(rp,tick)
            and record['context_sha256'] == record['artifacts'][str(jp)] == digest
            and value['reviewed_owners'] == record['reviewed_owners'] == expected
            and value['candidate_sha256'] == CANDIDATE_SHA and value['build_sha256'] == BUILD_SHA
            and value['role'] == 'terminal' and value['ranks'] == list(VALID)
            and value['full_frozen_admission_passed'] and value['observer_inverse_verified'] and value['paid_loop_inverse_verified']
            and not value['estimator_executed'] and not value['numerical_imports'] and not value['truth_accessed'],
            'Complete immutable target-free admission producer')
    for group in ('source_bindings','input_bindings'):
        require(value[group] == record[group], 'Admission record/context inventories identical')
        bind(value[group],tick)
    require(all(value['source_bindings'].get(k) == v for k,v in expected.items()), 'All reviewed owners included')
    value['input_bindings'].update({str(jp):digest,str(rp):sha(rp,tick),str(cp):sha(cp,tick)})
    value['reference_by_rank'] = {int(k):v for k,v in value['reference_by_rank'].items()}
    return value


def retired_admission(args, context):
    path = WATCH_OUT/'admission_retired.json'
    require(sha(path) == args.admission_watch_sha256, 'Independent retired-admission handoff')
    value = read(path,CONTEXT_LIMIT)
    require(value['schema'] == RETIRED_SCHEMA and value['passed'] and value['completed']
            and value['whole_owned_tree_retired'] and value['context_sha256'] == args.context_sha256
            and value['reviewed_owners'] == owners(args) and value['admission_elapsed_s'] < 60
            and value['whole_elapsed_s'] < 120 and value['maximum_sampled_rss_bytes'] < RSS_LIMIT
            and value['maximum_swap_growth_bytes'] == 0 and value['maximum_sample_interval_s'] <= 1,
            'Admission child was retired under the complete unchanged gates')
    context['input_bindings'][str(path)] = args.admission_watch_sha256
    return value


def validate_diagnostic(diagnostic, call):
    domain,startup,dispatches = (diagnostic[k] for k in ('domains','startup','dispatches'))
    keys = {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
    require(domain['variance_count'] == 16 and domain['selector_count'] == 47
            and set(domain['variance_domain']) == {str(l) for l in range(16)} and set(domain['selection_cells']) == keys
            and all(v['finite'] and math.isfinite(v['minimum']) and v['minimum'] >= 1e-10 and v['below_floor_count'] == 0
                    for v in domain['variance_domain'].values()) and all(v['floor'] == .01 and type(v['count']) is int
                    and 0 <= v['count'] <= 1024 and v['count'] % 32 == 0 for v in domain['selection_cells'].values()),
            'All16 unfloored finite domains and all47 original selector cells')
    cold = call == 0
    require(startup['prefix']['P_empty_at_entry'] is cold and startup['prefix']['P_ready_at_binary_entry']
            and len(startup['coefficients']) == 1 and startup['coefficients'][0]['empty_at_entry'] is cold
            and startup['coefficients'][0]['flops'] == (120 if cold else 0)
            and startup['extra_empty_before'] is cold and startup['extra_ready_after'], 'P/cstar/both serial bindings initialized once')
    bindings = startup['bindings']; extra = [x for x in bindings if x['name'] in EXTRA_LAYOUTS]
    require(len(extra) == (2 if cold else 0) and len(startup['extra_binding_regions']) == (1 if cold else 0)
            and startup['extra_binding_flops'] == (EXTRA_BINDINGS if cold else 0)
            and all(x['same_original_arena'] and x['arena_dtype'] == 'float32' and x['flops'] == x['expected_flops']
                    for x in bindings), 'Source-owned measured layout tariffs')
    if cold:
        require([x['name'] for x in extra] == list(EXTRA_LAYOUTS) and all(
            x['flops'] == EXTRA_LAYOUTS[x['name']] and x['reshape_calls'] == 34 for x in extra)
            and startup['extra_binding_regions'][0] == dict(flops=EXTRA_BINDINGS,operation_count=68,
                same_original_arena=True,nested_binding_flops_not_added_twice=True), 'All68 additional public reshapes paid')
    require(startup['observed_startup_region_flops'] == startup['prefix']['flops']+
            sum(x['flops'] for x in bindings+startup['coefficients']) and startup['extra_regions_are_nested_in_bindings']
            and startup['original_arena_bytes'] == 1168900096 and startup['no_second_backing_arena']
            and domain['added_participant_array_operations'] == startup['added_participant_array_operations'] == 0
            and startup['public_counters_only'], 'Disjoint startup total; nested extra regions never counted twice')
    expected = {l:(sum(domain['selection_cells'][f'{j}:2']['count'] for j in range(l-1))*2//8192
                    if domain['selection_cells'][f'{l-1}:0']['count'] == 832 else 0) for l in range(1,16)}
    counts = {l:0 for l in range(1,16)}
    for item in dispatches:
        require(item['layer'] in counts and item['shape'] == list(SHAPE) and item['dtype'] == 'float32'
                and item['flops'] == PRODUCT_FLOPS and item['same_original_arena'] and item['returned_original_out']
                and item['child_bypasses_dispatcher'] and [x['child'] for x in item['children']] == list(range(7))
                and all(x['depth'] == 5 and type(x['flops']) is int and x['flops'] > 0 for x in item['children'])
                and len({x['flops'] for x in item['children']}) == 1, 'Exact occurring serial tiles and seven direct children')
        counts[item['layer']] += 1
    require(counts == expected, 'Actual current-trajectory dispatch census, never frozen P1 hit counts')
    return domain,startup,dispatches


def after_bill(np, observer, trial, result, folder, refs):
    packet,startup,dispatches = observer.finish()
    domains = base().SMALL.diagnostics(np,packet)
    domains['selection_cells'] = {k:{name:v[name] for name in ('count','floor','indices_sha256','mask_sha256')}
                                 for k,v in domains['selection_cells'].items()}
    require(result['input_preparation'][-1]['weight_file_sha256'] == refs[trial['rank']]['weight_file_sha256'], 'Exact P1 official inputs')
    diagnostic = dict(call=trial['call'],rank=trial['rank'],domains=domains,startup=startup,dispatches=dispatches,
        truth_accessed=False,offline_after_saved_bill_and_context_release=True,
        old_bill_transplanted=False,parent_output_parity_required=False)
    validate_diagnostic(diagnostic,trial['call'])
    path = folder/f"call_{trial['call']:02d}_diagnostics.json"; write(path,diagnostic)
    item = dict(path=str(path),sha256=sha(path))
    result['artifacts'][str(path)] = item['sha256']; result['diagnostic_records'].append(item)


def finish_native(result):
    require([x['rank'] for x in result['trials']] == list(VALID+(37,)) and len(result['diagnostic_records']) == 41,
            'Complete40 primary calls and repeat37')
    records = [read(x['path']) for x in result['diagnostic_records']]
    for call,record in enumerate(records): validate_diagnostic(record,call)
    require(records[0]['domains'] == records[-1]['domains'] and result['trials'][0]['output_bytes_sha256'] ==
            result['trials'][-1]['output_bytes_sha256'], 'Exact repeated output/prevariance/ordered-support bytes')
    canonical = lambda ds:[{k:v for k,v in d.items() if k != 'operation_count'} for d in ds]
    require(canonical(records[0]['dispatches']) == canonical(records[-1]['dispatches']), 'Repeated serial arithmetic roster/F')
    startup = [r['startup']['observed_startup_region_flops'] for r in records]
    measured = result['trials'][0]['flops']-result['trials'][-1]['flops']
    require(measured == startup[0]-startup[-1] and measured > 0 and not records[-1]['startup']['bindings'],
            'Cold/repeat difference equals measured disjoint initialization, not assumed parent cost')
    result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
        initialization_flops=measured,initialization_ownership_passed=True,
        initialization_accounting=dict(cold_minus_repeat_flops=measured,observed_startup_difference_flops=startup[0]-startup[-1],
            per_call_observed_startup_region_flops=startup,startup_subtracted_from_primary_bills=False,
            old_initialization_value_assumed=False,additional_binding_flops=EXTRA_BINDINGS,
            nested_extra_binding_regions_not_double_counted=True),
        all16_domains_qualified=True,all47_selectors_qualified=True,repeat_output_domain_byte_exact=True,
        own_complete_bills=True,all_outputs_and_bills_sealed_before_scoring=True,
        additional_bindings_paid_once=True,same_existing_arena_qualified=True,dispatch_tariffs_qualified=True,
        dispatch_counts=[len(x['dispatches']) for x in records],primary_dispatch_count=sum(len(x['dispatches']) for x in records[:40]))


def verify_acquisition(record, end, digest, args, context, tick=None):
    """Stdlib saved-output/bill/domain verifier; no targets or model invocation."""
    flags = ('passed','completed','hashes_unchanged','sources_unchanged','own_complete_bills',
        'all16_domains_qualified','all47_selectors_qualified','repeat_output_domain_byte_exact',
        'initialization_ownership_passed','source_inverse_verified','paid_arithmetic_and_lifecycle_unchanged',
        'additional_bindings_paid_once','same_existing_arena_qualified','dispatch_tariffs_qualified')
    require(record['schema'] == NATIVE_SCHEMA and record['stage'] == 'native' and record['role'] == 'terminal'
            and all(record.get(k) is True for k in flags) and end['passed'] and end['completed']
            and end['result_sha256'] == digest and record['candidate_sha256'] == CANDIDATE_SHA
            and record['build_sha256'] == BUILD_SHA and record['builder_sha256'] == BUILDER_SHA
            and record['truth_accessed'] is False and record['ranks'] == list(VALID)
            and record['admission_context_sha256'] == args.context_sha256
            and record['admission_watch_sha256'] == args.admission_watch_sha256
            and record['prerequisites'] == context['prerequisites'] and record['proof'] == context['proof'], 'Completed fixed native endpoint')
    for group in ('source_bindings','input_bindings','artifacts'): bind(record.get(group,{}),tick)
    require(all(record['source_bindings'].get(k) == v for group in ('source_bindings','input_bindings')
                for k,v in context[group].items()), 'Every admitted dependency remains bound by the original paid-loop result')
    require(len(record['trials']) == len(record['input_preparation']) == len(record['diagnostic_records']) == 41,
            'Exact41 own prediction/input/diagnostic records')
    helper = load(CENSUS,'partial_d6_lifecycle_verifier',CENSUS_SHA); n = base(); paths = set()
    timings = n.legacy().TIMINGS
    refs = context['reference_by_rank']
    for call,(trial,prepared) in enumerate(zip(record['trials'],record['input_preparation'])):
        rank = (VALID+(37,))[call]
        require(trial['call'] == prepared['call'] == call and trial['rank'] == prepared['rank'] == rank
                and trial['includes_initialization'] is (call == 0) and trial['multiplier'] == max(.1,trial['flops']/BUDGET), 'Exact cold/primary/repeat ownership')
        for suffix,bill,life in (('',trial,'lifecycle'),('_input',prepared,'input_reset')):
            require(type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET
                    and all(math.isfinite(bill[k]) and bill[k] >= 0 for k in timings)
                    and bill['wall_s'] < 120 and bill['context_wall_s'] < 120 and 0 < bill['peak_rss_bytes'] < RSS_LIMIT,
                    'Complete actual bill/resources')
            bp,lp = OUTPUT/f'call_{call:02d}{suffix}_bill.json',OUTPUT/f'call_{call:02d}_{life}.json'
            require(read(bp) == bill and record['artifacts'][str(bp)] == sha(bp,tick)
                    and record['artifacts'][str(lp)] == sha(lp,tick), 'Own saved bill/lifecycle inventory')
            helper.lifecycle(dict(bill,lifecycle_receipt=str(lp)),bp,lp); paths.update((str(bp),str(lp)))
        require(prepared['flops'] == 16*1024*1024 and prepared['layers'] == 16 and not prepared['charged_to_participant']
                and all(prepared[k] is True for k in ('official_fnp_type','exact_bytes','independent_weight_buffers','context_closed'))
                and prepared['weight_file_sha256'] == refs[rank]['weight_file_sha256'], 'Official owned FP32 input convention')
        op,dp = OUTPUT/f'call_{call:02d}_rank{rank}.npy',OUTPUT/f'call_{call:02d}_diagnostics.json'
        require(trial['output'] == str(op) and trial['output_sha256'] == record['artifacts'][str(op)] == sha(op,tick)
                and record['diagnostic_records'][call] == dict(path=str(dp),sha256=sha(dp,tick))
                and record['artifacts'][str(dp)] == sha(dp,tick), 'Own output and complete diagnostic seal')
        n.npy_bytes(op,trial['output_bytes_sha256'])
        diagnostic = read(dp)
        require(diagnostic['call'] == call and diagnostic['rank'] == rank and diagnostic['truth_accessed'] is False
                and diagnostic['offline_after_saved_bill_and_context_release'], 'Detached target-free observation')
        validate_diagnostic(diagnostic,call); paths.update((str(op),str(dp)))
    require(set(record['artifacts']) == paths and len(paths) == 246, 'Only original205 artifacts plus41 small diagnostics')
    replay = dict(record); finish_native(replay)
    for key in ('initialization_flops','initialization_accounting','dispatch_counts','primary_dispatch_count'):
        require(replay[key] == record[key], 'Recomputed complete initialization and dispatch summary')


def verify_native(record, end, digest, args, context):
    return verify_acquisition(record,end,digest,args,context)


def worker_main(args):
    environment(); context = read_context(args,args.context_sha256); retired_admission(args,context)
    n = base(); Observer,source,proof = prepare(); legacy = n.legacy()
    require(proof == context['proof'] and tuple(legacy.VALID) == VALID, 'Same retired-admission source proofs')
    versions = {k:importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench','threadpoolctl')}
    require(versions == dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'), 'Qualified native frame')
    require(not OUTPUT.exists() and not OUTPUT.is_symlink() and not HERE.is_symlink(), 'Exclusive native40; no resume')
    candidate = HERE/'build_v2/estimator.py'; require(sha(candidate) == CANDIDATE_SHA, 'Frozen exact-shape source')
    pins = {**context['source_bindings'],**context['input_bindings']}
    header = dict(schema=NATIVE_SCHEMA,stage='native',role='terminal',completed=False,sources_unchanged=False,
        build_sha256=BUILD_SHA,builder_sha256=BUILDER_SHA,prerequisites=context['prerequisites'],versions=versions,
        truth_accessed=False,no_fit=True,score_measured=False,old_max_read_timing_gate_still_failed=True,
        native_original_sha256=NATIVE_SHA,native_main_sha256=proof['native_original_main_sha256'],
        native_adapter_parent_sha256=RC_SHA,source_inverse_verified=True,paid_arithmetic_and_lifecycle_unchanged=True,
        original_TL_diagnostics_unchanged=True,response_floor=.01,active_floor=.01,birth_floor=.01,
        own_complete_bills=False,old_initialization_value_assumed=False,auto_launch_next_stage=False,
        admission_context_sha256=args.context_sha256,admission_watch_sha256=args.admission_watch_sha256,
        proof=proof,full_admission_in_retired_process=True,production_qualified=False)
    namespace = dict(vars(legacy),__file__=str(Path(__file__).resolve()),PARENT=NEW,CANDIDATE=candidate,
        NATIVE_PINS=pins,FIXED_RANKS=VALID,Observer=Observer,observed_predict=n.observed_predict,
        after_bill=lambda np,obs,trial,result,folder:after_bill(np,obs,trial,result,folder,context['reference_by_rank']),
        finish_native=finish_native,NATIVE_HEADER=header)
    os.chdir(NEW); saved = sys.argv
    try:
        sys.argv = [str(__file__),'--source',str(candidate),'--split','validation','--out',str(OUTPUT)]
        exec(compile(source,str(__file__)+':preserved_paid_native','exec'),namespace)
        return namespace['main']()
    finally:
        sys.argv = saved


def main():
    args = arguments('child')
    if args is None: return 0
    return admission_main(args) if args.stage == 'admit' else worker_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
