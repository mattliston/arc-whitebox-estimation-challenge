"""Inert singleton cold worker for the fixed terminal-tree endpoint.

The reviewed TLv2 main supplies its literal paid and reset spans. A reversible
source adapter changes only proof location, module label, no-replay P check,
and archived-output comparison. Admission/IO reuse TLv2 metadata helpers.
No targets, fitting, head replay, uncharged estimator operations, or retries.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
TL = NEW / 'current_d50_terminal_linear_v2'
TL_NATIVE = TL / 'native.py'
TL_COMMON = TL / 'common.py'
TL_NATIVE_SHA = '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe'
TL_COMMON_SHA = '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a'
PROTOCOL_SHA = 'f4bf2907fa1973b6076d6f485cd2436d6b5bf13cfe26077653e52af3c3e59dad'
BUILDER_SHA = 'ae8f90bd3767f89e157b32dd75817b0c65e055091667ba20d2476e6a81a0c2f8'
TINY_SOURCE_SHA = 'd7ea9898f55555591f3d23cca0d04c0a412b38bca879f99d3f02f36866dd668c'
BUILD_SHA = 'a94d6109b2ec176a9592a6593b3a6f9ebc5bf8f37f953b78f59dd992b39bda00'
CANDIDATES = {
    'zero': (HERE/'build_v1/zero_estimator.py', 'f69e2263d599ba61fdd4a8c1c2dbe1b2e7585950244c205ed253ff3d186130f3'),
    'enabled': (HERE/'build_v1/estimator.py', '542411d1c9c71cad423f3ff43c1649ba14d411a052a1eef07247fdf16d448596'),
}
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
ROSTER = (('zero',35,'zero_0035'), ('zero',100,'zero_0100'),
          ('enabled',35,'enabled_0035'), ('enabled',100,'enabled_0100'),
          ('enabled',35,'enabled_0035_repeat'))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No target-array access')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module specification')
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def admission(build_sha256, tiny_sha256):
    """Replayable stdlib admission; never decodes arrays or opens weights."""
    require(build_sha256 == BUILD_SHA and re.fullmatch('[0-9a-f]{64}',tiny_sha256 or ''),
            'Fixed build and explicit completed tiny digest')
    pins = {}
    def pin(path, expected):
        path = Path(path).resolve()
        require(path.is_relative_to(P2) and sha(path) == expected, 'Prerequisite pin: '+str(path))
        require(str(path) not in pins or pins[str(path)] == expected, 'Conflicting prerequisite')
        pins[str(path)] = expected
    for path,digest in ((HERE/'build.py',BUILDER_SHA),(HERE/'check_tiny.py',TINY_SOURCE_SHA),
                        (HERE/'PROTOCOL.md',PROTOCOL_SHA),(TL_NATIVE,TL_NATIVE_SHA),(TL_COMMON,TL_COMMON_SHA)):
        pin(path,digest)
    records = {}
    for name,folder,digest in (('build',HERE/'build_v1',BUILD_SHA),('tiny',HERE/'tiny_v1',tiny_sha256)):
        result_path, end_path = folder/('build.json' if name == 'build' else 'result.json'), folder/'completion.json'
        pin(result_path,digest)
        pin(end_path,sha(end_path))
        record, end = read(result_path), read(end_path)
        require(all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
                and end['passed'] and end['completed'] and end['result_sha256'] == digest,
                'Successful sealed '+name)
        require(record['truth_accessed'] is False and record['exact_source_inverse'] is True,
                'Target-free inverse-certified prerequisite')
        for group in ('source_bindings','artifacts'):
            for path,expected in record[group].items():
                pin(path,expected)
        records[name] = record
    build,tiny = records['build'],records['tiny']
    require(build['schema'] == 'terminal_full_fourth_tree_build_v1' and
            build['source_bindings'][str(HERE/'build.py')] == BUILDER_SHA and
            build['only_enable_literal_differs'] and build['natural_coefficient'] == 1 and
            build['full_unmasked_legs'] and (build['source_layer'],build['readout_layer']) == (14,15),
            'Sole natural-strength terminal build')
    require(tiny['schema'] == 'terminal_full_fourth_tree_tiny_v1' and
            tiny['source_bindings'][str(HERE/'check_tiny.py')] == TINY_SOURCE_SHA and
            tiny['source_bindings'][str(HERE/'build.py')] == BUILDER_SHA and
            (tiny['seed'],tiny['n'],tiny['trees'],tiny['paths'],tiny['stars']) == (9309101,4,16,12,4) and
            tiny['relative_tolerance'] == 1e-11 and tiny['absolute_tolerance'] == 1e-12 and
            len(tiny['checks']) == 58 and all(x['passed'] for x in tiny['checks']) and
            tiny['wick_factorials_verified'] and tiny['zero_source_passed'] and
            tiny['resource_gate_passed'] and tiny['wall_s'] < 30 and tiny['peak_rss_bytes'] < 1_000_000_000,
            'Fixed tiny tree/Wick/trace/resource certificate')
    builder = module(HERE/'build.py','terminal_tree_reviewed_builder')
    original = builder.PARENT.read_bytes().decode('utf-8')
    require(sha(builder.PARENT) == builder.PARENT_SHA, 'Original d50 parent')
    for role,(path,digest) in CANDIDATES.items():
        require(build['candidates'][role] == dict(path=str(path),sha256=digest,enabled=role == 'enabled'),
                'Exact candidate map')
        pin(path,digest)
        require(builder.restore(path.read_bytes().decode('utf-8'),role == 'enabled') == original,
                'Reauthenticated whole-parent source inverse')
    return pins, records


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--role',choices=tuple(CANDIDATES))
    parser.add_argument('--rank',type=int,choices=(35,100))
    parser.add_argument('--output',type=Path)
    for key in ('source','protocol','build-result','tiny-result'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
    return args


def bootstrap(args):
    require(args.role in CANDIDATES and args.rank in (35,100) and args.output is not None, 'Fixed role/rank/output')
    require(sha(__file__) == args.source_sha256 and args.protocol_sha256 == PROTOCOL_SHA,
            'Reviewed native source/protocol')
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope','threadpoolctl')),
            'Admission precedes numerical imports')
    pins, prerequisites = admission(args.build_result_sha256,args.tiny_result_sha256)
    pins[str(Path(__file__).resolve())] = args.source_sha256
    owner = module(TL_COMMON,'terminal_tree_tl_metadata')
    for path,digest in owner.FIXED.items():
        require(sha(path) == digest,'Authenticate inherited source before import: '+str(path))
    prior = sys.modules.get('common')
    sys.modules['common'] = owner
    try:
        legacy = module(owner.BD/'native_v2.py','terminal_tree_legacy_native')
        template = module(owner.FQ/'native.py','terminal_tree_frozen_fq_proof')
        native = module(TL_NATIVE,'terminal_tree_frozen_tl_native')
    finally:
        if prior is None:
            sys.modules.pop('common',None)
        else:
            sys.modules['common'] = prior

    class Stage(owner.Stage):
        def __init__(self, ignored_args, source):
            require(Path(sys.prefix).resolve() == owner.OFFPY.resolve() and sys.flags.optimize == 0
                    and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'OFFPY/assertions/no bytecode')
            require(os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in BLAS)
                    and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'nice10/BLAS1/no fallback')
            self.folder = args.output.resolve()
            require(self.folder != HERE and self.folder.is_relative_to(HERE) and not self.folder.exists(),
                    'Exclusive owned cold-call directory')
            self.folder.mkdir(parents=True)
            self.started, self.rank = time.monotonic(),args.rank
            self.result = dict(schema='terminal_full_fourth_tree_native_v1',role=args.role,rank=args.rank,
                passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                started_utc=datetime.now(timezone.utc).isoformat(),truth_accessed=False,weights_read=False,
                parameter_updates=0,gradient_calls=0,network_calls=0,MPS_used=False,score_measured=False,
                head_replay_performed=False,training_admitted=False,source_strength=1 if args.role=='enabled' else 0,
                parent_source_augmented=True,captures={},bindings={},source_bindings={},artifacts={},
                prerequisites=dict(build_result_sha256=BUILD_SHA,tiny_result_sha256=args.tiny_result_sha256))
            self.bindings = self.result['bindings']
            self.old_handler = signal.signal(signal.SIGALRM,self.guard)
            signal.setitimer(signal.ITIMER_REAL,.5,.5)
            write(self.folder/'started.json',self.result)
            try:
                for path,digest in pins.items():
                    self.pin(path,digest,source=Path(path).suffix in ('.py','.md'))
                for path,digest in owner.FIXED.items():
                    self.pin(path,digest)
                self.inventory(read(owner.BD/'build_terminal_v1/build.json'))
                for path,digest in legacy.PINS.items():
                    self.pin(path,digest)
                self.legacy,self.template = legacy,template
                rows = read(owner.STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
                matches = [x for x in rows if x['role_rank'] == args.rank]
                require(len(matches) == 1,'Unique canonical TRAIN weight row')
                path = (owner.STORE/matches[0]['path']).resolve()
                self.pin(path,matches[0]['sha256'])
                row = read(path)
                require(row['role_rank'] == args.rank,'Canonical rank')
                self.weights_path = path.parent/'weights.npy'
                weight = self.pin(self.weights_path,row['files']['weights.npy']['sha256'])
                self.mlp_seed = int(row['mlp_seed'])
                self.candidate = CANDIDATES[args.role][0]
                self.result.update(weights=dict(path=str(self.weights_path),sha256=weight['sha256'],mlp_seed=self.mlp_seed),
                    payload=self.bindings[str(owner.PAYLOAD)],candidate=str(self.candidate),
                    candidate_sha256=CANDIDATES[args.role][1],parent_sha256=owner.CANDIDATE_SHA,
                    original_build_sha256=owner.FIXED[owner.BD/'build_terminal_v1/build.json'],
                    initial_reference_required=True)
                # Inherited reference lookup deliberately keeps the original d50 identity.
                self.reference = self.fq_reference(args.rank)
                self.guard()
            except BaseException:
                self.result['error'] = traceback.format_exc()
                self.close()
                raise

    class Observer(native.Observer):
        def __init__(self, candidate):
            super().__init__(candidate)
            self.source_code = candidate._terminal_tree_source.__code__
            self.readout_code = candidate._terminal_tree_mean_delta.__code__
            self.source_calls,self.readout_calls = 0,0
        def __call__(self, frame, event, answer):
            if event == 'call':
                self.source_calls += frame.f_code is self.source_code
                self.readout_calls += frame.f_code is self.readout_code
            super().__call__(frame,event,answer)

    context = types.SimpleNamespace(**owner.__dict__)
    context.arguments = lambda:args
    context.Stage = Stage
    context.CANDIDATE = CANDIDATES[args.role][0]
    return owner,native,context,Observer


def check_frozen_P(np, stage, candidate, packet, arrays, observer):
    """Parameter integrity only. No evaluation/reconstruction of the head."""
    payload_path = Path(stage.result['payload']['path'])
    with np.load(payload_path,allow_pickle=False) as archive:
        payload = {key:archive[key].copy() for key in archive.files}
    require(set(payload) == set(candidate._GNN_LITERAL) == set(candidate._GNN_PARAMS),'Original P keys')
    for key,value in payload.items():
        literal = np.asarray(candidate._GNN_LITERAL[key],dtype=np.float32)
        require(value.dtype == np.float32 and np.isfinite(value).all() and value.shape == literal.shape
                and value.tobytes() == literal.tobytes() == candidate._GNN_PARAMS[key].tobytes(),
                'Unchanged current P bytes: '+key)
    for key,item in packet['parameters'].items():
        captured = np.frombuffer(item['data'],dtype=item['dtype']).reshape(item['shape'])
        require(captured.dtype == np.float32 and captured.shape == payload[key].shape
                and captured.tobytes() == payload[key].tobytes(),'Actual captured P unchanged: '+key)
    expected = int(stage.result['role'] == 'enabled')
    require(candidate._TERMINAL_TREE_ENABLED is bool(expected) and
            observer.source_calls == observer.readout_calls == expected,'Exactly one source/readout or literal zero')
    stage.result.update(original_P_unchanged=True,captured_parameter_bytes_exact=True,
                        tree_source_calls=observer.source_calls,tree_readout_calls=observer.readout_calls)


def check_reference(np, stage, trial, arrays):
    old = stage.reference
    served,raw = stage.load(np,old['served']),stage.load(np,old['raw_unclamped'])
    count = 16 if stage.result['role'] == 'zero' else 15
    diag = stage.result['diagnostics']
    checks = dict(required=True,earlier15_served_byte_exact=arrays['served'][:15].tobytes() == served[:15].tobytes(),
        earlier15_raw_byte_exact=arrays['raw'][:15].tobytes() == raw[:15].tobytes(),
        required_rows_served_byte_exact=arrays['served'][:count].tobytes() == served[:count].tobytes(),
        required_rows_raw_byte_exact=arrays['raw'][:count].tobytes() == raw[:count].tobytes(),
        ordered_selectors_exact=diag['selection_cells'] == old['diagnostics']['selection_cells'],
        all_prevariance_bytes_exact=all(diag['variance_domain'][str(l)]['raw_sha256'] ==
            old['diagnostics']['variance_domain'][str(l)]['raw_sha256'] for l in range(16)))
    if stage.result['role'] == 'zero':
        checks.update(complete_cold_flops_exact=trial['flops'] == old['complete_bill']['flops'],
                      operation_count_exact=trial['operation_count'] == old['complete_bill']['operation_count'])
    stage.result['archived_cold_reference'] = checks
    stage.result['flops_increment_over_same_input_d50'] = trial['flops'] - old['complete_bill']['flops']
    require(all(checks.values()),'Required d50 earlier-output/state/zero-bill identity')


def adapted_main(native, context, Observer):
    text = TL_NATIVE.read_text()
    nodes = [x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name == 'main']
    require(len(nodes) == 1,'Unique frozen native main')
    original = ''.join(text.splitlines(keepends=True)[nodes[0].lineno-1:nodes[0].end_lineno])
    start = original.index("            result['archived_cold_reference'] = dict(required=stage.reference is not None)")
    end = original.index('                del served_ref,raw_ref\n',start) + len('                del served_ref,raw_ref\n')
    comparison = original[start:end]
    replacements = (
        ("__file__=str(Path(__file__).resolve()))", "__file__=str(TL_NATIVE))"),
        ("'terminal_linear_unmodified_d50'", "'terminal_tree_'+args.role"),
        ('            replay(np,stage,module,packet,arrays)\n',
         '            check_frozen_P(np,stage,module,packet,arrays,observer)\n'),
        (comparison,'            check_reference(np,stage,trial,arrays)\n'),
    )
    derived = original
    for old,new in replacements:
        require(derived.count(old) == 1,'Unique native adapter span')
        derived = derived.replace(old,new,1)
    restored = derived
    for old,new in reversed(replacements):
        require(restored.count(new) == 1,'Unique inverse native adapter span')
        restored = restored.replace(new,old,1)
    require(restored == original,'Exact native main source inverse')
    # The identity proof inspects frozen TL source; this separate literal inverse
    # proves that its paid blocks and reset operations survive this adapter.
    namespace = dict(native.__dict__,__file__=str(Path(__file__).resolve()),c=context,TL_NATIVE=TL_NATIVE,
        Observer=Observer,check_frozen_P=check_frozen_P,check_reference=check_reference)
    exec(compile(derived,str(Path(__file__).resolve()),'exec'),namespace)
    return namespace['main'],dict(original_native_source_sha256=TL_NATIVE_SHA,
        original_main_sha256=hashlib.sha256(original.encode()).hexdigest(),
        adapted_main_sha256=hashlib.sha256(derived.encode()).hexdigest(),exact_inverse=True,
        paid_blocks_and_reset_spans_unchanged=True,head_replay_removed=True)


def main():
    args = arguments()
    if not args.run:
        return 0
    owner,native,context,Observer = bootstrap(args)
    runner,proof = adapted_main(native,context,Observer)
    base = context.Stage
    class RecordedStage(base):
        def __init__(self,*values):
            super().__init__(*values)
            self.result['source_adapter'] = proof
    context.Stage = RecordedStage
    return runner()


if __name__ == '__main__':
    raise SystemExit(main())
