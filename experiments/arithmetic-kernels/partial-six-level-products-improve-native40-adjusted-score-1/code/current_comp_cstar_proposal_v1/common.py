"""Inert P1 admission/I/O reuse with fresh cstar-proposal authorities."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
P1 = NEW/'current_d50_native_objective_p1_v1'
CO = NEW/'current_bd_p_coadaptation_v1'
TL = NEW/'current_d50_terminal_linear_v2'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2/'public_g22_operator_features_v1/truthpost'
DATA = NEW/'joint_prefix_final_data_v1.py'
INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
PROTOCOL_SHA = '0a1db21c3a9f6f0a268ae8c47c1f4ad1c6b16ff6a0438aaeb2278dfaa6e97359'
MODEL_SHA = 'f5c26f04750fa147ef897484ddc3a152ad744e6170b7542e8046af3193a79656'
P1_COMMON_SHA = '567ccbae6caf516043d72d54a9073cc60e7fafeac31934d29379ef578fea0a7f'
P1_WORKER_SHA = 'd1160bfb2f5ac0edf4d425f09282ec79bbca310cb2f88cf7b97b6c59acbd426c'
HELPER_SHA = '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20'
CAPTURE_SHA = 'cb16019fe0a72b1ef11991e253992d3a9544d8f1b28df403b2c8b2ee290a650a'
CAPTURE_WATCH_SHA = '559dd7bdb1058fa060b9b0927af0ec5cd3cd32226bbc5093c921f4f4102d6b3b'
BANK_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
SCHEMA = 'current_comp_cstar_proposal_v1'
SEED, BUDGET = 9309099, 2**41
VERSIONS = {'numpy': '1.26.4', 'torch': '2.7.0'}
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
_CLASS = _PROOF = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.publishing')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    os.link(temporary, path)
    temporary.unlink()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Pinned metadata module')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--stage', choices=('qualify', 'train'))
    for key in ('source', 'common', 'model', 'protocol', 'watcher', 'watch-spec',
                'qualification', 'qualification-watch'):
        p.add_argument('--'+key+'-sha256')
    args = p.parse_args()
    if not args.run:
        p.print_help()
        return None
    require(args.stage is not None, 'Explicit separate stage')
    keys = ['source', 'common', 'model', 'protocol', 'watcher', 'watch_spec']
    if args.stage == 'train':
        keys += ['qualification', 'qualification_watch']
    else:
        require(args.qualification_sha256 is None and args.qualification_watch_sha256 is None,
                'No historical stage grants new qualification eligibility')
    for key in keys:
        value = getattr(args, key+'_sha256')
        require(isinstance(value, str) and len(value) == 64
                and all(x in '0123456789abcdef' for x in value), 'Explicit reviewed SHA: '+key)
    return args


def inherited_stage():
    """Compile only the byte-exact P1 I/O/cost methods; old admission excluded."""
    require(sha(P1/'common.py') == P1_COMMON_SHA, 'Exact P1 helper owner')
    source = (P1/'common.py').read_text()
    nodes = [x for x in ast.parse(source).body if isinstance(x, ast.ClassDef) and x.name == 'Stage']
    require(len(nodes) == 1, 'Unique original Stage owner')
    names = ('__init__', 'pin', 'inventory', 'receipt', 'weight_rows', 'capture_costs',
             'json', 'save', 'bundle', 'guard', 'finish')
    methods = {x.name: x for x in nodes[0].body if isinstance(x, ast.FunctionDef)}
    lines = source.splitlines(keepends=True)
    original = 'class Stage:\n'+''.join(''.join(lines[methods[k].lineno-1:methods[k].end_lineno])+'\n' for k in names)
    derived = original.replace('class Stage:\n', 'class ParentStage:\n', 1)
    require(derived.replace('class ParentStage:\n', 'class Stage:\n', 1) == original,
            'Exact inverse for every retained P1 method')
    env = dict(globals())
    exec(compile(derived, str(P1/'common.py')+':exact-I/O-subset', 'exec'), env)
    proof = dict(source_sha256=P1_COMMON_SHA, methods=list(names),
        original_subset_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_subset_sha256=hashlib.sha256(derived.encode()).hexdigest(),
        inverse_source_byte_exact=True, weight_rows_and_capture_costs_unchanged=True,
        original_failed_stage_admission_excluded=True)
    return env['ParentStage'], proof


def stage_class():
    global _CLASS, _PROOF
    if _CLASS is not None:
        return _CLASS
    parent, proof = inherited_stage()

    class CurrentStage(parent):
        def __init__(self, args, worker_path):
            super().__init__(args, worker_path)
            self.labels_enabled = False
            self.record.update(native_gradient_qualified=False, CS_failure_preserved=True,
                full_model_calls=0, retired_call_count=0, optimizer_updates=0,
                frozen_parent_parameter_updates=0, tiny_model_calls=0,
                common_source_proof=proof, parent_candidate_sha256=None)

        def pin(self, path, expected=None, group='input_bindings'):
            path = Path(path).resolve()
            if group == 'target_bindings':
                require(self.labels_enabled and self.args.stage == 'train'
                        and path in {TRUTH/f'truthpost_rank{r:04d}.npz' for r in self.train},
                        'Fresh successful qualification before fixed TRAIN targets only')
            return super().pin(path, expected, group)

        def three_receipt(self, folder, digest):
            folder = Path(folder)
            pins = [self.pin(folder/name, digest if name == 'result.json' else None)
                    for name in ('result.json', 'completion.json', 'finalization.json')]
            r, end, final = (read(folder/name) for name in ('result.json', 'completion.json', 'finalization.json'))
            require(all(x.get('passed') is True and x.get('completed') is True for x in (r, end, final))
                    and end['result_sha256'] == final['result_sha256'] == pins[0]
                    and final['completion_sha256'] == pins[1] and final['elapsed_s'] < 300,
                    'All three actual new qualification seals within300s')
            self.inventory(r)
            return r, pins

        def admit(self):
            args = self.args
            require(args.model_sha256 == MODEL_SHA and args.protocol_sha256 == PROTOCOL_SHA,
                    'Fixed prospective model/protocol, not CS information authority')
            self.owners = {self.worker_path: args.source_sha256, HERE/'common.py': args.common_sha256,
                HERE/'model.py': MODEL_SHA, HERE/'PROTOCOL.md': PROTOCOL_SHA,
                HERE/'watch.py': args.watcher_sha256, HERE/'WATCH_SPEC.md': args.watch_spec_sha256}
            for path, digest in {**self.owners, P1/'common.py': P1_COMMON_SHA,
                    P1/'worker.py': P1_WORKER_SHA, CO/'qualify.py': HELPER_SHA,
                    DATA: '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b'}.items():
                self.pin(path, digest, 'source_bindings')
            require(sys.executable == '/Users/matt/anaconda3/bin/python' and sys.dont_write_bytecode
                    and sys.flags.optimize == 0, 'AP interpreter, bytecode off, assertions enabled')
            require(os.getpriority(os.PRIO_PROCESS, 0) >= 10
                    and all(os.environ.get(k) == '1' for k in BLAS), 'nice10/all five BLAS limits1')
            require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
                    and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
                    and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'Fixed MPS launch frame')
            require(not {'numpy', 'torch', 'flopscope', 'threadpoolctl'} & set(sys.modules), 'Source admission before numerics')
            self.model = module(HERE/'model.py', 'cprop_exact_comp_model')
            self.h = module(CO/'qualify.py', 'cprop_original_scalar_and_array_io')
            for path, digest in self.model.authenticate().items():
                self.pin(path, digest, 'source_bindings' if Path(path).suffix in ('.py', '.md') else 'input_bindings')
            require(not {'numpy', 'torch', 'flopscope', 'threadpoolctl'} & set(sys.modules), 'All authenticated helpers are inert')
            declarations = {x.targets[0].id: x.value for x in ast.parse(DATA.read_text()).body
                if isinstance(x, ast.Assign) and len(x.targets) == 1 and isinstance(x.targets[0], ast.Name)}
            self.valid = tuple(ast.literal_eval(declarations['VALID']))
            self.train = tuple(r for r in range(35, 235) if r not in self.valid)
            require(len(self.train) == 160 and len(self.valid) == len(set(self.valid)) == 40, 'Exact original160/40 split')
            self.record.update(model_sha256=MODEL_SHA, parent_candidate_sha256=self.model.COMP_SHA,
                train_ranks=list(self.train), validation_ranks=list(self.valid))
            if args.stage == 'train':
                q, qp = self.three_receipt(HERE/'qualification_v1', args.qualification_sha256)
                w, wp = self.three_receipt(HERE/'qualification_watch_v1', args.qualification_watch_sha256)
                require(q['schema'] == SCHEMA and q['stage'] == 'qualify'
                        and q['execution_integrity_passed'] and q['hashes_unchanged'] and q['sources_unchanged']
                        and q['full_model_calls'] == q['retired_call_count'] == q['graphs_completed'] == 5
                        and len(q['graph_records']) == 5 and q['gradient_calls'] == q['zero_lr_Adam_calls'] == 3
                        and q['reference_forwards'] == 2 and q['parameter_updates'] == q['optimizer_updates'] == 0
                        and q['native_gradient_qualified'] is False and q['no_native_gradient_claim']
                        and q['training_targets_accessed'] is False and q['validation_targets_accessed'] is False,
                        'Complete fresh five-call proposal integrity, not a derivative gate')
                for flag in ('same_backend_null_byte_exact', 'first_repeat35_prediction_byte_exact',
                             'first_repeat35_gradient_byte_exact', 'all16_domains_qualified',
                             'all47_selectors_qualified', 'frozen_P_A_c0', 'zero_lr_delta_byte_exact',
                             'all_graphs_retired', 'all_persistent_owners_retired', 'CS_failure_preserved'):
                    require(q.get(flag) is True, 'Complete new qualification flag: '+flag)
                require(w['schema'] == 'current_comp_cstar_proposal_watch_v1' and w['stage'] == 'qualify'
                        and all(w.get(k) is True for k in ('hashes_unchanged', 'sources_unchanged',
                            'child_execution_completed', 'whole_owned_tree_retired', 'cleanup_passed',
                            'observational_groups_retired', 'actual_completed_samples', 'foreign_numerical_workers_checked'))
                        and w['retired_calls'] == w['completed_graph_count'] == 5
                        and [w['child_result_sha256'], w['child_completion_sha256'], w['child_finalization_sha256']] == qp
                        and w['maximum_combined_rss_bytes'] < 12_000_000_000
                        and w['maximum_swap_growth_bytes'] == 0 and w['maximum_interval_s'] <= 1
                        and w['elapsed_s'] < 300 and w['whole_cap_s'] == 300,
                        'Actual independent qualification resource/retirement authority')
                for record in (q, w):
                    require(all(record['source_bindings'].get(str(path)) == digest for path, digest in self.owners.items()),
                            'Qualification binds the exact same reviewed six new owners')
                require(q['environment']['versions'] == VERSIONS, 'Qualified exact library frame')
                self.record['prerequisites'] = dict(qualification_result_sha256=qp[0],
                    qualification_completion_sha256=qp[1], qualification_finalization_sha256=qp[2],
                    qualification_watch_result_sha256=wp[0], qualification_watch_completion_sha256=wp[1],
                    qualification_watch_finalization_sha256=wp[2])
                self.labels_enabled = True
            self.weights = self.weight_rows((35, 100) if args.stage == 'qualify' else range(35, 235))
            if args.stage == 'train':
                self.costs = self.capture_costs()
                self.pin(INDEX, 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971')
                self.targets = {x['rank']: x['target_sha256'] for x in read(INDEX)['target_sources']}
                require(set(self.targets) == set(self.train), 'Only original TRAIN1e9 target metadata')
            self.record['environment'] = dict(executable=sys.executable, nice=os.getpriority(os.PRIO_PROCESS, 0),
                launch={k: os.environ.get(k) for k in BLAS}, bytecode_disabled=True)
            self.json('admission.json', dict(source_bindings=self.record['source_bindings'],
                input_bindings=self.record['input_bindings'], weight_bindings=self.record['weight_bindings'],
                stage=args.stage, prerequisites=self.record['prerequisites'],
                training_targets_accessed=False, validation_targets_accessed=False,
                CS_failure_preserved=True, native_gradient_qualified=False))

        def graph_complete(self, index, item):
            require(index == len(self.record['graph_records']), 'Ordered fully retired call')
            resources = self.guard()
            marker = self.json(f'graph_{index:04d}_complete.json', dict(item, graph_index=index,
                passed=True, completed=True, graph_retired=True, elapsed_s=resources['elapsed_s'], resources=resources))
            self.record['graph_records'].append(marker)
            self.record['full_model_calls'] = self.record['retired_call_count'] = len(self.record['graph_records'])
            print(json.dumps(dict(event='retired', call=index, graph_index=index, rank=item['rank'], kind=item['kind'],
                complete_path=marker['path'], complete_sha256=marker['sha256'],
                host_peak_rss_bytes=resources['peak_rss_bytes'], mps_driver_bytes=resources['mps_driver_bytes']),
                sort_keys=True, allow_nan=False), flush=True)

        def close(self):
            self.record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                wall_s=time.monotonic()-self.start, graphs_completed=len(self.record['graph_records']))
            limit = 300 if self.args.stage == 'qualify' else 1800
            if self.record['passed'] and time.monotonic()-self.start >= limit:
                self.record.update(passed=False, completed=False, error='Whole stage expired before publication')
            write(self.folder/'result.json', self.record)
            result_sha = sha(self.folder/'result.json')
            ok = bool(self.record['passed'] and time.monotonic()-self.start < limit)
            write(self.folder/'completion.json', dict(schema=SCHEMA, stage=self.args.stage,
                passed=ok, completed=ok, result_sha256=result_sha))
            completion_sha = sha(self.folder/'completion.json')
            elapsed = time.monotonic()-self.start
            ok = bool(ok and elapsed < limit)
            write(self.folder/'finalization.json', dict(schema=SCHEMA, stage=self.args.stage,
                passed=ok, completed=ok, result_sha256=result_sha, completion_sha256=completion_sha, elapsed_s=elapsed))
            sha(self.folder/'finalization.json')
            return bool(ok and time.monotonic()-self.start < limit)

    _CLASS, _PROOF = CurrentStage, proof
    return _CLASS


def Stage(args, worker_path):
    return stage_class()(args, worker_path)


def source_proof():
    require(_PROOF is not None, 'Construct explicit source helper first')
    return dict(_PROOF)
