"""Stdlib-only admission and exclusive receipts for the one-pass proposal.

Old CO failures are authenticated as failures, never converted into admission.
Only the new execution-integrity receipt and its external watch admit labels.
"""
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
CO = NEW/'current_bd_p_coadaptation_v1'
TL = NEW/'current_d50_terminal_linear_v2'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2/'public_g22_operator_features_v1/truthpost'
DATA = NEW/'joint_prefix_final_data_v1.py'
INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
PROTOCOL_SHA = '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35'
MODEL_SHA = '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0'
HELPER_SHA = '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20'
CPU_SHA = '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607'
FAILED_MPS_SHA = '91d1f3b56d4ff6f365b6216e3832226ccf71913d4b5618fb8697e4906f1cae7d'
CAPTURE_SHA = 'cb16019fe0a72b1ef11991e253992d3a9544d8f1b28df403b2c8b2ee290a650a'
CAPTURE_WATCH_SHA = '559dd7bdb1058fa060b9b0927af0ec5cd3cd32226bbc5093c921f4f4102d6b3b'
BANK_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
PAYLOAD_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
SCHEMA = 'current_d50_native_objective_p1_v1'
SEED, BUDGET = 9309096, 2**41
VERSIONS = {'numpy': '1.26.4', 'torch': '2.7.0'}
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


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
    """Publish an exclusive complete JSON file, never a partial progress marker."""
    path = Path(path)
    temporary = path.with_name(path.name+'.publishing')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    os.link(temporary, path)  # Fails if the final path already exists.
    temporary.unlink()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module loader')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stage', choices=('qualify', 'train'))
    for name in ('source', 'common', 'protocol', 'watcher',
                 'qualification-result', 'qualification-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return None
    require(args.stage is not None, 'Explicit stage required')
    for name in ('source', 'common', 'protocol', 'watcher'):
        value = getattr(args, name+'_sha256')
        require(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), 'Reviewed digest: '+name)
    if args.stage == 'train':
        require(args.qualification_result_sha256 and args.qualification_watch_sha256,
                'Both new execution-integrity seals required')
    else:
        require(args.qualification_result_sha256 is None and args.qualification_watch_sha256 is None,
                'Qualification does not consume a prior P1 stage')
    return args


class Stage:
    def __init__(self, args, worker_path):
        self.args, self.start = args, time.monotonic()
        self.folder = HERE/('qualification_v1' if args.stage == 'qualify' else 'training_v1')
        self.folder.mkdir(exist_ok=False)
        self.worker_path = Path(worker_path).resolve()
        self.record = dict(schema=SCHEMA, stage=args.stage, passed=False, completed=False,
            hashes_unchanged=False, sources_unchanged=False, source_bindings={}, input_bindings={},
            weight_bindings={}, target_bindings={}, artifacts={}, graph_records=[],
            training_targets_accessed=False, validation_targets_accessed=False,
            no_native_gradient_claim=True, old_failed_gates_preserved=True,
            native_score_measured=False, checkpoint_selection=False, parameter_updates=0,
            maximum_sampled_mps_driver_bytes=0, prerequisites={},
            started_utc=datetime.now(timezone.utc).isoformat())
        self.cache, self.torch = {}, None

    def pin(self, path, expected=None, group='input_bindings'):
        path = Path(path).resolve()
        is_truth = path.is_relative_to(TRUTH) or (path.suffix in ('.npy', '.npz')
                                                and 'truth' in path.name.lower())
        require((not is_truth) or (group == 'target_bindings' and self.args.stage == 'train'),
                'No truth through source/input/artifact admission')
        require(path.is_relative_to(P2), 'Input outside the fixed project frame')
        key = str(path)
        if key not in self.cache:
            before = path.stat()
            value = sha(path)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    'Input changed while hashing')
            self.cache[key] = value
        value = self.cache[key]
        require(expected is None or value == expected, 'SHA mismatch: '+key)
        previous = self.record[group].get(key)
        require(previous is None or previous == value, 'Conflicting pin')
        self.record[group][key] = value
        return value

    def inventory(self, record):
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'bindings', 'artifacts'):
            for path, item in record.get(group, {}).items():
                digest = item['sha256'] if isinstance(item, dict) else item
                self.pin(path, digest, 'source_bindings' if group == 'source_bindings' else 'input_bindings')

    def receipt(self, folder, digest, passed=True):
        folder = Path(folder)
        actual = self.pin(folder/'result.json', digest)
        self.pin(folder/'completion.json')
        record, end = read(folder/'result.json'), read(folder/'completion.json')
        require(end['result_sha256'] == actual, 'Exact result/completion seal')
        require(record['passed'] is passed and record['completed'] is passed
                and end['passed'] is passed and end['completed'] is passed, 'Receipt status')
        self.inventory(record)
        return record

    def admit(self):
        args = self.args
        require(args.protocol_sha256 == PROTOCOL_SHA, 'Fixed prospective protocol')
        for path, digest in ((self.worker_path, args.source_sha256),
            (HERE/'common.py', args.common_sha256), (HERE/'watch.py', args.watcher_sha256),
            (HERE/'PROTOCOL.md', PROTOCOL_SHA), (CO/'model.py', MODEL_SHA), (CO/'qualify.py', HELPER_SHA),
            (DATA, '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b')):
            self.pin(path, digest, 'source_bindings')
        require(sys.executable == '/Users/matt/anaconda3/bin/python', 'AP interpreter only')
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 required')
        require(all(os.environ.get(k) == '1' for k in BLAS), 'All five BLAS launch variables1')
        require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
                and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
                and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'Fixed MPS environment')
        require(not {'numpy', 'torch', 'threadpoolctl'} & set(sys.modules), 'Admission before numerical imports')
        self.model = module(CO/'model.py', 'p1_frozen_co_model')
        self.h = module(CO/'qualify.py', 'p1_frozen_co_io')
        for path, digest in self.model.authenticate().items():
            self.pin(path, digest, 'source_bindings')
        require(not {'numpy', 'torch', 'threadpoolctl'} & set(sys.modules), 'Immutable helpers remain inert')
        declarations = {node.targets[0].id: node.value for node in ast.parse(DATA.read_text()).body
                        if isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)}
        self.valid = tuple(ast.literal_eval(declarations['VALID']))
        self.train = tuple(r for r in range(35, 235) if r not in self.valid)
        require(len(self.train) == 160 and len(self.valid) == 40 and len(set(self.valid)) == 40,
                'Exact existing160/40 split')
        cpu = self.receipt(CO/'stage_cpu_forward_v1', CPU_SHA)
        old = self.receipt(CO/'stage_mps_forward_v1', FAILED_MPS_SHA, passed=False)
        for rec, stage in ((cpu, 'cpu_forward'), (old, 'mps_forward')):
            require(rec['schema'] == 'current_d50_p_coadaptation_qualification_v1'
                    and rec['stage'] == stage and rec['alltruth_accessed'] is False
                    and rec['training_targets_accessed'] is False and rec['validation_targets_accessed'] is False
                    and rec['no_fit'] and rec['parameter_updates'] == 0
                    and rec['versions'] == VERSIONS
                    and rec['source_bindings'][str(CO/'model.py')] == MODEL_SHA, 'Old scope, not new admission')
        require('Actual-FP32 finite-direction backend gate' in old['error'], 'Preserve the actual CO failure')
        self.cotangents, self.payload = cpu['cotangents'], cpu['payloads']['null']
        self.reference35 = old['outputs']['35']['null']
        self.parent35 = old['parent_outputs']['35']
        require(self.payload['sha256'] == PAYLOAD_SHA and self.reference35['parent_byte_exact']
                and self.reference35['prediction']['raw_sha256'] == self.parent35['raw_sha256'],
                'Exact original d50 payload and old same-backend35 control')
        self.record.update(old_cpu_result_sha256=CPU_SHA, old_failed_mps_result_sha256=FAILED_MPS_SHA,
                           model_sha256=MODEL_SHA, train_ranks=list(self.train), validation_ranks=list(self.valid))
        if args.stage == 'train':
            qual = self.receipt(HERE/'qualification_v1', args.qualification_result_sha256)
            watch = self.receipt(HERE/'qualification_watch_v1', args.qualification_watch_sha256)
            require(qual['schema'] == SCHEMA and qual['stage'] == 'qualify'
                    and qual['hashes_unchanged'] and qual['sources_unchanged']
                    and qual['execution_integrity_passed'] and qual['training_targets_accessed'] is False
                    and qual['validation_targets_accessed'] is False and qual['no_native_gradient_claim']
                    and len(qual['graph_records']) == 4, 'New target-free integrity prerequisite')
            require(watch['schema'] == 'current_d50_native_objective_p1_watch_v1'
                    and watch['stage'] == 'qualify' and watch['hashes_unchanged'] and watch['sources_unchanged']
                    and watch['child_execution_completed'] and watch['whole_owned_tree_retired']
                    and watch['completed_graph_count'] == 4
                    and watch['child_result_sha256'] == args.qualification_result_sha256
                    and watch['child_completion_sha256'] == self.pin(HERE/'qualification_v1/completion.json'),
                    'Independent completed qualification resource seal')
            for record in (qual, watch):
                for path, digest in ((self.worker_path, args.source_sha256),
                    (HERE/'common.py', args.common_sha256), (HERE/'watch.py', args.watcher_sha256),
                    (HERE/'PROTOCOL.md', PROTOCOL_SHA)):
                    require(record['source_bindings'].get(str(path)) == digest, 'Same reviewed new sources')
            require(qual['environment']['versions'] == VERSIONS, 'Qualified library frame')
            self.record['prerequisites'] = dict(qualification_result_sha256=args.qualification_result_sha256,
                                               qualification_watch_sha256=args.qualification_watch_sha256)
        self.weights = self.weight_rows((35, 100) if args.stage == 'qualify' else range(35, 235))
        if args.stage == 'train':
            self.costs = self.capture_costs()
            self.pin(INDEX, 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971')
            self.targets = {x['rank']: x['target_sha256'] for x in read(INDEX)['target_sources']}
            require(set(self.targets) == set(self.train), 'TRAIN-only pinned1e9 label inventory')
        self.record['environment'] = dict(executable=sys.executable, nice=os.getpriority(os.PRIO_PROCESS, 0),
                                          launch={k: os.environ.get(k) for k in BLAS})
        self.json('admission.json', dict(source_bindings=self.record['source_bindings'],
            input_bindings=self.record['input_bindings'], weight_bindings=self.record['weight_bindings'],
            stage=args.stage, training_targets_accessed=False, validation_targets_accessed=False))

    def weight_rows(self, ranks):
        self.pin(STORE/'manifest.json', '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50')
        entries = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
        result = {}
        for rank in ranks:
            matches = [row for row in entries if row['role_rank'] == rank]
            require(len(matches) == 1, 'Unique canonical weight row')
            path = (STORE/matches[0]['path']).resolve()
            require(path.is_relative_to(STORE), 'Weight row scope')
            self.pin(path, matches[0]['sha256'], 'weight_bindings')
            row = read(path)
            require(row['role_rank'] == rank and row['role'] == 'observer_fit', 'Weight row identity')
            wp = path.parent/'weights.npy'
            digest = self.pin(wp, row['files']['weights.npy']['sha256'], 'weight_bindings')
            result[rank] = dict(path=str(wp), sha256=digest)
        return result

    def capture_costs(self):
        path = TL/'capture_train_v1/manifest.json'
        self.pin(path, CAPTURE_SHA)
        manifest = read(path)
        require(manifest['completed'] and manifest['target_accessed'] is False
                and manifest['candidate_sha256'] == BANK_SHA and manifest['train_ranks'] == list(self.train)
                and set(manifest['captures']) == {str(r) for r in self.train}, 'Qualified d50 cost inventory')
        for source, digest in manifest['source_bindings'].items():
            self.pin(source, digest)
        watch = self.receipt(TL/'capture_train_v1', CAPTURE_WATCH_SHA)
        require(watch['hashes_unchanged'] and watch['target_accessed'] is False
                and watch['candidate_sha256'] == BANK_SHA and watch['artifacts'][str(path)] == CAPTURE_SHA,
                'Qualified full-capture supervisor')
        costs = {}
        for rank in self.train:
            item = manifest['captures'][str(rank)]
            rp = Path(item['path']).resolve()
            require(rp.is_relative_to(TL) and rp.name == 'result.json', 'Fixed capture result ownership')
            self.pin(rp, item['sha256']); self.pin(rp.parent/'completion.json')
            record, end = read(rp), read(rp.parent/'completion.json')
            require(record['passed'] and record['completed'] and record['hashes_unchanged']
                    and record['sources_unchanged'] and end['passed'] and end['completed']
                    and end['result_sha256'] == item['sha256'] and record['rank'] == rank
                    and record['candidate_sha256'] == BANK_SHA and record['truth_accessed'] is False
                    and record['network_calls'] == 1 and record['cold'] and record['all_primary_cold']
                    and record['includes_initialization'] and record['initialization_subtracted'] is False
                    and record['input_preparation_in_participant_bill'] is False, 'Own complete cold d50 bill')
            for source in (TL/'native.py', TL/'common.py', TL/'PROTOCOL.md'):
                require(record['source_bindings'][str(source)] == manifest['source_bindings'][str(source)],
                        'Cost capture uses the qualified source')
            require(record['weights']['path'] == self.weights[rank]['path']
                    and record['weights']['sha256'] == self.weights[rank]['sha256'], 'Cost/gradient weight identity')
            bill = record['complete_bill']
            meta = record['bill']
            require(Path(meta['path']).resolve() == rp.parent/'bill.json', 'Owned complete bill path')
            self.pin(meta['path'], meta['sha256'])
            require(read(meta['path']) == bill and type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET
                    and bill['cold'] and bill['includes_initialization'] and bill['observed_predict_calls'] == 1
                    and bill['wall_s'] < 120 and 0 < bill['peak_rss_bytes'] < 8_000_000_000,
                    'Qualified actual cold metric')
            costs[rank] = dict(flops=bill['flops'], multiplier=max(.1, bill['flops']/BUDGET),
                               capture_result=item, bill=meta)
        mean = math.fsum(costs[r]['multiplier'] for r in self.train)/len(self.train)
        for item in costs.values():
            item['normalized_weight'] = item['multiplier']/mean
        self.record['cost_weights'] = self.json('cost_weights.json', dict(capture_manifest_sha256=CAPTURE_SHA,
            training_mean_multiplier=mean, weights={str(k): v for k, v in costs.items()},
            changed_candidate_cost_predicted=False, capture_features_decoded=False))
        return costs

    def json(self, name, value):
        path = self.folder/name
        write(path, value)
        digest = sha(path)
        self.record['artifacts'][str(path)] = digest
        return dict(path=str(path), sha256=digest)

    def save(self, np, name, value):
        path = self.folder/name
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        digest = sha(path)
        self.record['artifacts'][str(path)] = digest
        return dict(path=str(path), sha256=digest, **self.h.array_meta(value))

    def bundle(self, np, name, values):
        path = self.folder/name
        with path.open('xb') as stream:
            np.savez(stream, **values)
        digest = sha(path)
        self.record['artifacts'][str(path)] = digest
        return dict(path=str(path), sha256=digest,
                    arrays={key: self.h.array_meta(value) for key, value in values.items()})

    def guard(self):
        elapsed = time.monotonic()-self.start
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        driver = int(self.torch.mps.driver_allocated_memory()) if self.torch is not None else 0
        self.record['maximum_sampled_mps_driver_bytes'] = max(self.record['maximum_sampled_mps_driver_bytes'], driver)
        require(elapsed < (300 if self.args.stage == 'qualify' else 1800), 'Fixed stage wall bound')
        require(rss < 12_000_000_000 and driver < 16_000_000_000, 'Child RSS/MPS driver bound')
        return dict(elapsed_s=elapsed, peak_rss_bytes=rss, mps_driver_bytes=driver)

    def graph_complete(self, index, item):
        require(index == len(self.record['graph_records']), 'Monotone retired model-call completion')
        self.record['graph_records'].append(self.json(f'graph_{index:04d}_complete.json', dict(item,
            graph_index=index, passed=True, completed=True, graph_retired=True,
            elapsed_s=time.monotonic()-self.start, resources=self.guard())))

    def finish(self):
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'target_bindings', 'artifacts'):
            for path, digest in self.record[group].items():
                require(sha(path) == digest, 'Final hash changed: '+path)
        for imported in tuple(sys.modules.values()):
            name = getattr(imported, '__file__', None)
            if name:
                path = Path(name).resolve()
                if path.is_file() and path.suffix == '.py' and path.is_relative_to(NEW):
                    require(self.record['source_bindings'].get(str(path)) == sha(path), 'Unbound local import: '+str(path))
        require('threadpoolctl' not in sys.modules, 'AP must not import threadpoolctl')
        self.record.update(resources=self.guard(), hashes_unchanged=True, sources_unchanged=True,
                           passed=True, completed=True)

    def close(self):
        self.record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                           wall_s=time.monotonic()-self.start, graphs_completed=len(self.record['graph_records']))
        write(self.folder/'result.json', self.record)
        write(self.folder/'completion.json', dict(schema=SCHEMA, stage=self.args.stage,
            passed=self.record['passed'], completed=self.record['completed'],
            result_sha256=sha(self.folder/'result.json')))
