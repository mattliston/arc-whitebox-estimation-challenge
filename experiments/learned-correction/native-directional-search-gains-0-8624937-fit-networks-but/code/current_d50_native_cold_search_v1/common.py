"""Stdlib-only admission, exclusive IO, and provenance for the cold ND search.

Importing this module performs no experiment and imports no numerical library.
The reviewed admission file binds this module and every executable ND source.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
SCHEMA = 'current_d50_native_cold_search_v1'
PROTOCOL_SHA = '07339b5cd9ee06770145e36528a751d12e419ed514bcfba4d0465662a4368f8d'
FIT, CHECK = (35, 36, 38, 40), (41, 43, 44, 45)
P0_PATH = NEW / 'payload_response_rows_v1.npz'
P0_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
INDEX_PATH = NEW / 'joint_prefix_final_fit_v1/manifest.json'
INDEX_SHA = 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
WORKER_PATH = NEW / 'current_d50_cpu_pilot_v1/worker.py'
WORKER_SHA = '16e4762e49d94401041946a0cebc4b69dbfe9ae13c7c875e1834eeb73f3a46d1'
AP = Path('/Users/matt/anaconda3/bin/python')
OFFPY = P2 / 'data_gate/.venv/bin/python'
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
SOURCES = ('common.py', 'cpu.py', 'native.py', 'interpolate.py', 'judge.py', 'watch.py', 'PROTOCOL.md')
GROUPS = ('source_bindings', 'input_bindings', 'target_bindings', 'artifacts')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def array_meta(array):
    return dict(dtype=str(array.dtype), shape=list(array.shape),
                raw_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def native_prerequisite(stage, folder, digest, case, payload, lineage):
    record = stage.receipt(folder, digest, 'native', case)
    roster = FIT + CHECK + (100,) if case == 'baseline' else (FIT if case in ('plus', 'minus') else FIT + CHECK)
    sequence = ((35, 100, 35) + FIT[1:] + CHECK) if case == 'baseline' else (
        (35, 35) + FIT[1:] + (CHECK if case == 'candidate' else ()))
    require([x['rank'] for x in record['trials']] == list(sequence)
            and set(record['outputs']) == {str(r) for r in roster}
            and record['payload'] == payload
            and record['prerequisites'] == lineage
            and record['initialization_flops'] == 1383913786
            and all(record[k] is True for k in ('all_domain_checks_passed', 'all_selector_checks_passed',
                'repeat_byte_exact', 'repeat_diagnostics_byte_exact', 'own_complete_bills', 'all_primary_cold'))
            and record['truth_accessed'] is False and record['alltruth_accessed'] is False
            and record['check_targets_accessed'] is False and record['validation_targets_accessed'] is False,
            'Exact native payload/roster/bill/domain/target lineage: ' + case)
    require(case != 'baseline' or record['baseline_reference_passed'] is True,
            'Baseline exact native reference')
    return record


def truths(np, stage, ranks):
    ranks = tuple(ranks)
    infos = {rank: stage.target(rank, ranks) for rank in ranks}
    stage.attempt('before_target_array_access.json', dict(
        target_bindings=dict(stage.record['target_bindings']), ranks=list(ranks),
        prerequisites=stage.record['prerequisites'],
        predictions_and_complete_bills_already_sealed=True))
    values = {}
    for rank in ranks:
        stage.record['training_targets_accessed'] = True
        stage.record['check_targets_accessed'] = rank in CHECK or stage.record['check_targets_accessed']
        stage.record.setdefault('target_ranks_accessed', []).append(rank)
        with np.load(infos[rank]['path'], allow_pickle=False) as stream:
            value = np.asarray(stream['mc_post'], dtype=np.float64).copy()
        require(value.shape == (16, 1024) and np.isfinite(value).all(), 'Finite fixed 1e9 TRAIN target')
        stage.pin(infos[rank]['path'], infos[rank]['sha256'], 'target_bindings')
        values[rank] = value
    return values


def score(np, stage, native, targets, ranks):
    rows = []
    for rank in ranks:
        info = native['outputs'][str(rank)]
        prediction = stage.load_array(np, info)
        require(prediction.dtype == np.float32 and prediction.shape == (16, 1024)
                and np.isfinite(prediction).all() and (prediction >= 0).all(), 'All16 served native predictions')
        flops = info['effective_flops']
        require(type(flops) is int and 0 < flops <= 2**41, 'Own complete positive integer FLOP bill within budget')
        squared = (prediction.astype(np.float64) - targets[rank]) ** 2
        layer = squared.mean(axis=1)
        require(np.isfinite(layer).all(), 'Finite actual native raw errors')
        raw = float(layer[-1])
        multiplier = max(.1, flops / 2**41)
        rows.append(dict(rank=rank, raw_final_mse=raw, effective_flops=flops,
                         multiplier=multiplier, adjusted=raw * multiplier,
                         layer_mse=layer.tolist(), output=info))
    adjusted = float(np.mean([r['adjusted'] for r in rows], dtype=np.float64))
    require(np.isfinite(adjusted) and adjusted >= 0, 'Finite nonnegative actual native objective')
    return dict(ranks=list(ranks), per_net=rows, mean_adjusted=adjusted,
                mean_raw_final_mse=float(np.mean([r['raw_final_mse'] for r in rows], dtype=np.float64)),
                mean_complete_flops=float(np.mean([r['effective_flops'] for r in rows], dtype=np.float64)),
                objective='mean(per_net_native_raw * max(.1, own_complete_COLD_F/2^41))',
                evidence='native predictions against 1e9-frame TRAIN truth; not independent14 or official')


class Stage:
    def __init__(self, source, admission_sha256, stage, directory):
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'Admission precedes all numerical imports')
        admission = HERE / 'ADMISSION.json'
        require(isinstance(admission_sha256, str) and len(admission_sha256) == 64
                and sha(admission) == admission_sha256, 'Exact reviewed admission required')
        record = read(admission)
        expected = {str(HERE / name) for name in SOURCES}
        require(record.get('schema') == SCHEMA + '_admission'
                and record.get('reviewed') is True and record.get('numerical_work_begun') is False
                and set(record.get('source_bindings', {})) == expected,
                'Admission owns precisely the reviewed ND source roster')
        require(record['source_bindings'][str(HERE / 'PROTOCOL.md')] == PROTOCOL_SHA,
                'Protocol compiled into new code before seal')
        require(str(Path(source).resolve()) in expected, 'Admitted stage source')
        for path, digest in record['source_bindings'].items():
            require(sha(path) == digest, 'Reviewed source changed: ' + path)
        self.directory = Path(directory)
        if not self.directory.is_absolute():
            self.directory = HERE / self.directory
        require(self.directory.parent.resolve() == HERE, 'Stage output is a direct ND child')
        self.directory.mkdir(exist_ok=False)
        self.start = time.monotonic()
        self.closed = False
        self.record = dict(schema=SCHEMA, stage=stage, passed=False, completed=False,
            hashes_unchanged=False, scientific=False, admission_sha256=admission_sha256,
            source_bindings=dict(record['source_bindings']), input_bindings={},
            target_bindings={}, artifacts={}, training_targets_accessed=False,
            validation_targets_accessed=False, check_targets_accessed=False,
            native_gradient_qualified=False, MPS_used=False,
            started_utc=datetime.now(timezone.utc).isoformat())
        self.pin(admission, admission_sha256, 'source_bindings')
        write(self.directory / 'started.json', self.record)

    def environment(self, kind):
        require(kind in ('AP', 'OFFPY'), 'Fixed Python environment')
        require(sys.flags.optimize == 0, 'Assertions enabled')
        require(os.path.samefile(sys.executable, AP if kind == 'AP' else OFFPY),
                'Wrong Python executable')
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 required')
        require(all(os.environ.get(k) == '1' for k in BLAS), 'Explicit BLAS1 environment')
        require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'No bytecode writes')
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No MPS fallback')
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'Environment checks precede numerical imports')
        self.record['python'] = dict(executable=sys.executable, version=sys.version, environment=kind)

    def pin(self, path, expected=None, group='input_bindings'):
        require(group in GROUPS, 'Known pin group')
        path = str(Path(path).resolve())
        target = 'truthpost' in Path(path).parts or (
            Path(path).suffix in ('.npz', '.npy') and 'truth' in Path(path).name.lower())
        require(not target or group == 'target_bindings', 'Targets have separate hash-only provenance')
        observed = sha(path)
        require(expected is None or observed == expected, 'Input changed: ' + path)
        previous = self.record[group].get(path)
        require(previous is None or previous == observed, 'Conflicting immutable pin: ' + path)
        self.record[group][path] = observed
        return observed

    def inventory(self, record, targets=False):
        for group in GROUPS:
            if group == 'target_bindings' and not targets:
                continue
            for path, digest in record.get(group, {}).items():
                self.pin(path, digest, group)
        for path, digest in record.get('artifact_bindings', {}).items():
            self.pin(path, digest, 'artifacts')

    def receipt(self, directory, digest, expected_stage, expected_case=None):
        folder = Path(directory)
        if not folder.is_absolute():
            folder = HERE / folder
        rp, cp = folder / 'result.json', folder / 'completion.json'
        self.pin(rp, digest)
        self.pin(cp)
        record, completion = read(rp), read(cp)
        require(record.get('schema') == SCHEMA and record.get('stage') == expected_stage
                and record.get('completed') is True and record.get('passed') is True
                and record.get('hashes_unchanged') is True and not record.get('error')
                and completion.get('completed') is True and completion.get('passed') is True
                and completion.get('result_sha256') == digest
                and record.get('admission_sha256') == self.record['admission_sha256'],
                'Complete PASS receipt with same admission required: ' + str(folder))
        if expected_case is not None:
            require(record.get('case') == expected_case, 'Exact prerequisite native case')
        self.inventory(record, targets=False)
        return record

    def save(self, np, name, array):
        require(Path(name).name == name and name.endswith('.npy'), 'Flat exclusive NPY artifact name')
        path = self.directory / name
        require(not array.dtype.hasobject, 'No object array artifacts')
        with path.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        return dict(path=str(path), sha256=self.pin(path, group='artifacts'), **array_meta(array))

    def bundle(self, np, name, values):
        require(Path(name).name == name and name.endswith('.npz') and values,
                'Flat exclusive NPZ artifact name')
        require(all(isinstance(k, str) and not a.dtype.hasobject for k, a in values.items()),
                'Named non-object arrays only')
        path = self.directory / name
        with path.open('xb') as stream:
            np.savez(stream, **values)
        return dict(path=str(path), sha256=self.pin(path, group='artifacts'),
                    arrays={k: array_meta(a) for k, a in values.items()})

    def load_array(self, np, info):
        self.pin(info['path'], info['sha256'])
        value = np.load(info['path'], allow_pickle=False)
        require(array_meta(value) == {k: info[k] for k in ('dtype', 'shape', 'raw_sha256')},
                'Array file, shape, dtype and raw bytes all bound')
        return value

    def load_bundle(self, np, info):
        self.pin(info['path'], info['sha256'])
        with np.load(info['path'], allow_pickle=False) as stream:
            require(set(stream.files) == set(info['arrays']), 'Bundle exact key set')
            values = {k: stream[k].copy() for k in stream.files}
        require(all(array_meta(a) == info['arrays'][k] for k, a in values.items()),
                'Bundle exact dtype/shape/raw bytes')
        return values

    def target(self, rank, allowed):
        allowed = tuple(allowed)
        permitted = {'propose': FIT, 'interpolate': FIT, 'judge': FIT + CHECK}
        require(self.record['stage'] in permitted and allowed == permitted[self.record['stage']]
                and rank in allowed, 'Only the fixed stage-specific target roster')
        self.pin(INDEX_PATH, INDEX_SHA)
        index = read(INDEX_PATH)
        entries = [x for x in index['target_sources'] if int(x['rank']) == rank]
        require(len(entries) == 1, 'Unique fixed TRAIN target provenance')
        path = P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz'
        digest = self.pin(path, entries[0]['target_sha256'], 'target_bindings')
        return dict(path=str(path), sha256=digest, rank=rank, numerical_array_access=False)

    def attempt(self, name, obj):
        require(Path(name).name == name and name.endswith('.json'), 'Flat exclusive attempt receipt')
        path = self.directory / name
        write(path, obj)
        self.pin(path, group='artifacts')
        return dict(path=str(path), sha256=self.record['artifacts'][str(path)])

    def recheck(self):
        for group in GROUPS:
            for path, digest in self.record[group].items():
                require(sha(path) == digest, 'Immutable provenance changed: ' + path)
        self.record['hashes_unchanged'] = True

    def finish(self, passed=True, scientific=False):
        require(passed or scientific, 'A complete failed result must be a scientific stop')
        self.recheck()
        self.record.update(passed=bool(passed), scientific=bool(scientific), completed=True)

    def close(self, final_guard=None):
        require(not self.closed, 'Stage closes exactly once')
        self.closed = True
        try:
            self.recheck()
            if final_guard is not None:
                final_guard()
        except BaseException:
            self.record.update(passed=False, completed=False, hashes_unchanged=False,
                               final_check_error=traceback.format_exc())
        self.record.update(elapsed_s=time.monotonic() - self.start,
                           finished_utc=datetime.now(timezone.utc).isoformat())
        write(self.directory / 'result.json', self.record)
        write(self.directory / 'completion.json', dict(completed=self.record['completed'],
            passed=self.record['passed'], result_sha256=sha(self.directory / 'result.json')))
        return 0 if self.record['passed'] else 1
