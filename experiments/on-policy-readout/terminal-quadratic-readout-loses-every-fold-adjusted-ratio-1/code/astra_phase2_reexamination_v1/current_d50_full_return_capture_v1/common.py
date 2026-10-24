"""Inert pinned IO for a fixed target-blind full-return capture, not a fit."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
CO = NEW/'current_bd_p_coadaptation_v1'
BR = NEW/'current_d50_native_p_bridge_v1'
BD = NEW/'birth_direction_fit_v1'
Q = NEW/'current_d50_return_quantization_v1'
RANKS = (35, 100)
CASES = ('null','plus025','minus025','plus050','minus050')
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
HELPER_SHA = '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20'
MODEL_SHA = '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0'
CPU_SHA = '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607'
BUILD_SHA = '810a406585ce746f491ce9edcd27c2fa3d56d1d2881035b86fbe50470ea4126e'
NATIVE_SHAS = dict(null='039c98a5acffa4e88440ef331b7efafd78b4c56163043dcbc670c0eedc1cd333',
    plus025='3632107281976b7349d28e423d37e3760081e6868bdfa204f5d4b976503effb5',
    minus025='288ffb03f9ad5de23dcabea408ce57d9c4eae048fc06e2f0ed99060a2f9c6139',
    plus050='f3567dcd321ee989376b5a2cf36f7ffdacd54754bc4efd7bac6a578cab094dd9',
    minus050='baa4e9bc3f85403bfdb2fa4d9f9455a3a238eadcb12a247f8355276c1e7c606e')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
            'No truth array access or hashing')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def arguments(role):
    parser = argparse.ArgumentParser(description='Fixed full-return ' + role)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case', choices=CASES, default='null')
    parser.add_argument('--rank', type=int, choices=RANKS, default=35)
    for name in ('source-sha256','protocol-sha256','common-sha256'):
        parser.add_argument('--' + name)
    parser.add_argument('--manifest')
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
    return args


class Stage:
    def __init__(self, role, args, source):
        require(role in ('native','cpu','judge'), 'Fixed role')
        require(args.case in CASES and args.rank in RANKS, 'Fixed case/rank')
        require(all(x and len(x) == 64 for x in
                    (args.source_sha256,args.protocol_sha256,args.common_sha256)), 'Reviewed hashes required')
        require(sha(source) == args.source_sha256 and sha(__file__) == args.common_sha256
                and sha(HERE/'PROTOCOL.md') == args.protocol_sha256, 'Reviewed source changed')
        require(sys.flags.optimize == 0 and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Assertions/no bytecode')
        require(os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in BLAS)
                and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') == '0', 'nice10/BLAS1/no fallback')
        expected_prefix = P2/'data_gate/.venv' if role == 'native' else Path('/Users/matt/anaconda3')
        require(Path(sys.prefix).resolve() == expected_prefix.resolve(), 'Fixed interpreter')
        require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),
                'Pins before numerical imports')
        self.role, self.args = role, args
        self.started = time.monotonic()
        self.folder = HERE/(f'{role}_{args.case}_r{args.rank:04d}' if role != 'judge' else 'judgment_v1')
        self.folder.mkdir()
        self.result = dict(schema='current_d50_full_return_capture_v1',role=role,case=args.case,rank=args.rank,
            passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
            started_utc=datetime.now(timezone.utc).isoformat(),truth_accessed=False,
            parameter_updates=0,gradient_calls=0,network_calls=0,training_admitted=False,
            score_measured=False,MPS_used=False,original_derivative_gates_unchanged=True,
            bindings={},source_bindings={},artifacts={})
        self.bindings = self.result['bindings']
        self.old_handler = signal.signal(signal.SIGALRM,self.guard)
        signal.setitimer(signal.ITIMER_REAL,.5,.5)
        write(self.folder/'started.json',self.result)
        try:
            self.pin(source,args.source_sha256)
            self.pin(__file__,args.common_sha256)
            self.pin(HERE/'PROTOCOL.md',args.protocol_sha256,source=True)
            self.pin(CO/'qualify.py',HELPER_SHA)
            self.pin(CO/'model.py',MODEL_SHA)
            self.h = module(CO/'qualify.py','full_return_original_helpers')
            self.model = module(CO/'model.py','full_return_original_model')
            for path,digest in self.model.authenticate().items():
                self.pin(path,digest)
            self.old_cpu = self.sealed(CO/'stage_cpu_forward_v1',CPU_SHA,True)
            require(self.old_cpu['no_fit'] and self.old_cpu['parameter_updates'] == 0
                    and not self.old_cpu['alltruth_accessed'], 'Original target-blind CPU probes')
            self.old_native = self.old_native_case(args.case)
            self.reference_cpu = self.old_cpu['outputs'][str(args.rank)][args.case]
            matches = [x for x in self.old_native['trials'] if x['rank'] == args.rank]
            require(bool(matches), 'Original native cell')
            self.reference_native = matches[0]
            self.candidate = Path(self.old_native['source'])
            self.pin(BR/'build_probes_v1/build.json',BUILD_SHA)
            build = json.loads((BR/'build_probes_v1/build.json').read_text())
            require(build['passed'] and build['completed'] and build['no_fit']
                    and build['all_five_sources_complete'] and build['all_source_inverses_byte_exact']
                    and build['null_source_byte_exact'] and build['cstar_source_byte_exact']
                    and build['non_PKEYS_array_bytes_preserved'] and build['unused_layer0_array_bytes_preserved']
                    and build['parameter_updates'] == 0 and not build['targets_accessed']
                    and build['cpu_forward_result_sha256'] == CPU_SHA, 'Completed exact native probe build')
            self.inventory(build)
            item = build['candidates'][args.case]
            require(item['path'] == str(self.candidate) and item['sha256'] == self.old_native['candidate_sha256'],
                    'Exact old native candidate')
            require(item['payload_sha256'] == self.old_cpu['payloads'][args.case]['sha256']
                    and item['payload_path'] == self.old_cpu['payloads'][args.case]['path'], 'Same FP32 probe')
            self.pin(self.candidate,item['sha256'])
            paths, weights = self.h.weight_bindings()
            for path,info in weights.items():
                self.pin(path,info['sha256'])
            self.weights_path = paths[args.rank]
            # Use the exact already-authenticated STORE row, not an inferred seed.
            candidates = [Path(path) for path in weights if Path(path).suffix == '.json'
                          and Path(path).parent == self.weights_path.parent]
            require(len(candidates) == 1, 'Unique authenticated weight row')
            row = json.loads(candidates[0].read_text())
            require(row['role_rank'] == args.rank and row['files']['weights.npy']['sha256'] == sha(self.weights_path),
                    'Canonical rank/weight binding')
            self.mlp_seed = int(row['mlp_seed'])
            self.result.update(candidate=str(self.candidate),candidate_sha256=item['sha256'],
                weights_path=str(self.weights_path),mlp_seed=self.mlp_seed,
                old_cpu_sha256=CPU_SHA,old_native_sha256=NATIVE_SHAS[args.case],
                payload=self.old_cpu['payloads'][args.case],cotangent=self.old_cpu['cotangents'][str(args.rank)])
            self.guard()
        except BaseException:
            self.result['error'] = traceback.format_exc()
            self.close()
            raise

    def guard(self,*_):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != 'darwin':
            rss *= 1024
        self.result['resources'] = dict(peak_rss_bytes=rss,elapsed_s=time.monotonic()-self.started)
        require(rss < 8_000_000_000 and time.monotonic()-self.started < 120, 'Whole worker8GB/120s cap')

    def pin(self,path,expected=None,source=False):
        path = Path(path).resolve()
        require(path.is_relative_to(P2), 'Input outside Phase2')
        stat = path.stat()
        digest = sha(path)
        require(expected is None or digest == expected, 'Changed pin: ' + str(path))
        require(path.stat().st_mtime_ns == stat.st_mtime_ns, 'Changed during hashing')
        item = dict(path=str(path),sha256=digest,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        require(str(path) not in self.bindings or self.bindings[str(path)] == item, 'Conflicting/modified input')
        self.bindings[str(path)] = item
        if source or path.suffix == '.py':
            self.result['source_bindings'][str(path)] = digest
        return item

    def inventory(self,record):
        for group in ('source_bindings','input_bindings','artifacts','source_metadata',
                      'artifact_bindings','weight_bindings','dependencies','bindings'):
            for path,value in record.get(group,{}).items():
                if isinstance(value,dict):
                    require(value.get('path',path) == path, 'Mapping owner')
                    self.pin(path,value['sha256'])
                else:
                    self.pin(path,value)

    def sealed(self,folder,expected,completed):
        rp,cp = Path(folder)/'result.json',Path(folder)/'completion.json'
        self.pin(rp,expected)
        self.pin(cp)
        record,seal = json.loads(rp.read_text()),json.loads(cp.read_text())
        require(record['passed'] and seal['passed'] and seal['result_sha256'] == expected
                and 'error' not in record, 'Successful sealed acquisition')
        if completed:
            require(record['completed'] and seal['completed'], 'Explicit completion required')
        self.inventory(record)
        return record

    def old_native_case(self,case):
        record = self.sealed(BR/('run_'+case+'_v1'),NATIVE_SHAS[case],False)
        require(record['hashes_unchanged'] and not record['truth_accessed']
                and [x['rank'] for x in record['trials']] == [35,100,35]
                and record['initialization_flops'] == 1383913786,
                'Original native complete/repeat/input lifecycle')
        require(record['trials'][0]['output_bytes_sha256'] == record['trials'][2]['output_bytes_sha256'],
                'Old native repeat bytes')
        return record

    def load(self,np,item):
        path = str(Path(item['path']).resolve())
        require(path in self.bindings and self.bindings[path]['sha256'] == item['sha256'], 'Presealed array')
        self.pin(path,item['sha256'])
        value = np.load(path,allow_pickle=False)
        meta = self.h.array_meta(value)
        require(all(k not in item or item[k] == v for k,v in meta.items()) and np.isfinite(value).all(),
                'Array bytes/shape/dtype')
        return value

    def save(self,np,name,value):
        require(Path(name).name == name and np.isfinite(value).all(), 'Finite named output')
        path = self.folder/name
        with path.open('xb') as stream:
            np.save(stream,value,allow_pickle=False)
        item = self.pin(path)
        self.result['artifacts'][str(path)] = item['sha256']
        return dict(item,**self.h.array_meta(value))

    def finish(self,passed=True):
        local_imports = {}
        for name,value in list(sys.modules.items()):
            source = getattr(value,'__file__',None)
            if not source:
                continue
            path = Path(source).resolve()
            if path.is_relative_to(P2) and not path.is_relative_to(Path(sys.prefix).resolve()):
                require(str(path) in self.bindings, 'Unbound local import: ' + str(path))
                local_imports[name] = str(path)
        self.result['local_imports'] = local_imports
        for path,item in list(self.bindings.items()):
            self.pin(path,item['sha256'])
        self.guard()
        self.result.update(passed=bool(passed),completed=True,hashes_unchanged=True,sources_unchanged=True)

    def close(self):
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,self.old_handler)
        self.result.update(elapsed_s=time.monotonic()-self.started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(self.folder/'result.json',self.result)
        write(self.folder/'completion.json',dict(passed=self.result['passed'],completed=self.result['completed'],
                                                 result_sha256=sha(self.folder/'result.json')))
        print(json.dumps({k:self.result.get(k) for k in ('role','case','rank','passed','completed','elapsed_s','error')}),flush=True)
