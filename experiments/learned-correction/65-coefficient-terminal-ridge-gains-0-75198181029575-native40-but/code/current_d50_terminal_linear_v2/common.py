"""Inert, target-free IO for one cold native terminal-feature capture.

Cross-worker CAP-failure and initial35/100/repeat admission belong to the
separately reviewed supervisor. This module never imports a numerical library.
"""
import argparse
import ast
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
BD, FQ = NEW/'birth_direction_fit_v1', NEW/'current_d50_full_return_capture_v1'
OFFPY = P2/'data_gate/.venv'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
CANDIDATE = BD/'build_terminal_v1/estimator.py'
CANDIDATE_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
PAYLOAD = NEW/'payload_response_rows_v1.npz'
PAYLOAD_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
PROTOCOL_SHA = 'febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58'
LEGACY_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
TEMPLATE_SHA = '740d2a67eb1e0f9c568ba2c0fec0d52e1335d83e6e5b4b906d126c14f4baf2b4'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
TRAIN = tuple(r for r in range(35,235) if r not in VALID)
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
REFERENCES = {35:'4781b3200702ea2feee059630118cdf38e7b57a24f134ab7fe1ced331ccf09ec',
              100:'7c8e89a6e04aa2afb80bb0d185a9d1fdef4af24cdc3677e7a0e57e32cfb45314'}
FIXED = {
    CANDIDATE:CANDIDATE_SHA, PAYLOAD:PAYLOAD_SHA,
    BD/'native_v2.py':LEGACY_SHA, FQ/'native.py':TEMPLATE_SHA,
    BD/'build_terminal_v1/build.json':'4889c6246e8e0dd420cdb8fbca3ccbce721a163befd226339ebe6d0023cc070f',
    NEW/'joint_prefix_final_data_v1.py':'22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b',
    STORE/'manifest.json':'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
            'No truth-array read or hash')
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
    require(spec is not None and spec.loader is not None, 'Import specification')
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def arguments():
    parser = argparse.ArgumentParser(description='One cold current-d50 terminal-feature capture; no targets/fit')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--rank', type=int, choices=TRAIN)
    parser.add_argument('--output', type=Path)
    for name in ('source-sha256','common-sha256','protocol-sha256'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
    return args


def array_meta(value):
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())


class Stage:
    def __init__(self, args, source):
        require(args.rank in TRAIN and args.output is not None, 'Fixed TRAIN rank/output required')
        require(all(x and len(x) == 64 for x in
                    (args.source_sha256,args.common_sha256,args.protocol_sha256)), 'Reviewed hashes required')
        require(sha(source) == args.source_sha256 and sha(__file__) == args.common_sha256
                and args.protocol_sha256 == PROTOCOL_SHA and sha(HERE/'PROTOCOL.md') == PROTOCOL_SHA,
                'Reviewed source/protocol changed')
        require(Path(sys.prefix).resolve() == OFFPY.resolve() and sys.flags.optimize == 0
                and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'OFFPY/assertions/no bytecode')
        require(os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in BLAS)
                and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') == '0', 'nice10/BLAS1/no fallback')
        require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),
                'Authenticate before numerical imports')
        self.folder = args.output.resolve()
        require(self.folder != HERE and self.folder.is_relative_to(HERE) and not self.folder.exists(),
                'New exclusive output directory strictly inside this experiment')
        self.folder.mkdir(parents=True)
        self.started, self.rank = time.monotonic(), args.rank
        self.result = dict(schema='current_d50_terminal_linear_capture_v1',role='native',rank=args.rank,
            passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
            started_utc=datetime.now(timezone.utc).isoformat(),truth_accessed=False,
            weights_read=False,parameter_updates=0,gradient_calls=0,network_calls=0,
            training_admitted=False,score_measured=False,MPS_used=False,
            captures={},bindings={},source_bindings={},artifacts={})
        self.bindings = self.result['bindings']
        self.old_handler = signal.signal(signal.SIGALRM,self.guard)
        signal.setitimer(signal.ITIMER_REAL,.5,.5)
        write(self.folder/'started.json',self.result)
        try:
            self.pin(source,args.source_sha256)
            self.pin(__file__,args.common_sha256)
            self.pin(HERE/'PROTOCOL.md',PROTOCOL_SHA,source=True)
            for path,digest in FIXED.items():
                self.pin(path,digest)
            build = json.loads((BD/'build_terminal_v1/build.json').read_text())
            require(build['passed'] and build['inverse_source_byte_exact']
                    and build['all_other_parent_source_byte_exact']
                    and build['imports_and_learned_P_payloads_unchanged']
                    and build['candidate_sha256'] == CANDIDATE_SHA, 'Frozen d50 build')
            self.inventory(build)
            self.legacy = module(BD/'native_v2.py','terminal_linear_legacy_native')
            for path,digest in self.legacy.PINS.items():
                self.pin(path,digest)
            self.template = module(FQ/'native.py','terminal_linear_frozen_capture_template')
            split = ast.parse((NEW/'joint_prefix_final_data_v1.py').read_text())
            nodes = [n.value for n in split.body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id == 'VALID' for t in n.targets)]
            require(len(nodes) == 1 and ast.literal_eval(nodes[0]) == VALID
                    and len(TRAIN) == 160 and len(VALID) == 40, 'Original fixed160/40 split')
            entries = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
            matches = [x for x in entries if x['role_rank'] == args.rank]
            require(len(matches) == 1, 'Unique canonical TRAIN weight row')
            row_path = (STORE/matches[0]['path']).resolve()
            self.pin(row_path,matches[0]['sha256'])
            row = json.loads(row_path.read_text())
            self.weights_path = row_path.parent/'weights.npy'
            weight = self.pin(self.weights_path,row['files']['weights.npy']['sha256'])
            require(row['role_rank'] == args.rank, 'Exact rank ownership')
            self.mlp_seed = int(row['mlp_seed'])
            self.candidate = CANDIDATE
            self.result.update(weights=dict(path=str(self.weights_path),sha256=weight['sha256'],
                mlp_seed=self.mlp_seed),payload=self.bindings[str(PAYLOAD)],
                candidate=str(CANDIDATE),candidate_sha256=CANDIDATE_SHA,
                original_build_sha256=FIXED[BD/'build_terminal_v1/build.json'],
                initial_reference_required=args.rank in REFERENCES,
                cross_worker_admission='CAP failure and35/100/repeat seal checked by supervisor')
            self.reference = self.fq_reference(args.rank) if args.rank in REFERENCES else None
            self.guard()
        except BaseException:
            self.result['error'] = traceback.format_exc()
            self.close()
            raise

    def guard(self,*_):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != 'darwin':
            rss *= 1024
        elapsed = time.monotonic()-self.started
        self.result['resources'] = dict(peak_rss_bytes=rss,elapsed_s=elapsed)
        require(rss < 8_000_000_000 and elapsed < 120, 'Whole cold worker8GB/120s')

    def pin(self,path,expected=None,source=False):
        path = Path(path)
        path = path.resolve() if path.is_absolute() else (NEW/path).resolve()
        require(path.is_relative_to(P2), 'Input outside Phase2')
        before = path.stat()
        digest = sha(path)
        require((expected is None or digest == expected) and path.stat().st_mtime_ns == before.st_mtime_ns,
                'Changed source/input/artifact: '+str(path))
        item = dict(path=str(path),sha256=digest,bytes=before.st_size,mtime_ns=before.st_mtime_ns)
        require(str(path) not in self.bindings or self.bindings[str(path)] == item, 'Conflicting pin')
        self.bindings[str(path)] = item
        if source or path.suffix == '.py':
            self.result['source_bindings'][str(path)] = digest
        return item

    def inventory(self,record):
        for group in ('source_bindings','artifacts'):
            for path,value in record.get(group,{}).items():
                self.pin(value['path'],value['sha256']) if isinstance(value,dict) else self.pin(path,value)

    def fq_reference(self,rank):
        folder = FQ/f'native_null_r{rank:04d}'
        self.pin(folder/'result.json',REFERENCES[rank])
        self.pin(folder/'completion.json')
        record,seal = (json.loads((folder/name).read_text()) for name in ('result.json','completion.json'))
        require(record['passed'] and record['completed'] and record['hashes_unchanged']
                and record['sources_unchanged'] and seal['passed'] and seal['completed']
                and seal['result_sha256'] == REFERENCES[rank]
                and record['schema'] == 'current_d50_full_return_capture_v1'
                and (record['role'],record['case'],record['rank']) == ('native','null',rank)
                and record['candidate_sha256'] == CANDIDATE_SHA and not record['truth_accessed']
                and record['complete_bill']['cold'] and record['complete_bill']['includes_initialization']
                and record['observed_predict_calls'] == 1, 'Successful original cold native capture')
        require(record['weights_path'] == str(self.weights_path) and record['mlp_seed'] == self.mlp_seed,
                'Reference same canonical network')
        self.inventory(record)
        for key in ('served','raw_unclamped','bill','input_bill','lifecycle_receipt','input_lifecycle_receipt'):
            self.pin(record[key]['path'],record[key]['sha256'])
        require(json.loads(Path(record['bill']['path']).read_text()) == record['complete_bill'],
                'Reference complete bill content')
        self.result['reference_result'] = dict(path=str(folder/'result.json'),sha256=REFERENCES[rank])
        return record

    def load(self,np,item):
        self.pin(item['path'],item['sha256'])
        value = np.load(item['path'],allow_pickle=False)
        meta = array_meta(value)
        require(all(item.get(k,v) == v for k,v in meta.items()) and np.isfinite(value).all(), 'Array metadata/bytes')
        return value

    def save(self,np,name,value):
        require(Path(name).name == name and name.endswith('.npy') and np.isfinite(value).all(), 'Finite named capture')
        path = self.folder/name
        with path.open('xb') as stream:
            np.save(stream,value,allow_pickle=False)
        item = self.pin(path)
        self.result['artifacts'][str(path)] = item['sha256']
        return dict(item,**array_meta(value))

    def finish(self):
        for name,value in list(sys.modules.items()):
            raw = getattr(value,'__file__',None)
            if raw:
                path = Path(raw).resolve()
                if path.is_file() and path.suffix == '.py' and path.is_relative_to(NEW):
                    require(str(path) in self.bindings, 'Unbound local import: '+name)
        for path,item in list(self.bindings.items()):
            self.pin(path,item['sha256'])
        self.guard()
        self.result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)

    def close(self):
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,self.old_handler)
        self.result.update(elapsed_s=time.monotonic()-self.started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(self.folder/'result.json',self.result)
        write(self.folder/'completion.json',dict(passed=self.result['passed'],completed=self.result['completed'],
            result_sha256=sha(self.folder/'result.json')))
        print(json.dumps({k:self.result.get(k) for k in ('rank','passed','completed','elapsed_s','error')}),flush=True)
