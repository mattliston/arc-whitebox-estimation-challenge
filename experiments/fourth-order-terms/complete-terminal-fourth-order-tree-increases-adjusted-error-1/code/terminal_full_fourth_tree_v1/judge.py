"""Inert fixed two-TRAIN raw-times-own-cold-cost judge; no model or fitting.

All five native captures/bills, the repeated enabled35, and the prior source/
resource seals are authenticated before only TRAIN35/100 truths are opened.
A complete scientific FAIL exits0 but has passed=false; no tuning is admitted.
Use the separately reviewed external resource supervisor at execution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
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
import types

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent,HERE.parent.parent
OFFPY = P2/'data_gate/.venv'
PROTOCOL_SHA = 'f4bf2907fa1973b6076d6f485cd2436d6b5bf13cfe26077653e52af3c3e59dad'
NATIVE_SHA = '55d280e2ceac4373d68e46702636d1ed70a4fffeb473e87e0505ed7ec5d34484'
WATCH_SHA = 'e4d9b850af941b900ff33117b8a0163740cb20dde5b27173c9b01f8570445131'
INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
INDEX_SHA = 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
STORE_MANIFEST_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
TRAIN = (35,100)
OUT = HERE/'judgment_v1'
BUDGET, RSS_LIMIT, WALL_LIMIT = 2**41,2_000_000_000,120.0
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')


def module(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load(np,item):
    require(sha(item['path']) == item['sha256'],'Sealed prediction file')
    value = np.load(item['path'],allow_pickle=False)
    require(not value.dtype.hasobject and np.isfinite(value).all() and
            str(value.dtype) == item['dtype'] and list(value.shape) == item['shape'] and
            hashlib.sha256(value.tobytes(order='C')).hexdigest() == item['raw_sha256'],
            'Prediction shape/dtype/raw-byte binding')
    return value


def run(args,result,guard):
    pins = {str(Path(__file__).resolve()):args.source_sha256,str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,
            str(HERE/'native.py'):NATIVE_SHA,str(HERE/'watch.py'):WATCH_SHA,str(INDEX):INDEX_SHA}
    for path,digest in pins.items():
        require(sha(path) == digest,'Reviewed source/index: '+path)
    watch = module(HERE/'watch.py','terminal_tree_native_watch_verifier')
    aggregate,end = watch.sealed(HERE/'native_watch_v1',args.native_watch_result_sha256,pins)
    require(aggregate['schema'] == 'terminal_full_fourth_tree_native_watch_v1' and
            all(aggregate.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
                'all5_complete','all_predictions_and_bills_sealed_before_scoring','zero_all16_bytes_and_bill_exact',
                'earlier15_rows_byte_exact','repeat_byte_and_bill_exact','whole_owned_tree_retired')) and
            end['passed'] is True and aggregate['truth_accessed'] is False and aggregate['score_measured'] is False,
            'Complete fixed target-free native acquisition')
    require(aggregate['source_bindings'][str(HERE/'watch.py')] == WATCH_SHA and
            aggregate['source_bindings'][str(HERE/'native.py')] == NATIVE_SHA and
            aggregate['capacity'] == 1 and aggregate['elapsed_s'] < 900 and
            aggregate['maximum_sampled_rss_bytes'] < 8_000_000_000 and
            aggregate['maximum_swap_growth_bytes'] <= 512*1024**2 and
            aggregate['maximum_sample_interval_s'] <= 1.0,'Exact watcher/source/resource contract')
    p = aggregate['prerequisites']
    admission_args = types.SimpleNamespace(source_sha256=WATCH_SHA,native_sha256=NATIVE_SHA,
        protocol_sha256=PROTOCOL_SHA,build_result_sha256=p['build_result_sha256'],
        tiny_result_sha256=p['tiny_result_sha256'],tiny_watch_result_sha256=p['tiny_watch_result_sha256'],
        terminal_score_result_sha256=p['terminal_score_result_sha256'],
        terminal_validation_finished=aggregate['launch_attestations']['terminal_validation_finished'],
        exclusive_numerical_slot=aggregate['launch_attestations']['exclusive_numerical_slot'])
    worker,admission_pins = watch.authenticate(admission_args)
    for path,digest in admission_pins.items():
        require(path not in pins or pins[path] == digest,'Conflicting source admission')
        pins[path] = digest
    watch.pin_inventory(aggregate,pins,('bindings','source_bindings','artifacts'))
    require(aggregate['roster'] == [dict(role=r,rank=n,label=l) for r,n,l in worker.ROSTER] and
            len(aggregate['calls']) == 5,'Exact five-call order')
    records = []
    for call,(role,rank,label) in zip(aggregate['calls'],worker.ROSTER):
        require((call['role'],call['rank'],call['label']) == (role,rank,label) and call['exit_code'] == 0 and
                call['elapsed_s'] < 120,'Actual successful fresh cold call')
        item,record = watch.verify_worker(worker,admission_args,role,rank,HERE/'native_v1'/label,pins)
        require(item == call['result'],'Aggregate owns this immutable child result')
        records.append(record)
    native_seal_path = HERE/'native_watch_v1/predictions_and_bills_sealed.json'
    require(sha(native_seal_path) == aggregate['predictions_and_bills_seal_sha256'],'Acquisition pre-score seal')
    native_seal = read(native_seal_path)
    require(native_seal['truth_accessed'] is False and native_seal['all_predictions_and_bills_sealed'] and
            native_seal['calls'] == [x['result'] for x in aggregate['calls']] and
            native_seal['complete_cold_bills'] == [x['complete_bill'] for x in records],
            'All five predictions and own complete bills were sealed together')
    for key in ('host','hidden','sigma','delta','raw','served'):
        require(records[2]['captures'][key]['raw_sha256'] == records[4]['captures'][key]['raw_sha256'],
                'Enabled repeat byte seal: '+key)
    require(records[2]['complete_bill']['flops'] == records[4]['complete_bill']['flops'] and
            records[2]['complete_bill']['operation_count'] == records[4]['complete_bill']['operation_count'] and
            records[2]['diagnostics'] == records[4]['diagnostics'],'Repeat cost/state integrity')
    index = read(INDEX)
    require(index['manifest_sha256'] == STORE_MANIFEST_SHA and len(index['train_ranks']) == 160 and
            all(rank in index['train_ranks'] for rank in TRAIN),'Same canonical weights/TRAIN truth frame')
    target_entries = {}
    for rank in TRAIN:
        entries = [x for x in index['target_sources'] if x['rank'] == rank]
        require(len(entries) == 1,'Unique fixed TRAIN truth metadata')
        target_entries[rank] = entries[0]
    require(not any(x in sys.modules for x in ('numpy','flopscope','torch')),'All admission before numerical imports')
    import numpy as np
    require(np.__version__ == '2.4.6','Fixed official native-score NumPy convention')
    result['environment'] = dict(numpy=np.__version__,python=sys.version,executable=sys.executable,device='CPU')
    predictions,raws = [],[]
    for record in records:
        y,z = load(np,record['captures']['served']),load(np,record['captures']['raw'])
        require(y.shape == z.shape == (16,1024) and y.dtype == np.float32 and z.dtype == np.float64 and
                (y >= 0).all() and np.maximum(z,0.0).astype(np.float32).tobytes() == y.tobytes(),
                'Actual all16 native clamp/return bytes')
        predictions.append(y)
        raws.append(z)
    for baseline,candidate in ((0,2),(1,3)):
        require(predictions[baseline][:15].tobytes() == predictions[candidate][:15].tobytes() and
                raws[baseline][:15].tobytes() == raws[candidate][:15].tobytes(),
                'Both earlier15 output/raw rows are BYTE exact')
    require(predictions[2].tobytes() == predictions[4].tobytes() and
            raws[2].tobytes() == raws[4].tobytes(),'Repeated enabled full outputs BYTE exact')
    for path,digest in pins.items():
        require(sha(path) == digest,'Output/source seals before truth: '+path)
    guard()
    result.update(prerequisites=dict(native_watch_result_sha256=args.native_watch_result_sha256,**p),
        source_bindings={k:v for k,v in pins.items() if Path(k).suffix in ('.py','.md')},
        input_bindings={k:v for k,v in pins.items() if Path(k).suffix not in ('.py','.md')},
        all5_native_verified=True,all16_zero_bytes_and_own_bills_exact=True,
        earlier15_enabled_byte_exact=True,repeat_byte_and_bill_exact=True)
    seal_path = OUT/'predictions_and_bills_sealed_before_truth.json'
    write(seal_path,dict(truth_accessed=False,train_ranks=list(TRAIN),bindings=pins,
        native_watch_result_sha256=args.native_watch_result_sha256,
        all5_results=[x['result'] for x in aggregate['calls']],
        complete_cold_bills=[x['complete_bill'] for x in records],repeat_excluded=True,
        source_strength=1,all5_native_verified=True))
    result['artifacts'][str(seal_path)] = sha(seal_path)
    result['predictions_and_bills_seal_sha256'] = result['artifacts'][str(seal_path)]
    result['all_predictions_verified_before_truth'] = True
    # FIRST truth-file hash/read occurs only after the complete pre-truth seal.
    targets = {}
    for rank in TRAIN:
        path = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        result['truth_accessed'] = True
        expected = target_entries[rank]['target_sha256']
        require(sha(path) == expected,'Pinned observed1e9 TRAIN truth')
        result['target_bindings'][str(path)] = expected
        result['target_ranks_accessed'].append(rank)
        with np.load(path,allow_pickle=False) as archive:
            target = np.asarray(archive['mc_post'],dtype=np.float64).copy()
        require(target.shape == (16,1024) and np.isfinite(target).all(),'Finite fixed TRAIN mean truth')
        targets[rank] = target
        guard()
    for rank,baseline,candidate in ((35,0,2),(100,1,3)):
        row = dict(rank=rank,truth_sha256=target_entries[rank]['target_sha256'],per_role={})
        for role,slot in (('current_d50_zero',baseline),('enabled',candidate)):
            layer_mse = np.mean((predictions[slot].astype(np.float64)-targets[rank])**2,axis=1)
            raw = float(layer_mse[-1])
            flops = records[slot]['complete_bill']['flops']
            require(np.isfinite(layer_mse).all() and raw >= 0 and type(flops) is int and
                    0 < flops <= BUDGET,'Finite actual raw and complete measured cold FLOPs')
            multiplier = max(.1,flops/BUDGET)
            row['per_role'][role] = dict(raw_final_mse=raw,flops=flops,multiplier=multiplier,
                adjusted=raw*multiplier,layer_mse=layer_mse.tolist(),cold=True,includes_initialization=True)
        a,b = row['per_role']['enabled'],row['per_role']['current_d50_zero']
        require(b['raw_final_mse'] > 0 and b['adjusted'] > 0,'Positive baseline denominators')
        row.update(raw_ratio=a['raw_final_mse']/b['raw_final_mse'],adjusted_ratio=a['adjusted']/b['adjusted'],
            flops_ratio=a['flops']/b['flops'],raw_win=a['raw_final_mse'] < b['raw_final_mse'],
            adjusted_win=a['adjusted'] < b['adjusted'])
        result['records'].append(row)
    summary = {role:dict(mean_raw_final_mse=math.fsum(x['per_role'][role]['raw_final_mse'] for x in result['records'])/2,
        mean_adjusted=math.fsum(x['per_role'][role]['adjusted'] for x in result['records'])/2,
        mean_complete_cold_flops=sum(x['per_role'][role]['flops'] for x in result['records'])/2)
        for role in ('current_d50_zero','enabled')}
    advances = all(x['adjusted_win'] for x in result['records'])
    for path,digest in {**pins,**result['target_bindings'],**result['artifacts']}.items():
        require(sha(path) == digest,'Final immutable source/output/truth/seal: '+path)
    require(result['target_ranks_accessed'] == list(TRAIN),'No other truth ranks accessed')
    guard()
    result.update(passed=advances,completed=True,integrity_passed=True,numerical_gates_passed=True,
        hashes_unchanged=True,sources_unchanged=True,summary=summary,
        mean_adjusted_ratio=summary['enabled']['mean_adjusted']/summary['current_d50_zero']['mean_adjusted'],
        adjusted_wins=sum(x['adjusted_win'] for x in result['records']),both_adjusted_improve=advances,
        advances_to_fixed_native40=advances,scientific_advancement=advances,
        decision='PASS_fixed_two_TRAIN_gate' if advances else 'FAIL_fixed_two_TRAIN_gate',
        rule='BOTH native TRAIN35/100 final MSE * max(.1, own complete cold F/2^41) strictly improve.',
        scope='Observed1e9 TRAIN means; not VALID40, independent14, production or official score.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for key in ('source','protocol','native-watch-result'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for k,v in vars(args).items() if k != 'run'),
            'Explicit reviewed source/protocol/aggregate hashes')
    require(args.protocol_sha256 == PROTOCOL_SHA and sha(__file__) == args.source_sha256,
            'Reviewed fixed scorer')
    require(Path(sys.prefix).resolve() == OFFPY.resolve() and sys.flags.optimize == 0 and
            os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in BLAS) and
            os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and
            os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0','OFFPY/nice10/BLAS1/no bytecode/no MPS')
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(schema='terminal_full_fourth_tree_two_train_score_v1',passed=False,completed=False,
        integrity_passed=False,hashes_unchanged=False,sources_unchanged=False,
        truth_accessed=False,target_ranks_accessed=[],target_bindings={},records=[],artifacts={},
        source_strength=1,parameter_updates=0,model_calls=0,gradient_calls=0,MPS_used=False,
        validation_truths_accessed=False,independent14_score_measured=False,board_qualified=False,
        no_tuning_or_refit=True,scientific_advancement=False,advances_to_fixed_native40=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    def guard(*_):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(rss if sys.platform == 'darwin' else rss*1024)
        result.update(elapsed_s=time.monotonic()-started,peak_rss_bytes=rss)
        require(result['elapsed_s'] < WALL_LIMIT and rss < RSS_LIMIT,'Judge120s/2GB limit')
    previous = signal.signal(signal.SIGALRM,guard)
    signal.setitimer(signal.ITIMER_REAL,.5,.5)
    try:
        run(args,result,guard)
    except BaseException:
        result.update(passed=False,completed=False,integrity_passed=False,
            scientific_advancement=False,advances_to_fixed_native40=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        try:
            guard()
        except BaseException:
            result.update(passed=False,completed=False,integrity_passed=False,
                scientific_advancement=False,advances_to_fixed_native40=False,resource_error=traceback.format_exc())
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(OUT/'result.json',result)
        write(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            integrity_passed=result['integrity_passed'],result_sha256=sha(OUT/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','integrity_passed','mean_adjusted_ratio',
        'adjusted_wins','elapsed_s','error','resource_error')}),flush=True)
    return 0 if result['completed'] and result['integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
