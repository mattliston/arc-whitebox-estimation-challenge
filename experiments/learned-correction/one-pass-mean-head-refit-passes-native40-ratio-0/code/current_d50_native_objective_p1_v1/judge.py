"""Score the sole P1 proposal using native outputs and its own complete bills.

No refit, model, FNP, Torch or checkpoint selection. Source/byte/lifecycle
qualification precedes all VALID truth access. Surrogate comparisons are
diagnostics only; advancement uses the actual per-network score objective.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
INDEX = NEW/'joint_prefix_final_judge_cached_v2/receipt.json'
INDEX_SHA = '4968610553b95011cabdc651dd65d4c9057054f9d2d2bf73711426fff695893c'
NATIVE_WATCH_SHA = '234db460741f6c42033e059a72f3a0e2c3100d3b240641fd38926bdcbcde0f2c'


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def charts(record, count):
    require(len(record['domain_diagnostics']) == count,'Complete actual domain sequence')
    keys = {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
    for call,(trial,item) in enumerate(zip(record['trials'],record['domain_diagnostics'])):
        require(item['call'] == trial['call'] == call and item['rank'] == trial['rank']
                and item['variance_count'] == 16 and item['selector_count'] == 47
                and item['added_participant_array_operations'] == 0
                and set(item['variance_domain']) == {str(l) for l in range(16)}
                and set(item['selection_cells']) == keys, 'All16 domains/all47 selectors')
        for domain in item['variance_domain'].values():
            require(domain['finite'] and math.isfinite(domain['minimum']) and domain['minimum'] >= 1e-10
                    and domain['below_floor_count'] == 0, 'Finite unfloored native variance')
        for cell in item['selection_cells'].values():
            require(cell['floor'] == .01 and type(cell['count']) is int and 0 <= cell['count'] <= 1024
                    and cell['count'] % 32 == 0 and all(isinstance(cell[k],str) and len(cell[k]) == 64
                        for k in ('indices_sha256','mask_sha256')), 'Sealed native selector integrity')
    require({k:v for k,v in record['domain_diagnostics'][0].items() if k != 'call'} ==
            {k:v for k,v in record['domain_diagnostics'][-1].items() if k != 'call'}, 'Exact repeated chart')


def run(args,out,result,guard):
    require(sha(HERE/'native.py') == args.native_source_sha256,'Reviewed native source')
    spec = importlib.util.spec_from_file_location('p1_score_native_metadata',HERE/'native.py')
    n = importlib.util.module_from_spec(spec); spec.loader.exec_module(n)
    tl = n.legacy(); pins = {}; pin = tl.pinner(pins)
    pin(__file__,args.source_sha256); pin(HERE/'native.py',args.native_source_sha256)
    pin(HERE/'native_watch.py',NATIVE_WATCH_SHA)
    pin(HERE/'PROTOCOL.md',args.protocol_sha256)
    require(args.protocol_sha256 == n.PROTOCOL_SHA,'Fixed objective protocol')
    context = n.admission(args.build_sha256,args.native_source_sha256,pin)
    result.update(source_bindings=pins,truth_bindings={})
    controls = context['control_outputs']; control = context['control']
    acquisitions,blobs = {},{}
    for role,watch_stage,count in (('null','null',2),('terminal','native',41)):
        folder = HERE/f'native_{role}_v1'
        digest = args.native_result_sha256 if role == 'terminal' else pin(folder/'result.json')
        record = tl.sealed(folder,digest,pin)
        candidate = context['null' if role == 'null' else 'candidate']
        candidate_sha = tl.BANK_SHA if role == 'null' else context['candidate_sha256']
        require(record['schema'] == n.SCHEMA and record['stage'] == 'native' and record['role'] == role
                and record['candidate_sha256'] == candidate_sha
                and record['source_bindings'][str(HERE/'native.py')] == args.native_source_sha256
                and record['prerequisites']['build_result_sha256'] == args.build_sha256
                and record['prerequisites']['control_result_sha256'] == tl.CONTROL_SHA,
                'Exact built candidate/null and current native source/control')
        require(all(record[k] is True for k in ('own_complete_bills','passive_byte_observation',
            'source_adapter_inverse_verified','paid_arithmetic_and_lifecycle_unchanged',
            'all16_domains_qualified','all47_selectors_qualified','all_outputs_and_bills_sealed_before_scoring')),
            'Completed actual native qualification')
        require(record['native_original_sha256'] == n.NATIVE_SHA and record['native_main_sha256'] == n.derived_main()[1]
                and record['versions'] == tl.VERSIONS and record['threadpools'] == control['threadpools'],
                'Same qualified native arithmetic/backend frame')
        base,verify = tl.verifier(candidate,candidate_sha,pin)
        blobs[role] = verify(record,folder,(n.VALID[0],) if role == 'null' else n.VALID,pin)
        require(record['initialization_flops'] == n.INITIALIZATION,'Actual complete initialization once')
        charts(record,count)
        wf = HERE/f'native_{role}_watch_v1'
        wd = args.native_watch_sha256 if role == 'terminal' else pin(wf/'result.json')
        watch = tl.sealed(wf,wd,pin)
        require(watch['schema'] == 'current_d50_native_objective_p1_native_watch_v1' and watch['stage'] == watch_stage
                and watch['child_result_sha256'] == digest and watch['child_completion_sha256'] == pin(folder/'completion.json')
                and watch['child_execution_completed'] and watch['whole_owned_tree_retired']
                and watch['source_bindings'][str(HERE/'native.py')] == args.native_source_sha256
                and watch['source_bindings'][str(HERE/'native_watch.py')] == NATIVE_WATCH_SHA,
                'Independent native resource-completion seal')
        tl.bind(watch['source_bindings'],pin); tl.bind(watch['artifacts'],pin)
        acquisitions[role] = record
    require(acquisitions['null']['null_full_output_bill_parity'],'Measured null parity')
    for null_call,control_call in ((0,0),(1,40)):
        a,b = acquisitions['null']['trials'][null_call],control['trials'][control_call]
        require(blobs['null'][null_call] == controls[control_call] and a['flops'] == b['flops']
                and a['operation_count'] == b['operation_count'], 'Null all16 BYTE/full bill exact')
    native = acquisitions['terminal']
    surrogate = {}
    for rank,item in context['training']['validation_outputs'].items():
        pin(item['path'],item['sha256'])
        surrogate[int(rank)] = base.payload(item['path'],item['raw_sha256'],pin)
    pin(INDEX,INDEX_SHA)
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'All pretruth qualification is byte/metadata-only')
    require(all(sha(p) == d for p,d in pins.items()),'All prediction/source/bill seals intact')
    seal = out/'predictions_and_bills_sealed.json'
    write(seal,dict(truth_accessed=False,source_bindings=pins,build_result_sha256=args.build_sha256,
        native_result_sha256=args.native_result_sha256,native_watch_sha256=args.native_watch_sha256,
        native_trials=native['trials'],control_trials=control['trials'],validation_ranks=list(n.VALID),
        initialization_once=True,repeat_excluded=True,own_complete_bills=True))
    result['predictions_and_bills_seal_sha256'] = sha(seal)
    guard()
    import numpy as np
    require(np.__version__ == '2.4.6','Fixed score precision convention')
    rows = read(INDEX)['records']
    require(len(rows) == 40 and {r['rank'] for r in rows} == set(n.VALID),'Pinned existing40 truth index')
    expected = {r['rank']:r['target_sha256'] for r in rows}
    result['truth_accessed'] = True
    for call,rank in enumerate(n.VALID):
        path = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        require(sha(path) == expected[rank],'Fixed 1e9 VALID mean reference')
        result['truth_bindings'][str(path)] = expected[rank]
        with np.load(path,allow_pickle=False) as source:
            target = source['mc_post'].astype(np.float64)
        require(target.shape == (16,1024) and np.isfinite(target).all(),'Finite16-row mean reference')
        row = dict(rank=rank,call=call,per_role={})
        for role,outputs,acq in (('current_d50',controls,control),('terminal',blobs['terminal'],native)):
            value = np.frombuffer(outputs[call],dtype=np.float32).reshape(16,1024).astype(np.float64)
            mse = np.mean((value-target)**2,axis=1); bill = acq['trials'][call]
            require(np.isfinite(mse).all() and (mse >= 0).all(),'Finite measured native MSE')
            row['per_role'][role] = dict(raw_mse=float(mse[-1]),adjusted=float(mse[-1])*bill['multiplier'],
                flops=bill['flops'],multiplier=bill['multiplier'],layer_mse=mse.tolist())
        a,b = row['per_role']['terminal'],row['per_role']['current_d50']
        shadow = np.frombuffer(surrogate[rank],dtype=np.float32).reshape(16,1024).astype(np.float64)
        row.update(raw_ratio=a['raw_mse']/b['raw_mse'],adjusted_ratio=a['adjusted']/b['adjusted'],
            raw_win=a['raw_mse'] < b['raw_mse'],adjusted_win=a['adjusted'] < b['adjusted'],
            surrogate_raw_mse=float(np.mean((shadow[-1]-target[-1])**2)),
            native_surrogate_max_abs=float(np.max(np.abs(value-shadow))),
            native_surrogate_final_rms=float(np.sqrt(np.mean((value[-1]-shadow[-1])**2))))
        result['records'].append(row); print(json.dumps(row,allow_nan=False),flush=True)
        guard()
    summary = {role:dict(mean_raw_mse=math.fsum(r['per_role'][role]['raw_mse'] for r in result['records'])/40,
        mean_adjusted=math.fsum(r['per_role'][role]['adjusted'] for r in result['records'])/40,
        mean_complete_flops=sum(r['per_role'][role]['flops'] for r in result['records'])/40)
        for role in ('current_d50','terminal')}
    ratio = summary['terminal']['mean_adjusted']/summary['current_d50']['mean_adjusted']
    wins = sum(r['adjusted_win'] for r in result['records']); advances = ratio < 1 and wins >= 24
    require(all(sha(p) == d for p,d in pins.items()) and all(sha(p) == d for p,d in result['truth_bindings'].items())
            and sha(seal) == result['predictions_and_bills_seal_sha256'],'Unchanged final source/output/truth seals')
    guard()
    result.update(passed=advances,completed=True,integrity_passed=True,hashes_unchanged=True,
        summary=summary,mean_adjusted_ratio=ratio,adjusted_wins=wins,
        mean_raw_ratio=summary['terminal']['mean_raw_mse']/summary['current_d50']['mean_raw_mse'],
        raw_wins=sum(r['raw_win'] for r in result['records']),advances_to_independent14=advances,
        objective_improved=ratio < 1,wins_gate=wins >= 24,candidate_sha256=context['candidate_sha256'],
        surrogate_diagnostic_only=True,no_native_gradient_claim=True,
        artifacts={str(seal):sha(seal)},all_predictions_verified_before_truth=True,
        rule='Native per-net observed final MSE * own max(.1,F/2**41) mean improves d50; >=24/40 wins.',
        scope='Reused VALID40 observed1e9 truth; not independent14, production, or board evidence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for name in ('source','protocol','build','native-source','native-result','native-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(all(isinstance(v,str) and len(v) == 64 and set(v) <= set('0123456789abcdef')
                for k,v in vars(args).items() if k != 'run'), 'Reviewed exact source/receipt hashes')
    require(Path(sys.prefix).resolve() == (P2/'data_gate/.venv').resolve()
            and sys.flags.optimize == 0 and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')), 'OFFPY nice10/BLAS1/bytecode0')
    out = HERE/'score_v1'; out.mkdir(exist_ok=False)
    began = time.monotonic()
    result = dict(schema='current_d50_native_objective_p1_score_v1',stage='score',passed=False,completed=False,
        integrity_passed=False,hashes_unchanged=False,truth_accessed=False,records=[],requested_seals=vars(args),
        training_or_refit_performed=False,independent_panel_qualified=False,production_qualified=False,
        official_score=False,advances_to_independent14=False,started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json',result)
    def guard(*_):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result.update(peak_rss_bytes=rss,wall_s=time.monotonic()-began)
        require(rss < 2_000_000_000 and time.monotonic()-began < 120,'Offline judge2GB/120s')
    signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.5,.5)
    try:
        run(args,out,result,guard)
    except BaseException:
        result.update(passed=False,completed=False,integrity_passed=False,hashes_unchanged=False,
                      advances_to_independent14=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_s=time.monotonic()-began,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','mean_adjusted_ratio','adjusted_wins','error')}),flush=True)
    return 0 if result['integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
