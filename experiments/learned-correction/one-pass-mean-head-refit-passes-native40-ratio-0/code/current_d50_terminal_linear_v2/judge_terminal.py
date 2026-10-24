"""Actual terminal-P65 native40 objective; inert until all future seals exist.

No model, FNP, Torch, refit or shadow-fidelity gate. The native candidate and
current d50 are compared using their own complete per-call bills, cold37 once,
repeat37 excluded. All served outputs/bills are sealed before VALID truth IO.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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
NEW, P2 = HERE.parent,HERE.parent.parent
TRUTH_INDEX = NEW/'joint_prefix_final_judge_cached_v2/receipt.json'
TRUTH_INDEX_SHA = '4968610553b95011cabdc651dd65d4c9057054f9d2d2bf73711426fff695893c'
BUDGET = 2**41


def require(value, message):
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


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')


def run(args, out, result, guard):
    require(sha(HERE/'native_validation.py') == args.native_source_sha256,'Reviewed native adapter source')
    spec = importlib.util.spec_from_file_location('tl_terminal_native_metadata',HERE/'native_validation.py')
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
    pins,targets = {},{}
    result.update(source_bindings=pins,truth_bindings=targets)
    pin = helper.pinner(pins)
    pin(__file__,args.source_sha256); pin(HERE/'native_validation.py',args.native_source_sha256)
    pin(HERE/'PROTOCOL.md',args.protocol_sha256)
    require(args.protocol_sha256 == helper.PROTOCOL_SHA,'Fixed prospective protocol')
    context = helper.authenticate_fit(args.fit_sha256,pin)
    control,control_outputs = helper.control(pin)
    folder = HERE/'validation_terminal_v1'
    native = helper.sealed(folder,args.native_result_sha256,pin)
    require(native['schema'] == 'current_d50_terminal_linear_native40_v1'
            and native['candidate_sha256'] == context['candidate_sha256']
            and native['source_bindings'][str(HERE/'native_validation.py')] == args.native_source_sha256
            and native['prerequisites']['fit_result_sha256'] == args.fit_sha256
            and native['prerequisites']['fit_watch_sha256'] == context['fit_watch_sha256']
            and native['prerequisites']['control_result_sha256'] == helper.CONTROL_SHA
            and native['prerequisites']['source_inverse'] == context['proof'],
            'Exact sole fitted candidate, matching source and current control')
    require(all(native[k] is True for k in ('own_complete_bills','passive_byte_observation',
            'source_adapter_inverse_verified','paid_arithmetic_and_lifecycle_unchanged',
            'earlier15_rows_byte_exact','own_complete_F_and_opcounts_exact',
            'all16_domains_qualified','all47_selectors_qualified','all_outputs_and_bills_sealed_before_scoring')),
            'Completed native arithmetic, domain and earlier-output qualification')
    require(native['native_original_sha256'] == helper.NATIVE_SHA
            and native['native_main_sha256'] == helper.derived_main()[1]
            and native['versions'] == helper.VERSIONS,'Qualified exact native source/frame')
    base,verify = helper.verifier(context['candidate'],context['candidate_sha256'],pin)
    terminal_outputs = verify(native,folder,helper.VALID,pin)
    require(native['initialization_flops'] == control['initialization_flops'] == helper.INITIALIZATION,
            'Actual one-time initialization identical to current geometry')
    pools = native['threadpools']
    require(isinstance(pools,list) and pools == control['threadpools']
            and all(p['num_threads'] == 1 for p in pools if p.get('user_api') == 'blas'),
            'Same reported single-thread backend inventory; empty is unreported, not measured')
    diagnostics = native['domain_diagnostics']
    require(len(diagnostics) == 41,'All41 native domain records')
    keys = {f'{l}:{slot}' for l in range(16) for slot in range(2 if l == 15 else 3)}
    for call,(a,b,diag) in enumerate(zip(native['trials'],control['trials'],diagnostics)):
        require(a['call'] == b['call'] == call and a['rank'] == b['rank'] == diag['rank']
                and diag['call'] == call and a['includes_initialization'] == b['includes_initialization'],
                'Same current fixture and cold/warm call ownership')
        require(a['flops'] == b['flops'] and a['operation_count'] == b['operation_count'],
                'Fresh measured complete F/opcount unchanged, not transplanted bills')
        require(terminal_outputs[call][:15*1024*4] == control_outputs[call][:15*1024*4],
                'All earlier15 rows BYTE exact before any truth access')
        require(diag['variance_count'] == 16 and diag['selector_count'] == 47
                and diag['added_participant_array_operations'] == 0
                and set(diag['variance_domain']) == {str(l) for l in range(16)}
                and set(diag['selection_cells']) == keys
                and all(x['finite'] and math.isfinite(x['minimum']) and x['minimum'] >= 1e-10
                        and x['below_floor_count'] == 0 for x in diag['variance_domain'].values()),
                'Every actual native prevariance finite and unfloored')
        for cell in diag['selection_cells'].values():
            require(cell['floor'] == .01 and type(cell['count']) is int and 0 <= cell['count'] <= 1024
                    and cell['count'] % 32 == 0
                    and all(isinstance(cell[k],str) and len(cell[k]) == 64 for k in
                            ('indices_sha256','mask_sha256')),'Validated ordered selector byte seals')
    require({k:v for k,v in diagnostics[0].items() if k != 'call'} ==
            {k:v for k,v in diagnostics[-1].items() if k != 'call'},'Repeat domain/selector identity')
    watch = helper.sealed(HERE/'validation_watch_v1',args.native_watch_sha256,pin)
    require(watch['stage'] == 'native' and watch['child_result_sha256'] == args.native_result_sha256
            and watch['child_completion_sha256'] == pin(folder/'completion.json')
            and watch['source_bindings'][str(HERE/'native_validation.py')] == args.native_source_sha256
            and watch['child_execution_completed'] and watch['whole_owned_tree_retired'],
            'Completed independent native resource supervision')
    helper.bind(watch['source_bindings'],pin); helper.bind(watch['artifacts'],pin)
    pin(TRUTH_INDEX,TRUTH_INDEX_SHA)
    require(not any(x in sys.modules for x in ('numpy','flopscope','torch')),'All source/output/bill admission before numeric imports')
    require(all(sha(path) == digest for path,digest in pins.items()),'All prediction/source seals intact')
    guard()
    seal_path = out/'predictions_and_bills_sealed.json'
    write(seal_path,dict(truth_accessed=False,fit_sha256=args.fit_sha256,
        candidate_sha256=context['candidate_sha256'],source_bindings=pins,
        native_result_sha256=args.native_result_sha256,native_watch_sha256=args.native_watch_sha256,
        control_result_sha256=helper.CONTROL_SHA,validation_ranks=list(helper.VALID),
        initialization_once=True,repeat_excluded=True,earlier15_rows_byte_exact=True,
        native_trials=native['trials'],control_trials=control['trials']))
    result['predictions_and_bills_seal_sha256'] = sha(seal_path)
    result.update(all_predictions_verified_before_truth=True,candidate_sha256=context['candidate_sha256'],
        all_earlier15_outputs_byte_exact=True,all_own_complete_bills_verified=True,
        native_initialization_flops=native['initialization_flops'],control_initialization_flops=control['initialization_flops'])
    import numpy as np
    require(np.__version__ == '2.4.6','Exact original native-score NumPy convention')
    roster = read(TRUTH_INDEX)['records']
    require(len(roster) == 40 and {r['rank'] for r in roster} == set(helper.VALID),'Original fixed40 truth index')
    expected = {row['rank']:row['target_sha256'] for row in roster}
    paths = {rank:P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
             for rank in helper.VALID}
    result['truth_accessed'] = True
    for rank,path in paths.items():
        require(sha(path) == expected[rank],'Original pinned 1e9 VALID truth')
        targets[str(path)] = expected[rank]
    write(out/'truth_bindings.json',targets)
    result['truth_arrays_opened'] = True
    for call,rank in enumerate(helper.VALID):
        with np.load(paths[rank],allow_pickle=False) as data:
            truth = data['mc_post'].copy()
        require(truth.shape == (16,1024) and truth.dtype.kind == 'f' and np.isfinite(truth).all(),'Finite original16 mean truth')
        target = truth.astype(np.float64)
        row = dict(rank=rank,call=call,per_role={},truth_sha256=expected[rank])
        for role,blobs,acquisition in (('current_d50',control_outputs,control),('terminal',terminal_outputs,native)):
            prediction = np.frombuffer(blobs[call],dtype=np.float32).reshape(16,1024)
            layer_mse = np.mean((prediction.astype(np.float64)-target)**2,axis=1)
            raw = float(layer_mse[-1]); trial = acquisition['trials'][call]
            require(math.isfinite(raw) and raw >= 0,'Finite final observed MSE')
            row['per_role'][role] = dict(raw_mse=raw,adjusted=raw*trial['multiplier'],
                flops=trial['flops'],multiplier=trial['multiplier'],layer_mse=layer_mse.tolist())
        a,b = row['per_role']['terminal'],row['per_role']['current_d50']
        require(b['raw_mse'] > 0 and b['adjusted'] > 0,'Positive current comparator denominator')
        row.update(raw_ratio=a['raw_mse']/b['raw_mse'],adjusted_ratio=a['adjusted']/b['adjusted'],
                   raw_win=a['raw_mse'] < b['raw_mse'],adjusted_win=a['adjusted'] < b['adjusted'])
        result['records'].append(row)
        print(json.dumps(row,allow_nan=False),flush=True)
        guard()
    summary = {role:dict(mean_raw_mse=math.fsum(r['per_role'][role]['raw_mse'] for r in result['records'])/40,
        mean_adjusted=math.fsum(r['per_role'][role]['adjusted'] for r in result['records'])/40,
        mean_complete_flops=sum(r['per_role'][role]['flops'] for r in result['records'])/40)
        for role in ('current_d50','terminal')}
    ratio = summary['terminal']['mean_adjusted']/summary['current_d50']['mean_adjusted']
    wins = sum(r['adjusted_win'] for r in result['records'])
    advances = ratio < 1 and wins >= 24
    require(all(sha(path) == digest for path,digest in pins.items()),'Source/prediction end seals')
    require(all(sha(path) == digest for path,digest in targets.items()),'Truth end seals')
    require(sha(seal_path) == result['predictions_and_bills_seal_sha256'],'Pretruth seal unchanged')
    guard()
    result.update(passed=advances,completed=True,integrity_passed=True,hashes_unchanged=True,
        summary=summary,mean_adjusted_ratio=ratio,
        mean_raw_ratio=summary['terminal']['mean_raw_mse']/summary['current_d50']['mean_raw_mse'],
        adjusted_wins=wins,raw_wins=sum(r['raw_win'] for r in result['records']),
        objective_improved=ratio < 1,wins_gate=wins >= 24,advances_to_independent14=advances,
        rule='Mean of per-net final observed MSE * own max(.1,F/2**41) improves current d50, and >=24/40 wins.',
        scope='Reused VALID40 observed1e9 means; not independent14, a board result or production qualification.',
        artifacts={str(p):sha(p) for p in (seal_path,out/'truth_bindings.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for name in ('source','protocol','fit','native-source','native-result','native-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(all(isinstance(v,str) and len(v) == 64 and set(v) <= set('0123456789abcdef')
                for k,v in vars(args).items() if k != 'run'),'Reviewed future receipt/source hashes')
    require(Path(sys.prefix).resolve() == (P2/'data_gate/.venv').resolve()
            and importlib.metadata.version('numpy') == '2.4.6'
            and os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'OFFPY nice10 BLAS1')
    out = HERE/'terminal_score_v1'; out.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(schema='current_d50_terminal_linear_native40_score_v1',passed=False,completed=False,
        integrity_passed=False,hashes_unchanged=False,truth_accessed=False,truth_arrays_opened=False,
        training_or_refit_performed=False,shadow_or_VJP_gate=False,records=[],requested_seals=vars(args),
        independent_panel_qualified=False,production_qualified=False,official_score=False,
        advances_to_independent14=False,started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json',result)
    def guard(*_):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result.update(peak_rss_bytes=rss,wall_s=time.monotonic()-started)
        require(rss < 2_000_000_000 and time.monotonic()-started < 120,'Offline judge2GB/120s')
    signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.5,.5)
    try:
        run(args,out,result,guard)
    except BaseException:
        result.update(passed=False,completed=False,integrity_passed=False,hashes_unchanged=False,
                      advances_to_independent14=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_s=time.monotonic()-started,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','mean_adjusted_ratio','adjusted_wins','error')}),flush=True)
    return 0 if result['completed'] and result['integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
