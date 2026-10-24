"""One sealed two-TRAIN native objective comparison; no model or fitting."""
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
NEW,P2 = HERE.parent,HERE.parent.parent
PROTOCOL_SHA = 'c931293ede873bf193dec3229b7a02583a9ec96a2e2d95996550b9de20861d7f'
CONTROLS = {
    'd50':('birth_direction_fit_v1/panel_v2/terminal',
        '6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade',
        'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'),
    'p1':('current_d50_native_objective_p1_v1/panel_v1/terminal',
        'c8fa1cacfa98d0f45d38235283789cff67c59f32f72fa80a92dd385ed59f8c70',
        '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435')}
TRUTHS = {35:'ec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61',
          100:'637a63d46ac5480be50cca3f0021a1574eef80fa580ce98f1a8cc69b0b86450c'}
BUDGET = 2**41


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


def bind(mapping,pins):
    for key,value in mapping.items():
        path,digest = (value['path'],value['sha256']) if isinstance(value,dict) else (key,value)
        path = str(Path(path).resolve())
        require('truthpost' not in Path(path).parts,'No truth arrays before prediction seal')
        require(sha(path) == digest,'Unchanged bound artifact: '+path)
        require(path not in pins or pins[path] == digest,'Consistent binding')
        pins[path] = digest


def sealed(folder,digest,pins):
    bind({str(folder/'result.json'):digest},pins)
    record,end = read(folder/'result.json'),read(folder/'completion.json')
    require(record['passed'] and record['completed'] and record['hashes_unchanged']
            and end['passed'] and end['completed'] and end['result_sha256'] == digest,
            'Complete acquisition/watch seal')
    bind({str(folder/'completion.json'):sha(folder/'completion.json')},pins)
    for key in ('source_bindings','input_bindings','artifacts'):
        bind(record.get(key,{}),pins)
    return record


def trial_output(np,folder,trial,pins):
    call = trial['call']
    bp,lp = folder/f'call_{call:02d}_bill.json',folder/f'call_{call:02d}_lifecycle.json'
    require(read(bp) == trial,'Own saved bill equals scored trial')
    life = read(lp)
    require(life['saved_bill'] == str(bp) and life['saved_bill_sha256'] == sha(bp)
            and trial['lifecycle_receipt'] == str(lp)
            and life['rank'] == trial['rank'] and life['call'] == call
            and life['reset_api'] == 'flopscope.budget_reset'
            and life['candidate_state_reset'] is False and all(life[k] for k in
                ('complete','after_context_exit','context_released','closed_context_bill_unchanged_by_reset')),
            'Actual released-context lifecycle')
    for left,right in (('flops','flops'),('operation_count','operation_count'),
        ('context_wall_s','context_wall_s'),('backend_s','flopscope_backend_s'),
        ('overhead_s','flopscope_overhead_s'),('residual_s','native_residual_s')):
        require(life['closed_metrics'][left] == trial[right]
                and math.isfinite(trial[right]) and trial[right] >= 0,'Exact closed metrics')
    require(type(trial['flops']) is int and 0 < trial['flops'] <= BUDGET
            and trial['multiplier'] == max(.1,trial['flops']/BUDGET)
            and trial['wall_s'] < 120 and trial['context_wall_s'] < 120
            and 0 < trial['peak_rss_bytes'] < 8_000_000_000,'Native finite bounded own full bill')
    path = Path(trial['output'])
    require(path.parent == folder and path.name == f'call_{call:02d}_rank{trial["rank"]}.npy',
            'Own output path')
    bind({str(path):trial['output_sha256'],str(bp):sha(bp),str(lp):sha(lp)},pins)
    value = np.load(path,allow_pickle=False)
    require(type(value) is np.ndarray and value.dtype == np.float64
            and value.shape == (16,1024) and np.isfinite(value).all() and (value >= 0).all(),
            'All16 actual valid outputs')
    require(hashlib.sha256(value.tobytes()).hexdigest() == trial['output_bytes_sha256'],'Actual output bytes')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for key in ('source','native-source','watch-source','builder','build','candidate','native','watch'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.flags.optimize == 0
            and Path(sys.prefix).resolve() == (P2/'data_gate/.venv').resolve()
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1','Fixed OFFPY/nice10/no bytecode frame')
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope')),'Authenticate before numerical imports')
    sources = {str(Path(__file__).resolve()):args.source_sha256,
        str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(HERE/'native.py'):args.native_source_sha256,
        str(HERE/'watch_native.py'):args.watch_source_sha256,str(HERE/'build.py'):args.builder_sha256,
        str(HERE/'build_v1/estimator.py'):args.candidate_sha256}
    pins = {}; bind(sources,pins)
    out = HERE/'score_v1'
    require(not out.exists() and not out.is_symlink(),'Exclusive fixed two-TRAIN score')
    out.mkdir()
    result = dict(schema='current_d50_response10_candidate_score_v1',passed=False,completed=False,
        source_bindings=sources,input_bindings=pins,truth_bindings={},ranks=[35,100],
        score_population='Two TRAIN/development networks, fixed existing 1e9 references',
        official_score=False,independent_validation=False,production_qualified=False,
        candidate_sha256=args.candidate_sha256,training_or_refit_performed=False,
        estimator_executed=False,automatic_next_stage=False,truth_accessed=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json',result)
    def guard(*unused):
        require(time.monotonic()-began < 120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1_000_000_000,
                '120s/1GB complete score phase')
    old_handler = signal.signal(signal.SIGALRM,guard)
    signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        spec = importlib.util.spec_from_file_location('response_candidate_admission',HERE/'native.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        context = module.authenticate(args.build_sha256,args.candidate_sha256,args.builder_sha256)
        for key in ('source_bindings','input_bindings'):
            bind(context.get(key,{}),pins)
        native,watch = HERE/'native_v1',HERE/'native_watch_v1'
        acquisition = sealed(native,args.native_sha256,pins)
        resource_record = sealed(watch,args.watch_sha256,pins)
        require(acquisition['schema'] == 'current_d50_response10_candidate_native_v1'
                and acquisition['candidate_sha256'] == args.candidate_sha256
                and acquisition['truth_accessed'] is False and all(acquisition.get(k) is True for k in
                    ('all16_domains_qualified','all47_selectors_qualified','repeat_output_domain_byte_exact',
                     'initialization_ownership_passed','own_complete_bills','sources_unchanged')),
                'Fixed candidate complete native/domain/repeat admission')
        require(resource_record['child_result_sha256'] == args.native_sha256
                and resource_record['whole_owned_tree_retired'] and resource_record['completed_bill_count'] == 3,
                'Independent native resource supervision completed')
        require([t['rank'] for t in acquisition['trials']] == [35,100,35]
                and len(acquisition['input_preparation']) == 3,'Fixed native sequence')
        controls = {}
        for role,(folder,digest,candidate_digest) in CONTROLS.items():
            path = NEW/folder
            record = sealed(path,digest,pins)
            require(record['candidate_sha256'] == candidate_digest and record['truth_accessed'] is False
                    and [r['rank'] for r in record['trials'][:2]] == [35,100]
                    and len(record['trials']) == 15,'Archived same-input cold35 and warm100 controls')
            controls[role] = (path,record)
        require(not any(x in sys.modules for x in ('numpy','torch','flopscope')),'No numeric admission side effect')
        import numpy as np
        require(np.__version__ == '2.4.6','Qualified offline scoring NumPy')
        values = {'candidate': [trial_output(np,native,t,pins) for t in acquisition['trials']]}
        trials = {'candidate':acquisition['trials']}
        for role,(folder,record) in controls.items():
            values[role] = [trial_output(np,folder,t,pins) for t in record['trials'][:2]]
            trials[role] = record['trials'][:2]
        require(values['candidate'][0].tobytes() == values['candidate'][2].tobytes(),'Actual repeat bytes')
        for role in values:
            for call in range(2):
                require(trials[role][call]['call'] == call and
                        trials[role][call]['includes_initialization'] is (call == 0)
                        and values[role][call][0].tobytes() == values['d50'][call][0].tobytes(),
                        'Same fixed call initialization and exact first layer')
        seal = dict(source_bindings=sources,input_bindings=dict(pins),truth_accessed=False,
            candidate_sha256=args.candidate_sha256,trials=trials,native_sha256=args.native_sha256,
            watch_sha256=args.watch_sha256)
        write(out/'predictions_and_bills_sealed.json',seal)
        result.update(all_predictions_verified_before_truth=True,
            predictions_and_bills_seal_sha256=sha(out/'predictions_and_bills_sealed.json'))
        per_net = []
        for call,rank in enumerate((35,100)):
            path = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
            require(sha(path) == TRUTHS[rank],'Same frozen 1e9 reference')
            result['truth_bindings'][str(path)] = TRUTHS[rank]
            result['truth_accessed'] = True
            with np.load(path,allow_pickle=False) as item:
                target = item['mc_post'].astype(np.float64)
            require(target.shape == (16,1024) and np.isfinite(target).all(),'All16 finite reference')
            row = dict(rank=rank,metrics={})
            for role in values:
                raw = float(np.mean((values[role][call][-1]-target[-1])**2,dtype=np.float64))
                flops = trials[role][call]['flops']; multiplier = max(.1,flops/BUDGET)
                require(math.isfinite(raw) and raw > 0,'Finite positive final raw')
                row['metrics'][role] = dict(raw=raw,flops=flops,multiplier=multiplier,adjusted=raw*multiplier)
            per_net.append(row)
        means = {role:{key:sum(row['metrics'][role][key] for row in per_net)/2
                       for key in ('raw','flops','adjusted')} for role in values}
        comparisons = {}
        for role in ('d50','p1'):
            ratios = [row['metrics']['candidate']['adjusted']/row['metrics'][role]['adjusted'] for row in per_net]
            comparisons[role] = dict(mean_adjusted_ratio=means['candidate']['adjusted']/means[role]['adjusted'],
                mean_raw_ratio=means['candidate']['raw']/means[role]['raw'],
                per_network_adjusted_ratios=ratios,adjusted_wins=sum(r < 1 for r in ratios))
        advances = comparisons['d50']['mean_adjusted_ratio'] <= .95 and max(comparisons['d50']['per_network_adjusted_ratios']) <= 1
        bind(sources,{}); bind(pins,{})
        require(all(sha(path) == digest for path,digest in result['truth_bindings'].items()),'Targets unchanged')
        guard()
        result.update(passed=True,completed=True,hashes_unchanged=True,integrity_passed=True,
            per_net=per_net,means=means,comparisons=comparisons,advances_to_full_validation=advances,
            initialization_flops=acquisition['initialization_flops'])
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_handler)
        result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','means','comparisons',
        'advances_to_full_validation','elapsed_s','peak_rss_bytes','error')},allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
