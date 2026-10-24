"""Score only the sealed frozen NODE/P1 native40 composition; no fitting."""
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
P1 = NEW/'current_d50_native_objective_p1_v1'
RC = NEW/'current_d50_response10_candidate_v1'
PROTOCOL_SHA = '5aa1fefd2e7b60ce77814f39efc4dcea3bcd2cc7ed372d4841da4eb8f973696c'
BUILDER_SHA = 'f3a3c95b3a7e99a40d8951800fb9d6ec0c0252f0bd4193c3cc2941e6712e61bf'
BUILD_SHA = '78f7adbb09ce5749a442c0b4d171110627a0881a478fece4e19f4f5467ec26d6'
CANDIDATE_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
P1_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
P1_NATIVE_SHA = '7a1f37d80a7ed32850b10047408d85eed050220b4a6271dde90bdd21c07062e3'
P1_SCORE_SHA = '4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5'
HELPER_SHA = 'e4a9ef56dd5309e33cde2d67ad983e8655918af12909a8fdec58db0292c27eaf'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
BUDGET = 2**41


def helpers():
    """Reuse the qualified saved-FP32/lifecycle checker, never its main."""
    path = RC/'score_v2.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location('composition_saved_output_helpers',path)
    adapter = importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    source = adapter.derived_source()  # Proves frozen dtype-only inverse.
    namespace = dict(__file__=str(path),__name__='composition_saved_fp32_bundle')
    exec(compile(source,str(path)+':reused_helpers','exec'),namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for name in ('source','native-source','watch-source','zero','zero-watch','terminal','terminal-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    began = time.monotonic()
    h = helpers()
    require,sha,read,write,bind,sealed,trial_output = [h[k] for k in
        ('require','sha','read','write','bind','sealed','trial_output')]
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')),'No numeric imports before admission')
    require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS,0)>=10
            and Path(sys.prefix).resolve()==(P2/'data_gate/.venv').resolve()
            and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','Same OFFPY/nice10/no bytecode frame')
    require(all(isinstance(v,str) and len(v)==64 and set(v)<=set('0123456789abcdef')
                for k,v in vars(args).items() if k!='run'),'All reviewed digests required')
    sources = {str(Path(__file__).resolve()):args.source_sha256,
        str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(HERE/'build.py'):BUILDER_SHA,
        str(HERE/'native.py'):args.native_source_sha256,str(HERE/'watch_native.py'):args.watch_source_sha256,
        str(HERE/'build_v1/terminal_estimator.py'):CANDIDATE_SHA,
        str(RC/'score_v2.py'):HELPER_SHA,str(RC/'score.py'):h.get('ORIGINAL_SCORER_SHA',
            '3bc2aeb4d66a74984c2801c281a52e11b75a52f36d9bff5c877cca3daccbf2b4'),
        str(RC/'SCORE_DTYPE_V2.md'):'9995cd8ae375ae0decef7615e1b08678941510f00f1bd3bcbb7436a649bf30ac'}
    pins = {};bind(sources,pins)
    out = HERE/'score_v1'
    require(not out.exists() and not out.is_symlink(),'Exclusive terminal-only native40 score')
    out.mkdir(exist_ok=False)
    result = dict(schema='current_p1_node_composition_score_v1',passed=False,completed=False,
        source_bindings=sources,input_bindings=pins,truth_bindings={},truth_accessed=False,
        candidate_sha256=CANDIDATE_SHA,baseline_sha256=P1_SHA,training_or_refit_performed=False,
        estimator_executed=False,old_NODE_pilot_still_failed=True,official_score=False,
        production_qualified=False,independent_panel_qualified=False,automatic_next_stage=False,
        helper_dtype_inverse_verified=True,ranks=list(VALID),repeat_excluded=True,
        started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json',result)
    def guard(*unused):
        require(time.monotonic()-began<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1e9,
                'Original lightweight scoring120s/1GB bound')
    old_handler = signal.signal(signal.SIGALRM,guard)
    signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        return score_body(args,result,out,pins,h,guard)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        result.update(elapsed_s=time.monotonic()-began,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
                                         result_sha256=sha(out/'result.json')))
        print(json.dumps({k:result.get(k) for k in ('passed','summary','mean_adjusted_ratio',
            'adjusted_wins','classification','advances_to_independent14','error')},allow_nan=False),flush=True)


def score_body(args,result,out,pins,h,guard):
    require,sha,read,write,bind,sealed,trial_output = [h[k] for k in
        ('require','sha','read','write','bind','sealed','trial_output')]
    spec = importlib.util.spec_from_file_location('composition_literal_admission',HERE/'build.py')
    builder = importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    context = builder.verify_build(BUILD_SHA,CANDIDATE_SHA,BUILDER_SHA,PROTOCOL_SHA)
    for key in ('source_bindings','input_bindings'):
        bind(context[key],pins)
    acquisitions = {}
    for role in ('zero','terminal'):
        folder = HERE/f'native_{role}_v1'
        digest = getattr(args,role+'_sha256')
        native = sealed(folder,digest,pins)
        watch = sealed(HERE/f'native_{role}_watch_v1',getattr(args,role+'_watch_sha256'),pins)
        ranks = (35,100,35) if role=='zero' else VALID+(VALID[0],)
        require(native['candidate_sha256']==context['candidates'][role]['sha256']
                and native['truth_accessed'] is False and all(native.get(k) is True for k in
                    ('all16_domains_qualified','all47_selectors_qualified','repeat_output_domain_byte_exact',
                     'initialization_ownership_passed','own_complete_bills','sources_unchanged')),
                'Exact composition role/domain/bill ownership')
        require([t['rank'] for t in native['trials']]==list(ranks)
                and len(native['input_preparation'])==len(ranks),'Fixed native sequence')
        for path,digest2 in ((HERE/'native.py',args.native_source_sha256),
                            (HERE/'watch_native.py',args.watch_source_sha256),
                            (HERE/'PROTOCOL.md',PROTOCOL_SHA),(HERE/'build.py',BUILDER_SHA)):
            require(native['source_bindings'][str(path)]==digest2
                    and watch['source_bindings'][str(path)]==digest2,'Actual reviewed worker/watch sources')
        require(watch['child_result_sha256']==digest and watch['whole_owned_tree_retired']
                and watch['completed_bill_count']==len(ranks),'Complete independent resource supervision')
        acquisitions[role] = (folder,native)
    baseline_folder = P1/'native2_terminal_v1'
    baseline = sealed(baseline_folder,P1_NATIVE_SHA,pins)
    require(baseline['candidate_sha256']==P1_SHA and baseline['truth_accessed'] is False
            and [t['rank'] for t in baseline['trials']]==list(VALID+(VALID[0],))
            and baseline['all16_domains_qualified'] and baseline['all47_selectors_qualified']
            and baseline['own_complete_bills'],'Original retained P1 native40 frame')
    pscore = sealed(P1/'score2_v1',P1_SCORE_SHA,pins)
    require(pscore['candidate_sha256']==P1_SHA and pscore['integrity_passed']
            and pscore['advances_to_independent14'] and len(pscore['records'])==40,
            'Existing owned baseline numerical score')
    ppanel_folder = P1/'panel_v1/terminal'
    ppanel = sealed(ppanel_folder,'c8fa1cacfa98d0f45d38235283789cff67c59f32f72fa80a92dd385ed59f8c70',pins)
    require(ppanel['candidate_sha256']==P1_SHA and ppanel['truth_accessed'] is False
            and [t['rank'] for t in ppanel['trials'][:2]]==[35,100],'P1 zero-control outputs35/100')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')),'No numeric import side effects')
    import numpy as np
    require(np.__version__=='2.4.6','Qualified scoring NumPy')
    arrays = {}
    for role,(folder,native) in acquisitions.items():
        arrays[role] = [trial_output(np,folder,t,pins) for t in native['trials']]
        require(arrays[role][0].tobytes()==arrays[role][-1].tobytes(),'Own exact served repeat')
        require(native['trials'][0]['flops']-native['trials'][-1]['flops']==native['initialization_flops'],
                'Own actual cold/repeat initialization, no primary bill subtraction')
    for call,rank in enumerate((35,100,35)):
        t = next(t for t in ppanel['trials'] if t['rank']==rank)
        parent = trial_output(np,ppanel_folder,t,pins)
        require(arrays['zero'][call].tobytes()==parent.tobytes(),'Actual zero/P1 all16 byte parity')
    parents = [trial_output(np,baseline_folder,t,pins) for t in baseline['trials']]
    require(parents[0].tobytes()==parents[-1].tobytes(),'Original P1 repeat bytes')
    for call,rank in enumerate(VALID):
        for role,trial in (('P1',baseline['trials'][call]),('composition',acquisitions['terminal'][1]['trials'][call])):
            require(trial['call']==call and trial['rank']==rank
                    and trial['includes_initialization'] is (call==0),'Matched primary/cold ownership')
        require(arrays['terminal'][call][0].tobytes()==parents[call][0].tobytes(),'Exact Gaussian first layer')
    seal = dict(truth_accessed=False,source_bindings=result['source_bindings'],input_bindings=dict(pins),
                terminal_sha256=args.terminal_sha256,terminal_watch_sha256=args.terminal_watch_sha256,
                candidate_sha256=CANDIDATE_SHA,baseline_native_sha256=P1_NATIVE_SHA,
                all41_terminal_verified=True,zero_all16_P1_bytes_exact=True,repeat_excluded=True)
    write(out/'predictions_and_bills_sealed.json',seal)
    result.update(all_predictions_verified_before_truth=True,
        predictions_and_bills_seal_sha256=sha(out/'predictions_and_bills_sealed.json'))
    records = []
    for call,rank in enumerate(VALID):
        truth = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        expected = pscore['truth_bindings'][str(truth)]
        require(sha(truth)==expected,'Same fixed1e9 reference')
        result['truth_bindings'][str(truth)] = expected;result['truth_accessed'] = True
        with np.load(truth,allow_pickle=False) as file:
            target = file['mc_post'].astype(np.float64)
        require(target.shape==(16,1024) and np.isfinite(target).all(),'All16 finite reference means')
        row = dict(call=call,rank=rank,per_role={})
        for role,value,trial in (('P1',parents[call],baseline['trials'][call]),
                ('composition',arrays['terminal'][call],acquisitions['terminal'][1]['trials'][call])):
            mse = np.mean((value.astype(np.float64)-target)**2,axis=1)
            raw = float(mse[-1]);f = trial['flops'];multiplier = max(.1,f/BUDGET)
            require(math.isfinite(raw) and raw>0,'Finite positive final raw error')
            row['per_role'][role] = dict(raw_mse=raw,flops=f,multiplier=multiplier,
                                        adjusted=raw*multiplier,layer_mse=mse.tolist())
        old = pscore['records'][call]
        require(old['rank']==rank and row['per_role']['P1']==old['per_role']['terminal'],
                'Exactly recomputed original P1 raw, own bill and adjusted result')
        a,b = row['per_role']['composition'],row['per_role']['P1']
        row.update(raw_win=a['raw_mse']<b['raw_mse'],adjusted_win=a['adjusted']<b['adjusted'])
        records.append(row)
    summary = {role:{key:sum(r['per_role'][role][key] for r in records)/40
                     for key in ('raw_mse','flops','adjusted')} for role in ('P1','composition')}
    a,b = summary['composition'],summary['P1']
    wins = sum(r['adjusted_win'] for r in records)
    pareto = a['raw_mse']<=b['raw_mse'] and a['flops']<=b['flops'] and (
        a['raw_mse']<b['raw_mse'] or a['flops']<b['flops'])
    improved = a['adjusted']<b['adjusted']
    bind(result['source_bindings'],{});bind(pins,{})
    require(all(sha(p)==v for p,v in result['truth_bindings'].items()),'Truth files unchanged')
    guard()
    result.update(passed=True,completed=True,hashes_unchanged=True,integrity_passed=True,
        records=records,summary=summary,adjusted_wins=wins,raw_wins=sum(r['raw_win'] for r in records),
        mean_adjusted_ratio=a['adjusted']/b['adjusted'],mean_raw_ratio=a['raw_mse']/b['raw_mse'],
        objective_improved=improved,pareto_improvement=pareto,
        classification='Pareto improvement' if pareto else 'beneficial tradeoff' if improved else 'not advanced',
        advances_to_independent14=improved and wins>=24,
        aggregation='Mean of per-net raw*max(.1,own completeF/2^41); repeat excluded')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
