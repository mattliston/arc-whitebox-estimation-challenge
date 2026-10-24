"""One fixed persistent NumPy-free series, with independent whole-tree limits."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
RUNNER = NEW/'run_remote_series_v1.py'
CLIENT = NEW/'check_remote_series_v1.py'
RUNNER_SHA = '00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1'
CLIENT_SHA = 'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55'
CANDIDATE = 'current_comp_partial_d6_v1/build_v1/estimator.py'
CANDIDATE_SHA = 'ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d'
TAG = 'current_comp_partial_d6_production_v1'
RANKS = (35,100,500,3016500,35)
SCHEMA = 'current_comp_partial_d6_production_watch_v1'
INIT, BUDGET = 1944970298, 2**41
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
LAUNCH_ENV = {**dict.fromkeys(BLAS,'1'),'PYTHONDONTWRITEBYTECODE':'1',
              'FLOPSCOPE_MAX_ARRAY_BYTES':'4294967296'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def transport():
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == TRANSPORT_SHA,
            'Frozen independently qualified observation transport')
    spec = importlib.util.spec_from_file_location('cd_production_transport',TRANSPORT)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    mon,classifier = helper.transport()
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),
            'Supervisor is stdlib-only, no numerical runtime import')
    return helper,mon,classifier


def check_bill(row, index):
    require(type(index) is int and 0 <= index < 5 and row['index'] == index
        and row['rank'] == RANKS[index] and row['output_shape'] == [16,1024]
        and row['output_dtype'] == 'float32','Exact five-call context/output identities')
    f,t = row['flops'],row['timing']
    require(type(f) is int and 0 < f <= BUDGET and row['util'] == f/BUDGET,
            'Own complete context FLOPs, no subtracted preparation or startup')
    require(isinstance(t,dict) and t and all(type(v) in (int,float)
        and math.isfinite(v) and v >= 0 for v in t.values())
        and 0 < t['wall_time_s'] < 60 and t['residual_wall_time_s'] < .4,
        'Every original production wall<60s/residual<.4s gate')
    require(re.fullmatch('[0-9a-f]{64}',row['output_sha256'] or ''),
            'Actual returned output SHA')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    keys = ('source','checker','protocol','spec','launch','preparation')
    for key in keys:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = previous = last_progress = time.monotonic()
    require(args.numerical_slot_clear and os.getpriority(os.PRIO_PROCESS,0) >= 10
        and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and all(os.environ.get(k) == v for k,v in LAUNCH_ENV.items()),
        'Exclusive nice10/BLAS1/bytecodeoff/assertions/full-array-bound launch')
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in keys),
            'Exact reviewed owners and actual successful launch/preparation hashes')
    h,mon,classifier = transport()
    own = {Path(__file__).resolve():args.source_sha256,
        HERE/'production_check_v2.py':args.checker_sha256,
        HERE/'PRODUCTION_PROTOCOL_V2.md':args.protocol_sha256,
        HERE/'PRODUCTION_WATCH_SPEC_V2.md':args.spec_sha256,
        RUNNER:RUNNER_SHA,CLIENT:CLIENT_SHA,NEW/CANDIDATE:CANDIDATE_SHA,
        TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}
    prep = HERE/'production_v1/prepare'
    launch_path = prep/'launch.json'
    inputs = {**h.MOCK_PINS,launch_path:args.launch_sha256,
              prep/'result.json':args.preparation_sha256}
    out = HERE/'production_watch_v1'
    series = NEW/(TAG+'.json')
    usage = NEW/(TAG+'_resources.json')
    server_log = NEW/(TAG+'_server.log')
    outputs = [NEW/f'{TAG}_{i}_r{r}.f32' for i,r in enumerate(RANKS)]
    require(not out.exists() and not any(p.exists() for p in (series,usage,server_log,*outputs))
        and not HERE.is_symlink(),'Exclusive fresh production series and watch')
    out.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA,passed=False,completed=False,hashes_unchanged=False,
        sources_unchanged=False,candidate_sha256=CANDIDATE_SHA,candidate=CANDIDATE,tag=TAG,
        launch_sha256=args.launch_sha256,preparation_sha256=args.preparation_sha256,
        source_bindings={str(p):v for p,v in own.items()},
        input_bindings={str(p):v for p,v in inputs.items()},artifacts={},ranks=list(RANKS),
        numerical_imports=False,truth_arrays_accessed=False,official_score=False,
        uploaded=False,submission_authorized=False,native_fidelity_qualified=False,
        whole_owned_tree_retired=False,child_execution_completed=False,
        maximum_combined_rss_bytes=0,maximum_interval_s=0.,maximum_swap_growth_bytes=0.,
        completed_sample_intervals_s=[],observed_records=[],full_model_calls=0,
        whole_cap_s=600.,progress_cap_s=120.,combined_rss_limit_bytes=8_000_000_000,
        numerical_slot_clear_attested=True)
    child=log=reader=stream=None; swap0=None; samples=0; pending=''

    def deadline():
        now = time.monotonic()
        require(now-began < 600. and now-last_progress < 120.,
                'Whole600s/next-complete-context120s through finalization')

    def observe():
        nonlocal previous,last_progress,swap0,samples,pending
        absolute = min(previous+1.,began+600.,last_progress+120.)
        snapshot = mon.collect(absolute)
        scope = mon.scope(snapshot['rows'],os.getpid(),None if child is None else child.pid)
        allowed = scope.pop('allowed')
        require(not classifier(snapshot['rows'],allowed),'No foreign numerical worker')
        for row in snapshot['rows']:
            if row['pid'] not in allowed:
                try:
                    argv = shlex.split(row['command'])
                except ValueError:
                    argv = row['command'].split()
                require('flopscope_server' not in argv,'No separate foreign numerical server')
        if swap0 is None:
            swap0 = snapshot['swap']['used_bytes']
        growth = snapshot['swap']['used_bytes']-swap0
        require(growth <= 0 and 0 < scope['combined_rss_bytes'] < 8_000_000_000,
                'Zero added system swap and full owned tree plus watch<8GB')
        if reader is not None:
            pending += reader.read()
            require(len(pending) < 2*1024**2,'Bounded original runner stdout tail')
            while '\n' in pending:
                line,pending = pending.split('\n',1)
                try:
                    row = json.loads(line)
                except (ValueError,TypeError):
                    continue
                if isinstance(row,dict) and 'index' in row and 'rank' in row:
                    deadline(); index = len(result['observed_records'])
                    check_bill(row,index)
                    path = outputs[index]
                    require(path.stat().st_size == 16*1024*4
                        and h.sha(path) == row['output_sha256'],
                        'Completed context, saved output and full actual bill precede progress')
                    result['observed_records'].append(row)
                    result['full_model_calls'] = index+1
                    last_progress = time.monotonic()
        require(snapshot['commands_reaped'] and not mon.PENDING,'Observation groups retired')
        deadline()
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,**scope,
            swap_bytes=snapshot['swap']['used_bytes'],swap_growth_bytes=growth,
            completed_calls=len(result['observed_records']),transport=snapshot['command_records'],
            uncached_single_process_table=True),allow_nan=False)+'\n')
        stream.flush()
        previous,interval = mon.complete(previous,absolute)
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval)
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        result['completed_sample_intervals_s'].append(interval); samples += 1

    def tick():
        deadline()
        if time.monotonic()-previous >= .2:
            observe()

    def interrupted(*unused):
        raise RuntimeError('Owned fixed production series interrupted')

    handlers = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        stream = (out/'samples.jsonl').open('x'); observe()
        for p,digest in {**own,**inputs}.items():
            require(h.sha(p,tick) == digest,'Reviewed fixed launch/source: '+str(p))
        rp,ep,fp = (prep/name for name in ('result.json','completion.json','finalization.json'))
        r,e,f = map(h.read,(rp,ep,fp))
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == args.preparation_sha256
            and f['completion_sha256'] == h.sha(ep,tick)
            and r['launch_sha256'] == args.launch_sha256,'Actual successful preparation3 chain')
        for p in (ep,fp):
            inputs[p] = h.sha(p,tick)
        launch = h.read(launch_path)
        argv = [str(OFFPY),'-B','-u',str(RUNNER),'--candidate',CANDIDATE,'--tag',TAG]
        require(launch['schema'] == 'current_comp_partial_d6_production_launch_v1'
            and launch['candidate_sha256'] == CANDIDATE_SHA and launch['candidate'] == CANDIDATE
            and launch['tag'] == TAG and launch['ranks'] == list(RANKS)
            and launch['command'] == argv and launch['cwd'] == str(NEW)
            and launch['env'] == LAUNCH_ENV and launch['initialization_flops'] == INIT
            and launch['uploaded'] is False and launch['estimator_executed'] is False
            and launch['truth_accessed'] is False and not launch.get('submission_authorized',False)
            and not launch.get('user_explicit_confirmation_bank_authorized',False),
            'Exact fixed preparation authority, not release authority')
        for group in ('source_bindings','input_bindings'):
            require(isinstance(launch[group],dict) and launch[group],'Complete source/input map')
            for p,digest in launch[group].items():
                require(p not in result[group] or result[group][p] == digest,'Consistent launch binding')
                require(h.sha(p,tick) == digest,'Qualified launch source/input bytes')
                result[group][p] = digest
        require(all(launch['source_bindings'].get(str(p)) == digest for p,digest in own.items()
            if p in (Path(__file__).resolve(),HERE/'production_check_v2.py',HERE/'PRODUCTION_PROTOCOL_V2.md',
                      HERE/'PRODUCTION_WATCH_SPEC_V2.md',RUNNER,CLIENT,NEW/CANDIDATE)),
            'All production owners were pinned before launch')
        for rank in set(RANKS):
            item = launch['fixture_bindings'][str(rank)]
            require(h.sha(item['path'],tick) == item['sha256'],'Exact native panel physical weight fixture')
            result['input_bindings'][item['path']] = item['sha256']
        env = dict(os.environ,**LAUNCH_ENV,PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
        log = (out/'runner.log').open('x'); reader = (out/'runner.log').open()
        child = subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result.update(child_pid=child.pid,command=argv)
        h.write(out/'started.json',result)
        while child.poll() is None:
            observe(); time.sleep(.15)
        observe()
        require(child.wait(timeout=1) == 0 and result['full_model_calls'] == 5,'Exactly five complete original contexts')
        mon.cleanup([child],time.monotonic()+.5)
        measured,usage_record = h.read(series),h.read(usage)
        require(measured['status'] == 'completed' and measured['source_sha256'] == CANDIDATE_SHA
            and measured['candidate'] == usage_record['candidate'] == CANDIDATE
            and measured['records'] == result['observed_records'] and measured['repeated_output_byte_identical'],
            'Actual complete original-client series matches all observed bills')
        require(measured['records'][0]['output_sha256'] == measured['records'][-1]['output_sha256']
            and measured['records'][0]['flops']-measured['records'][-1]['flops'] == INIT,
            'Original output repeat and complete once-only initialization including D6 bindings')
        inner = usage_record['samples']
        require(inner and 0 < usage_record['peak_combined_rss_bytes'] == max(x[1] for x in inner) < 8_000_000_000,
            'Original leader-only sampler also passes its tighter8GB qualification')
        for p in (series,usage,server_log,*outputs):
            result['artifacts'][str(p)] = h.sha(p,tick)
        for p,digest in {**result['source_bindings'],**result['input_bindings'],
                         **{str(p):v for p,v in inputs.items()},**result['artifacts']}.items():
            require(h.sha(p,tick) == digest,'Final fresh complete nontarget source/input/output hash')
        result['input_bindings'].update({str(p):v for p,v in inputs.items()})
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            child_execution_completed=True,whole_owned_tree_retired=True,
            actual_completed_samples=True,foreign_numerical_workers_checked=True,
            all5_complete=True,initialization_flops=INIT,repeat_excluded=True,
            original_numerical_runner_and_client_unchanged=True)
        observe()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            if child is not None:
                mon.cleanup([child],time.monotonic()+.5)
            mon.retire_pending(time.monotonic()+.5)
            result.update(cleanup_passed=True,whole_owned_tree_retired=True,observational_groups_retired=True)
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for handle in (reader,log,stream):
            if handle is not None:
                handle.close()
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=swap0)
        if result['elapsed_s'] >= 600. or time.monotonic()-last_progress >= 120.:
            result.update(passed=False,completed=False,error='Whole/progress deadline including cleanup')
        for name in ('started.json','runner.log','samples.jsonl'):
            p = out/name
            if p.exists():
                result['artifacts'][str(p)] = h.sha(p)
        h.write(out/'result.json',result); rh = h.sha(out/'result.json')
        elapsed = time.monotonic()-began
        ok = bool(result['passed'] and elapsed < 600. and time.monotonic()-last_progress < 120.)
        h.write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
        eh = h.sha(out/'completion.json'); elapsed = time.monotonic()-began
        ok = bool(ok and elapsed < 600. and time.monotonic()-last_progress < 120.)
        h.write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,
            completion_sha256=eh,elapsed_s=elapsed))
        h.sha(out/'finalization.json')
        ok = bool(ok and time.monotonic()-began < 600. and time.monotonic()-last_progress < 120.)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','full_model_calls',
        'maximum_combined_rss_bytes','maximum_interval_s','maximum_swap_growth_bytes','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

