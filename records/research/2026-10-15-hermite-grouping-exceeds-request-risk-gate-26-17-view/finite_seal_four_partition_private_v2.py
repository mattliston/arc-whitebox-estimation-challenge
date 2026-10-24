"""HELD finite fresh two-look seal; collection and decisions separately granted.
Initial8 has fresh4e and GOP2 measurements, four fixed two-identity shards. Conditional8 only after EXTEND.
No allocation, truth production, release trigger, numerical retry or submission.
"""
import argparse,ast,hashlib,importlib.util,json,math,os,re,signal,statistics,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2];R=O/'release'
Q=O/'lead/runs/astra_gop2_current_delivery_v1'
NATIVE=O/'lead/runs/astra_gop2_current_native10_v1'
SOURCE=Q/'candidate_current4e_GOP2_ON_HELD_v1.py'
SOURCE_SHA='0170697df8866cff2bc43b13fa7a41faf5a8d73a9c527985e12ac8a29a4c89e4'
BASE=O/'lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py'
BASE_SHA='4e1bf080bb6751b487c62842e369a427633e0c4686f089fd3f8e6448bb416f0e'
PY=Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
RT=O.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
JUDGE=O/'scout_native/run_cert_v2.py'
FIRST=list(range(7190064,7190072));EXTRA=list(range(7190072,7190080))
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
STATS=('_ASTRA_COMPUTE_STATS','_ASTRA_BCC_STATS','_ASTRA_BCC_CTX','_ASTRA_BCC_RIGHT','_ASTRA_QL_STATS','_OTT_CONTEXT','_DCR_STASH','_DCR_CALLS','_DCR_OPEN','_ASTRA_UBOWN_STATS','_ASTRA_CLUSTER_STATS','_UBC_STATS','_UBX_STATS','_UBX_RC_STATS','_UBX_STATE','_UBX_RCACHE','_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC','_ASTRA_CLUSTER_CTX')
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');ENV.update({k:'1'for k in THREADS})
for flag in ('PB_ALL','MODE','DRY','STATS'):ENV.pop(flag,None)
ENV.update(PB_ALL='1',STATS=','.join(STATS))
ACTIVE_PROTOCOL=D/'SEAL_PROTOCOL_HELD_v2.json';N=None;STEPS=[];RETIRED=True
def source_for(step):return BASE if step.startswith('base_')else SOURCE
def set_phase(phase,shard):
 global N,STEPS
 assert phase in ('initial8','extension8') and type(shard)is int and shard in(0,1,2,3)
 key=phasekey(phase,shard);N=D/'numerical_v1'/key;STEPS=[f'base_{key}',f'arm_{key}']
def phasekey(phase,shard):
 assert phase in ('initial8','extension8') and type(shard)is int and shard in(0,1,2,3)
 return f'{phase}_s{shard}'
def seeds_for(phase,shard=None):
 assert phase in ('initial8','extension8')
 seeds=FIRST if phase=='initial8'else EXTRA
 if shard is None:return seeds
 assert type(shard)is int and shard in(0,1,2,3)
 return seeds[2*shard:2*shard+2]
def truth_path(seed):
 seed=int(seed)
 if 7190080<=seed<7190088:return Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_seal80_reusable_truth_v1/isolated/analyst/fresh')/f'truth_seed{seed}.npz'
 return O/f'analyst/fresh/truth_seed{seed}.npz'

def command_for(phase,role,shard):
 assert role in('base','arm')
 step=f'{role}_{phasekey(phase,shard)}';out=D/'numerical_v1'/phasekey(phase,shard)/step/'cert.json'
 return [str(RT),'-B',str(JUDGE),str(source_for(step)),','.join(map(str,seeds_for(phase,shard))),str(out)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stamp():
    return subprocess.check_output(['/bin/date', '-u', '+%FT%TZ'], text=True).strip()

def read(path):
    return json.loads(Path(path).read_text())

def exclusive(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')

def clean_guard(path, command=None):
    g = read(path)
    assert g['completed'] is True and g['exit_code'] == 0 and (g.get('error') is None)
    assert g['whole_group_retired'] is True and g['remaining_rows'] == []
    assert g['rss_backstop_bytes'] == g['footprint_trigger_bytes'] == 7500000000
    assert g['max_rss_bytes'] < 7500000000 and g['max_footprint_bytes'] < 7500000000
    if command is not None:
        assert g['command'] == command
    return g

def resources():
    text = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
    free = int(re.search('System-wide memory free percentage: (\\d+)%', text).group(1))
    rows = subprocess.check_output(['/bin/ps', '-axo', 'rss=,comm='], text=True, timeout=8)
    heavy = sum((int(s.split(None, 1)[0]) >= 1000000 and 'python' in s.lower() for s in rows.splitlines() if s.strip()))
    return dict(utc=stamp(), memory_free_percent=free, heavy_python_ge1GB=heavy, counted=heavy + 1, truth_reserve=1)

def done(step):
    p = D / f'SEAL_STEP_{step}_DONE_v1.json'
    if not p.exists():
        return False
    r = read(p)
    for path, expected in r['output_sha256'].items():
        assert sha(path) == expected, ('completed output changed', path)
    return True

def finish(step, paths, details):
    exclusive(D / f'SEAL_STEP_{step}_DONE_v1.json', dict(utc=stamp(), details=details, output_sha256={str(p): sha(p) for p in paths}))

def admission_event(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')

def admit(step):
    assert RETIRED is True and step in STEPS
    started = time.monotonic()
    attempt = 0
    last_status = None
    while True:
        state = resources()
        elapsed = time.monotonic() - started
        attempt += 1
        okay = state['memory_free_percent'] >= 60 and state['counted'] < 10 and (elapsed <= 1800)
        record = dict(step=step, controller_pid=os.getpid(), attempt=attempt, elapsed_seconds=elapsed, admitted=okay, **state)
        admission_event(N / 'admission_wait.jsonl', record)
        if okay:
            admission_event(N / 'admission.jsonl', record)
            return record
        if last_status is None or elapsed - last_status >= 30:
            print(json.dumps(dict(status='waiting_for_prospective_headroom_no_native_child', step=step, memory_free_percent=state['memory_free_percent'], counted=state['counted'], elapsed_seconds=elapsed, slot_reserved=True)), flush=True)
            last_status = elapsed
        assert elapsed < 1800, 'Bounded admission wait expired; no automatic retry'
        time.sleep(min(15, 1800 - elapsed))

def admissions(controller_pid=None):
    controller_pid = os.getpid() if controller_pid is None else controller_pid
    records = [json.loads(x) for x in (N / 'admission.jsonl').read_text().splitlines()]
    assert [x['step'] for x in records] == STEPS
    assert all((x['controller_pid'] == controller_pid and x['memory_free_percent'] >= 60 and (x['counted'] < 10) for x in records))
    attempts = [json.loads(x) for x in (N / 'admission_wait.jsonl').read_text().splitlines()]
    assert [x for x in attempts if x['admitted']] == records
    assert all((x['admitted'] is True and 0 <= x['elapsed_seconds'] <= 1800 for x in records))
    return dict(path=str(N / 'admission.jsonl'), sha256=sha(N / 'admission.jsonl'), records=records, wait_path=str(N / 'admission_wait.jsonl'), wait_sha256=sha(N / 'admission_wait.jsonl'))

def numerical(step, command, seconds, extra_env=None):
    global RETIRED
    pins()
    assert os.getpriority(os.PRIO_PROCESS, 0) == 0 and all((os.environ.get(k) == '1' for k in THREADS))
    assert step in STEPS
    output = N / step
    assert not output.exists()
    output.mkdir()
    guard, log = (output / 'guard.json', output / 'run.log')
    started, owner = (D / f'SEAL_STEP_{step}_STARTED_v1.json', D / f'SEAL_STEP_{step}_OWNER_v1.json')
    assert not any((p.exists() for p in [guard, log, started, owner])), 'unfinished step needs root review/new version'
    before = admit(step)
    pins()
    env = dict(ENV, **extra_env or {})
    args = [str(PY), '-B', str(R / 'guard_footprint_v3.py'), '--record', str(guard), '--seconds', str(seconds), '--footprint-limit', '7500000000', '--rss-backstop', '7500000000', *command]
    exclusive(started, dict(utc=stamp(), command=args, resources=before, source_sha256=sha(source_for(step)), controller_sha256=sha(__file__), env={k: env.get(k) for k in THREADS + ('PB_ALL', 'STATS')}))
    child, failure = (None, None)
    with log.open('x') as stream:
        try:
            child = subprocess.Popen(args, cwd=R, env=env, stdout=stream, stderr=subprocess.STDOUT)
            RETIRED = False
            exclusive(owner, dict(utc=stamp(), controller_pid=os.getpid(), owned_guard_pid=child.pid, command=args, controller_nice=0, guard_nice=0, native_nice=10))
            while child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current = resources()
                    with (N / 'resources_v1.jsonl').open('a') as f:
                        f.write(json.dumps(dict(step=step, **current)) + '\n')
                    assert current['memory_free_percent'] >= 50 and current['counted'] <= 10, ('live resource bound', current)
        except BaseException as error:
            failure = error
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child is None:
                RETIRED = True
            else:
                try:
                    g = read(guard)
                    RETIRED = child.poll() is not None and g.get('whole_group_retired') is True and (g.get('remaining_rows') == [])
                except (OSError, ValueError):
                    RETIRED = False
            if failure is not None:
                exclusive(D / f'SEAL_STEP_{step}_INTERRUPTED_v1.json', dict(utc=stamp(), error=repr(failure), owned_guard_pid=child.pid if child else None, owned_retirement_verified=RETIRED, slot_reservation_retained=not RETIRED))
    if failure is not None:
        raise failure
    assert child.returncode == 0 and RETIRED
    clean_guard(guard, command)
    pins()
    return guard

def decision(differences, phase):
    assert phase in ('initial8', 'extension8')
    assert len(differences) == (8 if phase == 'initial8' else 16)
    assert all((math.isfinite(v) for v in differences))
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(differences))
    verdict = 'PASS' if mean < 0 else 'EXTEND' if phase == 'initial8' and mean < se else 'FAIL'
    return (mean, se, verdict)


def pins():
 c=read(ACTIVE_PROTOCOL)
 assert c['purpose']=='GOP2_current4e_fresh_two_slot_two_look_seal'
 assert c['first8']==FIRST and c['conditional8']==EXTRA and c['truth_N']==15990784
 assert c['fixed_shards']=={phase:[seeds_for(phase,i)for i in(0,1,2,3)]for phase in('initial8','extension8')}and c['max_numerical_slots']==4
 assert c['source_sha256']==sha(SOURCE)==SOURCE_SHA and c['baseline_sha256']==sha(BASE)==BASE_SHA
 assert c['fresh_baseline_required']is True and c['prior_selected8_reused']is False
 assert c['collection_may_overlap_production']is True and c['production_prerequisites_before_grade']is True
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 for p,h in c['source_pins'].items():assert sha(p)==h,('changed seal pin',p)
 return c

def pinned(entry):
 p=Path(entry['path']).resolve();assert sha(p)==entry['sha256'];return read(p)

def registered(path):
 global ACTIVE_PROTOCOL
 p=Path(path).resolve();assert p.parent==D;r=read(p)
 assert r['approved']is True and r['reviewed_by']=='root'and r['action']=='register_GOP2_current4e_fresh_seal_two_look_two_slot'
 assert r['controller_sha256']==sha(__file__)
 ACTIVE_PROTOCOL=Path(r['protocol_path']).resolve();assert ACTIVE_PROTOCOL.parent==D and sha(ACTIVE_PROTOCOL)==r['protocol_sha256']
 assert pins()['status']=='REGISTERED'
 assert r['first8']==FIRST and r['conditional8']==EXTRA and r['truth_N']==15990784
 assert r['sources']==dict(base=dict(path=str(BASE),sha256=BASE_SHA),arm=dict(path=str(SOURCE),sha256=SOURCE_SHA))
 assert r['collection_may_overlap_production']is True and r['production_prerequisites_before_grade']is True
 assert r['no_baseline_outcome_selection']is True and r['prior_selected8_reused']is False
 allocation=pinned(r['allocation'])
 assert allocation['approved']is True and allocation['reviewed_by']=='root'
 assert allocation['purpose']=='GOP2_current4e_seal_initial8_conditional8'
 assert allocation['initial8']==FIRST and allocation['conditional8']==EXTRA
 assert allocation['baseline_candidate']==dict(path=str(BASE),sha256=BASE_SHA)
 assert allocation['arm_candidate']==dict(path=str(SOURCE),sha256=SOURCE_SHA)
 assert allocation['no_prior_candidate_or_baseline_evaluations_on_block']is True
 assert allocation['original_decisions_preserved']is True
 # Same accepted fixed6/source barrier as the scientific collector; no candidate import.
 helper=NATIVE/'finite_gop2_current_native10_HELD_v1.py'
 spec=importlib.util.spec_from_file_location('_seal_qualification_metadata',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 qualified=m.qualification(r)
 r['_registration_sha256']=sha(p);r['_qualification']=qualified
 return r

def prior_extension(r):
 p=D/'DECISION_initial8_v1.json';x=read(p)
 assert x['verdict']=='EXTEND'and x['seeds']==FIRST
 assert x['registration_sha256']==r['_registration_sha256']
 for target,h in x['evidence_sha256'].items():assert sha(target)==h
 return p

def checked(phase,role,shard):
 import numpy as np
 key=phasekey(phase,shard);step=f'{role}_{key}';seeds=seeds_for(phase,shard);source=source_for(step)
 out=D/'numerical_v1'/key/step/'cert.json';guard=out.with_name('guard.json');pred=out.with_name('cert_pred.npz')
 dp=D/f'SEAL_STEP_{step}_DONE_v1.json';sp=D/f'SEAL_STEP_{step}_STARTED_v1.json';op=D/f'SEAL_STEP_{step}_OWNER_v1.json'
 done=read(dp)
 for p,h in done['output_sha256'].items():assert sha(p)==h
 x=read(out);g=clean_guard(guard,command_for(phase,role,shard));start,owner=read(sp),read(op)
 assert x['candidate']==str(source)and x['sha256']==sha(source)
 assert x['seeds']==seeds and[row['seed']for row in x['rows']]==seeds
 assert start['source_sha256']==sha(source)and start['controller_sha256']==sha(__file__)
 assert start['env']=={**{k:'1'for k in THREADS},'PB_ALL':'1','STATS':','.join(STATS)}
 assert owner['command']==start['command']and owner['command'][11:]==command_for(phase,role,shard)
 assert owner['controller_nice']==owner['guard_nice']==0 and owner['native_nice']==10
 with np.load(pred,allow_pickle=False)as z:
  assert set(z.files)==set(map(str,seeds))
  for row in x['rows']:
   y=z[str(row['seed'])]
   assert y.shape==(16,1024)and y.dtype==np.float32 and np.isfinite(y).all()
   assert hashlib.sha256(y.tobytes()).hexdigest()[:16]==row['pred_sha']
   assert len(row['mse_layers'])==16 and all(math.isfinite(v)and v>0 for v in row['mse_layers'])
   assert row['mse_final']==row['mse_layers'][-1]
   assert type(row['flops'])is int and 0.1*2**41<=row['flops']<2**41
   assert type(row['ops'])is int and row['ops']>0
   if role=='arm':
    s=row['stats']
    for k in ('_ASTRA_BCC_CTX','_ASTRA_BCC_RIGHT','_OTT_CONTEXT','_DCR_STASH','_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC','_ASTRA_CLUSTER_CTX'):assert s[k]==[None]
    assert s['_DCR_CALLS']==[]and s['_DCR_OPEN']==[False]
    for k in ('_UBX_STATE','_UBX_RCACHE'):
     a=s[k];assert a['run']is None and a['cache']=={}and a['bytes']==0
    assert s['_UBX_RCACHE']['owner']is None
    a=s['_UBC_STATS'];assert a['retired']is True and a['live_cache_bytes']==a['retained_cache_bytes']==0
    assert a['peak_cache_bytes']<=512000000 and a['max_resident_bytes']<=512000000
 return x['rows'],{str(p):sha(p)for p in [out,guard,pred,dp,sp,op]},(owner['owned_guard_pid'],g['pid'])

def collect(args,r):
 import numpy as np
 set_phase(args.phase,args.shard);key=phasekey(args.phase,args.shard);g=read(args.evidence)
 assert g['approved']is True and g['reviewed_by']=='root'and g['action']=='collect_fixed_fresh_seal_phase'
 assert g['registration_sha256']==r['_registration_sha256']and g['controller_sha256']==sha(__file__)
 assert g['phase']==args.phase and g['numerical_slots']==4 and 1<=g['logical_reserved_after_launch']<=7
 assert g['no_followup_authorization']is True
 assert os.getpriority(os.PRIO_PROCESS,0)==0 and all(os.environ.get(k)=='1'for k in THREADS)
 assert all(os.environ.get(k)is None for k in ('PB_ALL','MODE','DRY','STATS'))
 seeds=seeds_for(args.phase,args.shard)
 if args.phase=='extension8':assert g['initial_decision_sha256']==sha(prior_extension(r))
 assert g['fixed_shards']==[seeds_for(args.phase,i)for i in(0,1,2,3)]
 assert set(g['truth_sha256'])==set(map(str,seeds_for(args.phase)))
 for seed in seeds:
  p=truth_path(seed);assert sha(p)==g['truth_sha256'][str(seed)]
  with np.load(p,allow_pickle=False)as z:assert int(z['seed'])==seed and int(z['N'])==15990784
 assert not N.exists()and not(D/f'PHASE_READY_{key}_v1.json').exists()
 N.mkdir(parents=True);lock=D/f'OWNER_LOCK_{key}_v1';lock.mkdir()
 exclusive(D/f'PHASE_OWNER_{key}_v1.json',dict(utc=stamp(),phase=args.phase,shard=args.shard,controller_pid=os.getpid(),controller_sha256=sha(__file__),root_grant_sha256=sha(args.evidence),registration_sha256=r['_registration_sha256']))
 error=None
 try:
  evidence={};guards=[]
  for role in ('base','arm'):
   step=f'{role}_{key}';cmd=command_for(args.phase,role,args.shard);guard=numerical(step,cmd,7320)
   out=N/step/'cert.json';finish(step,[out,guard,out.with_name('cert_pred.npz')],dict(seeds=seeds,role=role))
   evidence.update(checked(args.phase,role,args.shard)[1]);guards.append(guard)
  admission=admissions()
  assert sorted(N.glob('*/guard.json'))==sorted(guards)
  exclusive(D/f'PHASE_READY_{key}_v1.json',dict(utc=stamp(),phase=args.phase,shard=args.shard,controller_pid=os.getpid(),controller_sha256=sha(__file__),registration_sha256=r['_registration_sha256'],root_grant_sha256=sha(args.evidence),seeds=seeds,steps=STEPS,source_sha256=SOURCE_SHA,baseline_sha256=BASE_SHA,evidence_sha256=evidence,admission_evidence=admission,expected_guards=list(map(str,guards)),no_grade_emitted=True,managed_exit0_required=True,production_prerequisites_before_grade=True))
 except BaseException as e:
  error=e;exclusive(D/f'PHASE_STOPPED_{key}_v1.json',dict(utc=stamp(),error=repr(e),owned_retirement_verified=RETIRED,slot_reservation_retained=not RETIRED,no_retry=True));raise
 finally:
  exclusive(D/f'PHASE_LIFECYCLE_RETIREMENT_{key}_v1.json',dict(utc=stamp(),controller_pid=os.getpid(),error=repr(error)if error else None,all_owned_guards_retired=RETIRED,slot_reservation_retained=not RETIRED))
  if RETIRED:lock.rmdir()

def root_phase_retirement(entry,phase,shard,r):
 key=phasekey(phase,shard)
 root=pinned(entry);ready=D/f'PHASE_READY_{key}_v1.json';x=read(ready)
 assert root['approved']is True and root['reviewed_by']=='root'
 assert root['actual_managed_exit_observed']is True and root['exit_code']==0
 assert root['all_owned_PID_PPID_PGID_absent']is True and root['ready_sha256']==sha(ready)
 assert root['phase']==phase and root['shard']==shard
 assert root['controller_pid']==x['controller_pid']and root['registration_sha256']==r['_registration_sha256']
 assert root['all_expected_rows_arrays_sources_guards_recomputed']is True
 assert x['registration_sha256']==r['_registration_sha256']and x['seeds']==seeds_for(phase,shard) and x['phase']==phase and x['shard']==shard
 assert x['source_sha256']==SOURCE_SHA and x['baseline_sha256']==BASE_SHA and x['no_grade_emitted']is True
 for p,h in x['evidence_sha256'].items():assert sha(p)==h
 assert not(D/f'OWNER_LOCK_{key}_v1').exists()and not(D/f'PHASE_STOPPED_{key}_v1.json').exists()
 ids=set(root['owned_pids_checked_absent']);groups=set(root['owned_pgids_checked_absent'])
 assert x['controller_pid']in ids and len(ids)>=5 and len(groups)==2
 rows=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()]
 assert not[row for row in rows if row[0]in ids or row[1]in ids or row[2]in groups]
 return {str(ready):sha(ready),entry['path']:entry['sha256']}

def production_acceptance(g):
 p=Path(g['production_root_acceptance']['path']).resolve();assert sha(p)==g['production_root_acceptance']['sha256']
 x=read(p);assert x['approved']is True and x['reviewed_by']=='root'and x['candidate_sha256']==SOURCE_SHA
 for k in('normal_pass','slow_pass','own20_accepted','both_actual_managed_exits_observed','all_owned_PID_PPID_PGID_absent'):assert x[k]is True
 assert x['evidence_sha256']
 for path,h in x['evidence_sha256'].items():assert sha(path)==h
 return dict(path=str(p),sha256=sha(p),candidate_sha256=SOURCE_SHA)

def grade(args,r):
 g=read(args.evidence)
 assert g['approved']is True and g['reviewed_by']=='root'and g['action']=='grade_complete_fresh_seal_phase'
 assert g['numerical_slots']==0 and g['registration_sha256']==r['_registration_sha256']and g['controller_sha256']==sha(__file__)
 assert g['phase']==args.phase
 # Complete final production is required before any seal decision.
 boundary=production_acceptance(g)
 if args.phase=='extension8':assert g['initial_decision_sha256']==sha(prior_extension(r))
 phases=['initial8']if args.phase=='initial8'else['initial8','extension8']
 assert set(g['phase_retirements'])==set(phases)
 pairs={role:[]for role in ('base','arm')};evidence={str(args.evidence):sha(args.evidence)}
 # Validate every shard's complete owner barrier before reading comparative rows.
 for phase in phases:
  assert set(g['phase_retirements'][phase])=={'0','1','2','3'}
  for shard in(0,1,2,3):evidence.update(root_phase_retirement(g['phase_retirements'][phase][str(shard)],phase,shard,r))
 for phase in phases:
  for shard in(0,1,2,3):
   for role in pairs:
    rows,ev,_=checked(phase,role,shard);pairs[role].extend(rows);evidence.update(ev)
 seeds=FIRST if args.phase=='initial8'else FIRST+EXTRA
 assert all([row['seed']for row in pairs[role]]==seeds for role in pairs)
 differences=[math.log(a['mse_final']/b['mse_final'])+math.log(a['flops']/b['flops'])for b,a in zip(pairs['base'],pairs['arm'])]
 mean,se,verdict=decision(differences,args.phase)
 exclusive(D/f'DECISION_{args.phase}_v1.json',dict(utc=stamp(),registration_sha256=r['_registration_sha256'],seeds=seeds,n=len(seeds),mean_log_net=mean,se_log_net=se,verdict=verdict,per_seed_log_net=differences,evidence_sha256=evidence,production=boundary,release_or_fire_authorized=False))
 print(json.dumps(dict(phase=args.phase,n=len(seeds),verdict=verdict,mean_log_net=mean,se_log_net=se)))

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','grade']);p.add_argument('phase',choices=['initial8','extension8']);p.add_argument('registration',type=Path);p.add_argument('evidence',type=Path);p.add_argument('--shard',type=int,choices=[0,1,2,3]);a=p.parse_args()
 assert (a.shard in(0,1,2,3))if a.action=='collect'else a.shard is None
 r=registered(a.registration)
 return collect(a,r)if a.action=='collect'else grade(a,r)

def interrupted(signum,frame):raise KeyboardInterrupt('Owned seal controller signal%s'%signum)
if __name__=='__main__':
 previous=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,previous)
