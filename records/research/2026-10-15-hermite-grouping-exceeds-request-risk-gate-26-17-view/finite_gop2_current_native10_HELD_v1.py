"""HELD independent same-physics release measurement: ten identities, ten calls.
Two serial five-network processes. Root-approved independent census-a supplies same-physics BASEQ; accepted current4e ten is the improvement reference. No comparison, release, seal or submission.
"""
import argparse,ast,hashlib,importlib.util,json,math,os,re,signal,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2];P2=O.parent;R=O/'release'
B=O/'lead/runs/astra_gop2_current_boundary_v1'
Q=O/'lead/runs/astra_gop2_current_delivery_v1'
SOURCE=Q/'candidate_current4e_GOP2_ON_HELD_v1.py'
SOURCE_SHA='0170697df8866cff2bc43b13fa7a41faf5a8d73a9c527985e12ac8a29a4c89e4'
BASE_SOURCE=SOURCE # independent census-a dataset, same immutable physics
BASE_SHA=SOURCE_SHA
PY=Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
RT=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
RANKS=[651,378,424,328,287,641,548,557,316,390]
HALVES=[RANKS[:5],RANKS[5:]]
STEPS=['arm_a1','arm_a2']
N=D/'numerical_v1';ACTIVE_PROTOCOL=D/'NATIVE_PROTOCOL_HELD_v1.json'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');ENV.update({k:'1'for k in THREADS})
for flag in ('PB_ALL','MODE','DRY','STATS'):ENV.pop(flag,None)
RETIRED=True

def source_for(step):
 assert step in STEPS
 return BASE_SOURCE if step.startswith('base_') else SOURCE

def ranks_for(step):
 assert step in STEPS
 return HALVES[int(step[-1])-1]

def command_for(step):
 return [str(RT),'-B',str(O/'engineer/quick.py'),str(source_for(step)),','.join(map(str,ranks_for(step))),str(N/step/'quick.json')]


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
    assert g['completed'] is True and g['exit_code'] == 0 and g.get('error') is None
    assert g['whole_group_retired'] is True and g['remaining_rows'] == []
    assert g['rss_backstop_bytes'] == g['footprint_trigger_bytes'] == 7500000000
    assert g['max_rss_bytes'] < 7500000000 and g['max_footprint_bytes'] < 7500000000
    if command is not None:
        assert g['command'] == command
    return g

def resources():
    text = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', text).group(1))
    rows = subprocess.check_output(['/bin/ps', '-axo', 'rss=,comm='], text=True, timeout=8)
    heavy = sum(int(s.split(None, 1)[0]) >= 1000000 and 'python' in s.lower() for s in rows.splitlines() if s.strip())
    return dict(utc=stamp(), memory_free_percent=free, heavy_python_ge1GB=heavy, counted=heavy + 1, truth_reserve=1)

def done(step):
    p = D / f'NATIVE_STEP_{step}_DONE_v1.json'
    if not p.exists():
        return False
    r = read(p)
    for path, expected in r['output_sha256'].items():
        assert sha(path) == expected, ('completed output changed', path)
    return True

def finish(step, paths, details):
    exclusive(D / f'NATIVE_STEP_{step}_DONE_v1.json', dict(utc=stamp(), details=details,
              output_sha256={str(p): sha(p) for p in paths}))

def admission_event(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')

def admit(step):
    # Keep the granted slot reserved; no native child exists during this wait.
    assert RETIRED is True and step in STEPS
    started=time.monotonic();attempt=0;last_status=None
    while True:
        state=resources();elapsed=time.monotonic()-started;attempt+=1
        okay=state['memory_free_percent']>=60 and state['counted']<8 and elapsed<=1800
        record=dict(step=step,controller_pid=os.getpid(),attempt=attempt,
                    elapsed_seconds=elapsed,admitted=okay,**state)
        admission_event(N/'admission_wait.jsonl',record)
        if okay:
            admission_event(N/'admission.jsonl',record)
            return record
        if last_status is None or elapsed-last_status>=30:
            print(json.dumps(dict(status='waiting_for_prospective_headroom_no_native_child',
                  step=step,memory_free_percent=state['memory_free_percent'],counted=state['counted'],
                  elapsed_seconds=elapsed,slot_reserved=True)),flush=True)
            last_status=elapsed
        assert elapsed<1800, 'Bounded admission wait expired; no automatic retry'
        time.sleep(min(15,1800-elapsed))

def admissions(controller_pid=None):
    controller_pid=os.getpid() if controller_pid is None else controller_pid
    records=[json.loads(x)for x in (N/'admission.jsonl').read_text().splitlines()]
    assert [x['step']for x in records]==STEPS
    assert all(x['controller_pid']==controller_pid and x['memory_free_percent']>=60 and x['counted']<8 for x in records)
    attempts=[json.loads(x)for x in (N/'admission_wait.jsonl').read_text().splitlines()]
    assert [x for x in attempts if x['admitted']] == records
    assert all(x['admitted'] is True and 0<=x['elapsed_seconds']<=1800 for x in records)
    return dict(path=str(N/'admission.jsonl'),sha256=sha(N/'admission.jsonl'),records=records,
                wait_path=str(N/'admission_wait.jsonl'),wait_sha256=sha(N/'admission_wait.jsonl'))

def numerical(step, command, seconds, extra_env=None):
    global RETIRED
    pins()
    assert os.getpriority(os.PRIO_PROCESS, 0) == 0 and all(os.environ.get(k) == '1' for k in THREADS)
    assert step in STEPS
    output=N/step
    assert not output.exists()
    output.mkdir()
    guard, log = output/'guard.json', output/'run.log'
    started, owner = D / f'NATIVE_STEP_{step}_STARTED_v1.json', D / f'NATIVE_STEP_{step}_OWNER_v1.json'
    assert not any(p.exists() for p in [guard, log, started, owner]), 'unfinished step needs root review/new version'
    before=admit(step)
    pins()
    env = dict(ENV, **(extra_env or {}))
    args = [str(PY), '-B', str(R / 'guard_footprint_v3.py'), '--record', str(guard),
            '--seconds', str(seconds), '--footprint-limit', '7500000000', '--rss-backstop', '7500000000', *command]
    exclusive(started, dict(utc=stamp(), command=args, resources=before, source_sha256=sha(source_for(step)),
                           controller_sha256=sha(__file__), env={k: env.get(k) for k in THREADS + ('PB_ALL', 'STATS')}))
    child, failure = None, None
    with log.open('x') as stream:
        try:
            child = subprocess.Popen(args, cwd=R, env=env, stdout=stream, stderr=subprocess.STDOUT)
            RETIRED = False
            exclusive(owner, dict(utc=stamp(), controller_pid=os.getpid(), owned_guard_pid=child.pid, command=args,
                                  controller_nice=0, guard_nice=0, native_nice=10))
            while child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current = resources()
                    with (D / 'native_resources_v1.jsonl').open('a') as f:
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
                    RETIRED = child.poll() is not None and g.get('whole_group_retired') is True and g.get('remaining_rows') == []
                except (OSError, ValueError):
                    RETIRED = False
            if failure is not None:
                exclusive(D / f'NATIVE_STEP_{step}_INTERRUPTED_v1.json', dict(utc=stamp(), error=repr(failure),
                          owned_guard_pid=child.pid if child else None, owned_retirement_verified=RETIRED,
                          slot_reservation_retained=not RETIRED))
    if failure is not None:
        raise failure
    assert child.returncode == 0 and RETIRED
    clean_guard(guard, command)
    pins()
    return guard

def quick_checked(path,source,ranks):
    import numpy as np
    x=read(path)
    assert x['candidate']==str(source) and sha(source) in (SOURCE_SHA,BASE_SHA)
    rows=x['rows'];assert len(rows)==len(ranks) and [r['rank'] for r in rows]==ranks
    assert len({r['rank'] for r in rows})==len(ranks)
    evidence={str(path):sha(path)}
    for r in rows:
        assert 'error' not in r and 'errors' not in r
        assert math.isfinite(float(r['mse'])) and float(r['mse'])>0
        assert math.isfinite(float(r['flops'])) and float(r['flops'])>0 and int(r['ops'])>0
        p=Path(path).with_name(Path(path).stem+f"_pred_{r['rank']}.npy")
        a=np.load(p,allow_pickle=False)
        assert a.shape==(16,1024) and a.dtype==np.dtype('float32') and np.isfinite(a).all()
        assert hashlib.sha256(a.tobytes()).hexdigest()[:16]==r['pred_sha']
        evidence[str(p)]=sha(p)
    return x,evidence


def pins():
 c=read(ACTIVE_PROTOCOL)
 assert c['purpose']=='GOP2_current4e_final_native10_with_census_samephysics_BASEQ'
 assert c['ranks']==RANKS and c['halves']==HALVES and c['steps']==STEPS
 assert c['distinct_network_identities']==10 and c['total_native_calls']==10
 assert c['independent_release_measurement_required']is True and c['no_historical_bills_or_timings']is True
 assert c['collection_may_overlap_production']is True and c['production_prerequisites_before_comparison']is True
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 for p,h in c['source_pins'].items():assert sha(p)==h,('changed pin',p)
 assert sha(SOURCE)==SOURCE_SHA and sha(BASE_SOURCE)==BASE_SHA
 assert SOURCE.read_bytes().startswith(BASE_SOURCE.read_bytes())
 assert c['same_physics_baseline_source_sha256']==BASE_SHA and c['source_sha256']==SOURCE_SHA
 assert c['comparison_authority']=='unchanged compare_twin_native_v5 against independently measured census-a GOP2 ten after full own20 root custody and production gates'
 return c

def prior_boundary(g):
 rp=Path(g['boundary_root_retirement_path']).resolve();ap=B/'ASTRA_BOUNDARY_OWNER_RETIREMENT_v1.json'
 cp=B/'BOUNDARY_COMPLETE_HELD_v1.json'
 assert rp.parent==B and sha(rp)==g['boundary_root_retirement_sha256']
 assert sha(ap)==g['boundary_owner_retirement_sha256'] and sha(cp)==g['boundary_completion_sha256']
 root,a,complete=read(rp),read(ap),read(cp)
 assert root['approved']is True and root['reviewed_by']=='root'
 assert root['owner_retirement_sha256']==sha(ap) and root['complete_sha256']==sha(cp)
 assert root['actual_managed_exit_observed']is True and root['exit_code']==0
 assert root['all_owned_PID_PPID_PGID_absent']is True
 assert a['status']=='IVD_FHM_DENSE_NORMAL_SLOW_COMPLETE_OWNER_REAPED_HELD'
 assert a['candidate_sha256']==SOURCE_SHA and a['complete_sha256']==sha(cp)
 assert a['controller_pid']==root['controller_pid']==complete['controller_pid']
 assert a['managed_session_id']==root['managed_session_id'] and a['actual_exit_observed']is True and a['exit_code']==0
 assert a['all_owned_PID_PPID_PGID_absent']is True and a['remaining_owned_rows']==[]
 assert a['controller_sha256']==sha(B/'finite_gop2_current_boundary_HELD_v1.py')
 assert complete['candidate_sha256']==SOURCE_SHA and complete['all_owned_guards_retired']is True
 assert complete['status']=='NORMAL_SLOW_COMPLETE_HELD_BEFORE_NATIVE10_SEAL_RELEASE'
 for p,h in a['evidence_sha256'].items():assert sha(p)==h,('changed boundary evidence',p)
 assert not(B/'OWNER_LOCK_v1').exists()
 rows=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()]
 ids=set(a['owned_pids_checked_absent']);groups=set(a['owned_pgids_checked_absent'])
 assert not[r for r in rows if r[0]in ids or r[1]in ids or r[2]in groups]
 return dict(root_path=str(rp),root_sha256=sha(rp),owner_path=str(ap),owner_sha256=sha(ap),complete_path=str(cp),complete_sha256=sha(cp))

def qualification(g):
 rp=Path(g['qualification_root_acceptance_path']).resolve();qp=Path(g['qualification_protocol_path']).resolve()
 assert rp.parent==Q and qp.parent==Q
 assert sha(rp)==g['qualification_root_acceptance_sha256']and sha(qp)==g['qualification_protocol_sha256']
 a=read(rp)
 assert a['approved']is True and a['reviewed_by']=='root'
 assert a['actual_managed_exit_observed']is True and a['exit_code']==0
 assert a['all_owned_PID_PPID_PGID_absent']is True and a['all_fixed_qualification_checks_recomputed']is True
 assert a['candidate_sha256']==sha(SOURCE)and a['protocol_sha256']==sha(qp)
 assert a['complete_calls']==2 and a['all16_finite']is True and a['first_layer_exact']is True and a['fifteen_routes_and_cleanup']is True and a['setup_identical']is True
 q=read(qp);assert q['candidates']['ON']['sha256']==sha(SOURCE)
 assert q['purpose']=='gop2_current4e_qualification2'and q['new_native_calls_max']==2
 for p,h in q['source_pins'].items():assert sha(p)==h,('qualification source drift',p)
 return dict(root_path=str(rp),root_sha256=sha(rp),protocol_path=str(qp),protocol_sha256=sha(qp),candidate_sha256=sha(SOURCE))

def baseline_reuse(g):
 C=O/'lead/runs/astra_gop2_current_own20_v1'
 old=O/'lead/runs/astra_iop_off_native10_v1'
 p=Path(g['baseline_reuse_root_path']).resolve()
 assert p.parent==D and sha(p)==g['baseline_reuse_root_sha256']
 r=read(p);assert r['approved']is True and r['reviewed_by']=='root'
 assert r['scope']=='independent_GOP2_census_a_MSE_predictions_samephysics_BASEQ_and_accepted_current4e_ten_only'
 assert r['new_arm_source_sha256']==SOURCE_SHA and r['no_outcome_selection']is True
 assert r['keys_wrapper_host_metadata_only_no_added_FNP_math']is True
 assert r['no_census_timing_or_bill_production_transfer']is True and r['independent_new_final_five_plus_five_required']is True
 rp=Path(r['own20_root_retirement_path']).resolve();assert rp.parent==C and sha(rp)==r['own20_root_retirement_sha256']
 root=read(rp);assert root['approved']is True and root['reviewed_by']=='root'
 assert root['actual_managed_exit_observed']is True and root['exit_code']==0 and root['all_owned_PID_PPID_PGID_absent']is True
 ap=C/'ASTRA_CENSUS_OWNER_RETIREMENT_v1.json';cp=C/'CENSUS_COMPLETE_HELD_v1.json'
 assert sha(ap)==root['owner_retirement_sha256']and sha(cp)==root['complete_sha256']
 a=read(ap);complete=read(cp)
 assert a['candidate_sha256']==SOURCE_SHA and a['all20_complete_all16_finite_actual_keys']is True and a['all_owned_PID_PPID_PGID_absent']is True
 assert complete['candidate_sha256']==SOURCE_SHA and a['complete_sha256']==sha(cp)
 assert a['controller_sha256']==sha(C/'finite_gop2_current_census_HELD_v1.py')
 for path,h in a['evidence_sha256'].items():assert sha(path)==h
 base=C/'numerical_v1/census_a.json'
 assert r['datasets']['same_physics']==dict(path=str(base),sha256=sha(base))
 assert a['evidence_sha256'][str(base)]==sha(base)and base.resolve()!=N/'quick_arm_ten.json'
 bm=json.loads(base.read_text(),parse_float=lambda _:None)
 assert bm['candidate']==str(SOURCE)and [x['rank']for x in bm['rows']]==RANKS
 for rank in RANKS:
  f=base.with_name(base.stem+f'_pred_{rank}.npy');assert a['evidence_sha256'][str(f)]==sha(f)
 rp4=old/'ROOT_NATIVE_OWNER_RETIREMENT_v1.json'
 assert sha(rp4)==r['current4e_native_root_retirement_sha256']=='175a3c79b89959de107409a881a78514d9f2c748335bd11630874d406288ffa9'
 oldr=read(rp4);assert oldr['approved']is True and oldr['reviewed_by']=='root'
 assert oldr['actual_managed_exit_observed']is True and oldr['exit_code']==0 and oldr['all_owned_PID_PPID_PGID_absent']is True
 assert oldr['candidate_sha256']=='4e1bf080bb6751b487c62842e369a427633e0c4686f089fd3f8e6448bb416f0e'
 assert oldr['distinct_network_identities']==10 and oldr['total_native_calls']==10
 for path,h in oldr['evidence_sha256'].items():assert sha(path)==h
 cb=old/'numerical_v1/quick_arm_ten.json'
 assert r['datasets']['current_best']==dict(path=str(cb),sha256=sha(cb))and oldr['evidence_sha256'][str(cb)]==sha(cb)
 for rank in RANKS:
  f=cb.with_name(cb.stem+f'_pred_{rank}.npy');assert oldr['evidence_sha256'][str(f)]==sha(f)
 assert base.resolve()!=cb.resolve()!=N/'quick_arm_ten.json'
 return dict(root_path=str(p),root_sha256=sha(p),own20_root_sha256=sha(rp),current4e_native_root_sha256=sha(rp4),datasets=r['datasets'],same_physics_baseline_independent=True)

def grant(path):
 global ACTIVE_PROTOCOL
 path=Path(path).resolve();assert path.parent==D
 g=read(path)
 assert g['approved']is True and g['reviewed_by']=='root'
 assert g['purpose']=='GOP2_current4e_final_native10_with_census_samephysics_BASEQ'
 ACTIVE_PROTOCOL=Path(g['protocol_path']).resolve();assert ACTIVE_PROTOCOL.parent==D
 assert sha(ACTIVE_PROTOCOL)==g['protocol_sha256']and sha(__file__)==g['controller_sha256']
 c=pins();assert c['status']=='REGISTERED'
 assert g['numerical_slots']==1 and g['steps']==STEPS and g['ranks']==RANKS
 assert g['distinct_network_identities']==10 and g['total_native_calls']==10
 assert g['source_sha256']==SOURCE_SHA and g['baseline_sha256']==BASE_SHA
 assert g['independent_release_measurement_required']is True and g['no_followup_authorization']is True
 assert g['collection_may_overlap_production']is True and g['production_prerequisites_before_comparison']is True
 assert type(g['logical_reserved_after_launch'])is int and 1<=g['logical_reserved_after_launch']<=8
 assert all(os.environ.get(k)=='1'for k in THREADS)and all(os.environ.get(k)is None for k in('PB_ALL','MODE','DRY','STATS'))
 assert os.getpriority(os.PRIO_PROCESS,0)==0
 qualified=qualification(g);qualified['baseline_reuse']=baseline_reuse(g)
 return c,g,qualified

def collect(step):
 source=source_for(step);ranks=ranks_for(step);cmd=command_for(step)
 guard=numerical(step,cmd,7200)
 out=N/step/'quick.json';x,e=quick_checked(out,source,ranks)
 assert clean_guard(guard,cmd)['command']==cmd
 finish(step,[Path(p)for p in e]+[guard],dict(source=str(source),source_sha256=sha(source),ranks=ranks))

def merge(role):
 assert role in ('base','arm')
 source=BASE_SOURCE if role=='base'else SOURCE
 rows=[];inputs={}
 for i,ranks in enumerate(HALVES,1):
  p=N/f'{role}_a{i}'/'quick.json';x,e=quick_checked(p,source,ranks)
  rows+=x['rows'];inputs.update(e)
 out=N/f'quick_{role}_ten.json';exclusive(out,dict(candidate=str(source),rows=rows))
 for rank in RANKS:
  i=1 if rank in HALVES[0]else 2
  src=N/f'{role}_a{i}'/f'quick_pred_{rank}.npy'
  dst=out.with_name(out.stem+f'_pred_{rank}.npy')
  with dst.open('xb')as f:f.write(src.read_bytes())
 _,e=quick_checked(out,source,RANKS);inputs.update(e)
 return dict(path=str(out),sha256=sha(out),evidence_sha256=inputs)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['numeric']);p.add_argument('root_grant',type=Path);a=p.parse_args()
 c,g,qualified=grant(a.root_grant)
 assert not N.exists() and not(D/'NATIVE_COMPLETE_HELD_v1.json').exists()
 lock=D/'OWNER_LOCK_v1';lock.mkdir();N.mkdir()
 exclusive(D/'NATIVE_OWNER_v1.json',dict(utc=stamp(),controller_pid=os.getpid(),controller_sha256=sha(__file__),candidate_sha256=SOURCE_SHA,baseline_sha256=BASE_SHA,root_grant_sha256=sha(a.root_grant)))
 error=None
 def interrupted(signum,frame):raise KeyboardInterrupt('Owned native signal%s'%signum)
 previous=signal.signal(signal.SIGTERM,interrupted)
 try:
  for step in STEPS:collect(step)
  merged={role:merge(role)for role in ('arm',)}
  pins();qualification(g);baseline_reuse(g);admission=admissions()
  guards=[N/s/'guard.json'for s in STEPS]
  assert sorted(N.glob('*/guard.json'))==sorted(guards)
  evidence={}
  for role in merged:evidence.update(merged[role]['evidence_sha256'])
  for step in STEPS:
   for kind in ['STARTED','OWNER','DONE']:
    p=D/f'NATIVE_STEP_{step}_{kind}_v1.json';evidence[str(p)]=sha(p)
   evidence.update(read(D/f'NATIVE_STEP_{step}_DONE_v1.json')['output_sha256'])
  exclusive(D/'NATIVE_COMPLETE_HELD_v1.json',dict(utc=stamp(),status='INDEPENDENT_RELEASE_NATIVE_TEN_COMPLETE_HELD_BEFORE_PRODUCTION_ELIGIBILITY_COMPARISON_RELEASE',controller_pid=os.getpid(),controller_sha256=sha(__file__),candidate_sha256=SOURCE_SHA,baseline_sha256=BASE_SHA,root_grant_sha256=sha(a.root_grant),protocol_sha256=sha(ACTIVE_PROTOCOL),ranks=RANKS,halves=HALVES,steps=STEPS,distinct_network_identities=10,total_native_calls=10,qualification=qualified,production_prerequisites_before_comparison=True,merged=merged,admission_evidence=admission,evidence_sha256=evidence,expected_guards=list(map(str,guards)),all_owned_guards_retired=RETIRED,comparison_performed=False,release_trigger_emitted=False,managed_reaping_and_root_retirement_required=True,no_followup_authorization=True))
 except BaseException as e:
  error=e;exclusive(D/'NATIVE_STOPPED_v1.json',dict(utc=stamp(),error=repr(e),all_owned_guards_retired=RETIRED,slot_reservation_retained=not RETIRED,no_retry=True));raise
 finally:
  exclusive(D/'NATIVE_OWNER_RETIREMENT_v1.json',dict(utc=stamp(),controller_pid=os.getpid(),error=repr(error)if error else None,all_owned_guards_retired=RETIRED,slot_reservation_retained=not RETIRED))
  if RETIRED:lock.rmdir()
  signal.signal(signal.SIGTERM,previous)

if __name__=='__main__':main()

