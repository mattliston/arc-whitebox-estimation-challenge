"""HELD fixed final-source normal/slow boundary only. Root own20 acceptance required.
No candidate rewrite, native10, seal, release confirmation, retry or submission.
"""
import argparse,ast,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2];P2=O.parent;R=O/'release'
J=O/'lead/runs/astra_gtr_iop_joint_v1';C=O/'lead/runs/astra_gop2_current_own20_v1';P=O/'lead/runs/astra_gtr_iop_production_recipe_v1'
Q=O/'lead/runs/astra_gop2_current_delivery_v1'
PROV=Q/'candidate_current4e_GOP2_ON_HELD_v1.py';PARENT=O/'lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py'
PY=Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
N=D/'numerical_parallel_v2';ACTIVE_PROTOCOL=D/'BOUNDARY_PROTOCOL_HELD_v3.json'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');ENV.update({k:'1'for k in THREADS})
for flag in('PB_ALL','MODE','DRY','STATS'):ENV.pop(flag,None)
RETIRED=True

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
    p = D / f'BOUNDARY_STEP_{step}_DONE_v2.json'
    if not p.exists():
        return False
    r = read(p)
    for path, expected in r['output_sha256'].items():
        assert sha(path) == expected, ('completed output changed', path)
    return True


def finish(step, paths, details):
    exclusive(D / f'BOUNDARY_STEP_{step}_DONE_v2.json', dict(utc=stamp(), details=details,
              output_sha256={str(p): sha(p) for p in paths}))


def admission_event(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')

def admit(step):
    # Keep the granted slot reserved; no native child exists during this wait.
    assert RETIRED is True and step in ['final_normal','final_slow']
    started=time.monotonic();attempt=0;last_status=None
    while True:
        state=resources();elapsed=time.monotonic()-started;attempt+=1
        okay=state['memory_free_percent']>=60 and state['counted']<10 and elapsed<=1800
        record=dict(step=step,controller_pid=os.getpid(),attempt=attempt,
                    elapsed_seconds=elapsed,admitted=okay,**state)
        admission_event(N/step/'admission_wait.jsonl',record)
        if okay:
            admission_event(N/step/'admission.jsonl',record)
            return record
        if last_status is None or elapsed-last_status>=30:
            print(json.dumps(dict(status='waiting_for_prospective_headroom_no_native_child',
                  step=step,memory_free_percent=state['memory_free_percent'],counted=state['counted'],
                  elapsed_seconds=elapsed,slot_reserved=True)),flush=True)
            last_status=elapsed
        assert elapsed<1800, 'Bounded admission wait expired; no automatic retry'
        time.sleep(min(15,1800-elapsed))

def admissions(step,controller_pid):
    records=[json.loads(x)for x in (N/step/'admission.jsonl').read_text().splitlines()]
    assert [x['step']for x in records]==[step]
    assert all(x['controller_pid']==controller_pid and x['memory_free_percent']>=60 and x['counted']<10 for x in records)
    attempts=[json.loads(x)for x in (N/step/'admission_wait.jsonl').read_text().splitlines()]
    assert [x for x in attempts if x['admitted']]==records
    assert all(x['admitted']is True and 0<=x['elapsed_seconds']<=1800 for x in records)
    return dict(path=str(N/step/'admission.jsonl'),sha256=sha(N/step/'admission.jsonl'),records=records,
                wait_path=str(N/step/'admission_wait.jsonl'),wait_sha256=sha(N/step/'admission_wait.jsonl'))

def numerical(step, command, seconds, extra_env=None):
    global RETIRED
    pins()
    assert os.getpriority(os.PRIO_PROCESS, 0) == 0 and all(os.environ.get(k) == '1' for k in THREADS)
    assert step in ('final_normal','final_slow')
    output=N/step
    assert not output.exists()
    output.mkdir()
    guard, log = output/'guard.json', output/'run.log'
    started, owner = D / f'BOUNDARY_STEP_{step}_STARTED_v2.json', D / f'BOUNDARY_STEP_{step}_OWNER_v2.json'
    assert not any(p.exists() for p in [guard, log, started, owner]), 'unfinished step needs root review/new version'
    before=admit(step)
    pins()
    env = dict(ENV, **(extra_env or {}))
    args = [str(PY), '-B', str(R / 'guard_footprint_v3.py'), '--record', str(guard),
            '--seconds', str(seconds), '--footprint-limit', '7500000000', '--rss-backstop', '7500000000', *command]
    exclusive(started, dict(utc=stamp(), command=args, resources=before, source_sha256=sha(PROV),
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
                    with (D / 'boundary_resources_v1.jsonl').open('a') as f:
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
                exclusive(D / f'BOUNDARY_STEP_{step}_INTERRUPTED_v2.json', dict(utc=stamp(), error=repr(failure),
                          owned_guard_pid=child.pid if child else None, owned_retirement_verified=RETIRED,
                          slot_reservation_retained=not RETIRED))
    if failure is not None:
        raise failure
    assert child.returncode == 0 and RETIRED
    clean_guard(guard, command)
    pins()
    return guard


def pb(path):
    found=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_PB_SINGLE'for t in n.targets)]
    return ast.literal_eval(found[-1].value)

def pins():
    c=read(ACTIVE_PROTOCOL)
    assert c['purpose']=='GOP2_current4e_final_boundary_normal_slow'
    assert c['fixed_modes']==['normal','slow']and c['ranks']==[35,100]
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    for p,h in c['source_pins'].items():assert sha(p)==h,('changed pin',p)
    assert sha(PROV)=='0170697df8866cff2bc43b13fa7a41faf5a8d73a9c527985e12ac8a29a4c89e4'
    assert sha(PARENT)=='4e1bf080bb6751b487c62842e369a427633e0c4686f089fd3f8e6448bb416f0e'
    assert PROV.read_bytes().startswith(PARENT.read_bytes())and pb(PROV)==pb(PARENT)and len(pb(PROV))==649
    return c

def prior_census(g):
    assert g['own20_managed_exit_and_whole_group_retirement_approved']is True
    rp=Path(g['own20_root_retirement_path']).resolve();assert rp.parent==C and sha(rp)==g['own20_root_retirement_sha256']
    root=read(rp);assert root['approved']is True and root['reviewed_by']=='root'
    ap=C/'ASTRA_CENSUS_OWNER_RETIREMENT_v1.json';assert sha(ap)==g['own20_owner_retirement_sha256']
    a=read(ap);cp=C/'CENSUS_COMPLETE_HELD_v1.json'
    assert root['owner_retirement_sha256']==sha(ap)and root['complete_sha256']==sha(cp)
    assert root['actual_managed_exit_observed']is True and root['exit_code']==0 and root['all_owned_PID_PPID_PGID_absent']is True
    assert a['status']=='COMBINED_OWN20_OWNER_REAPED_PB_RETAINED_HELD'
    assert type(g['own20_managed_session_id'])is int and g['own20_managed_session_id']>0
    assert a['managed_session']==g['own20_managed_session_id']and a['managed_session_exit_code']==0
    assert a['all20_new_processes_no_partial_reuse']is True and a['all20_complete_all16_finite_actual_keys']is True
    assert a['controller_sha256']==sha(C/'finite_gop2_current_census_HELD_v1.py')
    gp=Path(g['own20_grant_path']).resolve();assert gp.parent==C and sha(gp)==g['own20_grant_sha256']==a['root_grant_sha256']
    assert a['candidate_sha256']==sha(PROV)and a['complete_sha256']==sha(cp)
    assert a['remaining_owned_rows']==[]and len(a['owned_pids_checked_absent'])==5 and a['all_owned_PID_PPID_PGID_absent']is True
    assert a['MSE_fields_inspected']is False and a['historical_predictions_keys_bills_timings_reused']is False and a['old_physics_prediction_equality_gate']is False
    for p,h in a['evidence_sha256'].items():assert sha(p)==h,('changed own20 evidence',p)
    assert a['actual_coverage']['complete_networks']==20 and a['actual_coverage']['PB_order_preserved']is True and a['actual_coverage']['PB_single_count']==649
    assert not(C/'OWNER_LOCK_v1').exists()
    rows=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()]
    ps=set(a['owned_pids_checked_absent']);pg=set(a['owned_pgids_checked_absent'])
    assert not[r for r in rows if r[0]in ps or r[1]in ps or r[2]in pg]
    return dict(root_receipt=str(rp),root_sha256=sha(rp),owner_receipt_sha256=sha(ap),complete_sha256=a['complete_sha256'])

def grant(path):
    global ACTIVE_PROTOCOL
    g=read(path);assert Path(path).resolve().parent==D
    assert g['approved']is True and g['reviewed_by']=='root'and g['purpose']=='GOP2_current4e_final_boundary_normal_slow'
    ACTIVE_PROTOCOL=Path(g['protocol_path']);assert ACTIVE_PROTOCOL.resolve().parent==D
    assert g['protocol_sha256']==sha(ACTIVE_PROTOCOL)and g['controller_sha256']==sha(__file__)
    c=pins();assert c['status']=='REGISTERED'
    assert g['numerical_slots']==2 and g['fixed_modes']==['normal','slow']and g['ranks']==[35,100]
    assert g['candidate_sha256']==sha(PROV)and g['no_followup_authorization']is True
    identityp=Path(g['final_source_identity_root_path']).resolve()
    assert identityp.parent==D and sha(identityp)==g['final_source_identity_root_sha256']
    identity=read(identityp)
    assert identity['approved']is True and identity['reviewed_by']=='root'
    assert identity['candidate_sha256']==sha(PROV)and identity['same_source_as_qualified_ON_136137']is True
    assert identity['PB_order_unchanged']is True and identity['PB_single_count']==649
    assert identity['own20_root_retirement_sha256']==g['own20_root_retirement_sha256']
    qp=Path(identity['qualification_protocol_path']).resolve();ap=Path(identity['qualification_root_acceptance_path']).resolve()
    assert qp.parent==Q and ap.parent==Q and sha(qp)==identity['qualification_protocol_sha256']and sha(ap)==identity['qualification_root_acceptance_sha256']
    accepted=read(ap)
    assert accepted['approved']is True and accepted['reviewed_by']=='root'
    assert accepted['candidate_sha256']==sha(PROV)and accepted['protocol_sha256']==sha(qp)
    assert accepted['actual_managed_exit_observed']is True and accepted['exit_code']==0 and accepted['all_owned_PID_PPID_PGID_absent']is True
    assert accepted['all_fixed_qualification_checks_recomputed']is True and accepted['complete_calls']==2
    assert accepted['all16_finite']is True and accepted['first_layer_exact']is True and accepted['fifteen_routes_and_cleanup']is True and accepted['setup_identical']is True
    assert type(g['logical_reserved_after_launch'])is int and 2<=g['logical_reserved_after_launch']<=10
    assert all(os.environ.get(k)=='1'for k in THREADS)and all(os.environ.get(k)is None for k in('PB_ALL','MODE','DRY','STATS'))
    assert os.getpriority(os.PRIO_PROCESS,0)==0
    return c,g,prior_census(g)

def boundary(slow=False):
    step='final_slow'if slow else 'final_normal';out=N/step
    assert not done(step)
    args=[str(PY),'-B',str(R/'boundary_run.py'),str(PROV),'35,100',str(out)]
    args+=['--setup-slowdown','1.3']if slow else ['--short-setup']
    guard=numerical(step,args,7200)
    m,l=read(out/'manifest.json'),read(out/'launcher.json')
    rows=[json.loads(s)for s in(out/'rows.jsonl').read_text().splitlines()if s.strip()]
    assert [r['rank']for r in rows]==[35,100]
    assert m['sha256']==l['sha256']==sha(PROV)and m['ranks']==[35,100]
    assert m['short_setup']is True and m['setup_slowdown']==(1.3 if slow else 1.0)
    assert m['sites']is False and m['profiled']is False
    assert not m['setup']['failed']and l['client_exit']==l['server_exit']==0
    pins()
    return out

def assess(slow=False):
    step='final_slow'if slow else 'final_normal';out=N/step
    assert not done(step)
    guard=out/'guard.json';clean_guard(guard)
    assessment=out/'assessment_v1'

    command=[str(PY),'-B',str(D/'early_count_gate_v1.py'),str(PROV),str(out),'slow'if slow else 'normal',str(assessment)]
    if slow:command.append(str(N/'final_normal/assessment_v1'))
    with(out/'early_gate.log').open('x')as f:
        subprocess.run(command,env=ENV,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    assert read(assessment/'early_gate.json')['passed']is True
    finish(step,[guard,out/'manifest.json',out/'launcher.json',out/'rows.jsonl',assessment/'early_gate.json',assessment/'stage_c.json',assessment/'judge.txt'],dict(candidate=str(PROV),sha256=sha(PROV),ranks=[35,100],slow=slow))
    return out

def collect(mode,grant_path):
    c,g,census=grant(grant_path)
    step='final_'+mode;assert mode in ('normal','slow')
    N.mkdir(exist_ok=True)
    lock=D/f'OWNER_LOCK_{mode}_v2';lock.mkdir()
    owner=D/f'BOUNDARY_COLLECTOR_{mode}_OWNER_v2.json'
    exclusive(owner,dict(utc=stamp(),mode=mode,controller_pid=os.getpid(),controller_sha256=sha(__file__),
              protocol_sha256=sha(ACTIVE_PROTOCOL),candidate_sha256=sha(PROV),root_grant_sha256=sha(grant_path)))
    error=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Owned boundary signal%s'%signum)
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        out=boundary(mode=='slow')
        ae=admissions(step,os.getpid());pins();prior_census(g)
        paths=[out/'guard.json',out/'manifest.json',out/'launcher.json',out/'rows.jsonl',
               Path(ae['path']),Path(ae['wait_path']),owner,
               D/f'BOUNDARY_STEP_{step}_STARTED_v2.json',D/f'BOUNDARY_STEP_{step}_OWNER_v2.json']
        exclusive(D/f'BOUNDARY_COLLECTED_{mode}_v2.json',dict(status='NUMERICAL_COLLECTION_COMPLETE_GATES_PENDING',
                  utc=stamp(),mode=mode,controller_pid=os.getpid(),controller_sha256=sha(__file__),
                  candidate_sha256=sha(PROV),protocol_sha256=sha(ACTIVE_PROTOCOL),root_grant_sha256=sha(grant_path),
                  census=census,admission_evidence=ae,output_sha256={str(p):sha(p)for p in paths},
                  all_owned_guards_retired=RETIRED,production_pass_not_yet_decided=True))
    except BaseException as e:
        error=e;exclusive(D/f'BOUNDARY_COLLECTOR_{mode}_STOPPED_v2.json',dict(utc=stamp(),error=repr(e),
                  all_owned_guards_retired=RETIRED,slot_reservation_retained=not RETIRED,no_retry=True));raise
    finally:
        exclusive(D/f'BOUNDARY_COLLECTOR_{mode}_RETIREMENT_v2.json',dict(utc=stamp(),controller_pid=os.getpid(),
                  error=repr(error)if error else None,all_owned_guards_retired=RETIRED,slot_reservation_retained=not RETIRED))
        if RETIRED:lock.rmdir()
        signal.signal(signal.SIGTERM,previous)

def retired_collectors(path,grant_path):
    r=read(path);assert r['approved']is True and r['reviewed_by']=='root'
    assert r['status']=='GOP2_PARALLEL_BOUNDARY_COLLECTORS_REAPED'
    assert r['candidate_sha256']==sha(PROV)and r['controller_sha256']==sha(__file__)
    assert r['protocol_sha256']==sha(ACTIVE_PROTOCOL)and r['root_grant_sha256']==sha(grant_path)
    assert r['both_actual_managed_exits_observed']is True and r['exit_codes']==[0,0]
    assert r['all_owned_PID_PPID_PGID_absent']is True and [x['mode']for x in r['collectors']]==['normal','slow']
    assert len({x['controller_pid']for x in r['collectors']})==len({x['managed_session_id']for x in r['collectors']})==2
    for mode,owner in zip(('normal','slow'),r['collectors']):
        p=D/f'BOUNDARY_COLLECTED_{mode}_v2.json';x=read(p)
        assert owner['collected_path']==str(p)and owner['collected_sha256']==sha(p)
        assert owner['actual_managed_exit_observed']is True and owner['exit_code']==0
        assert owner['controller_pid']==x['controller_pid']and x['all_owned_guards_retired']is True
        assert x['root_grant_sha256']==sha(grant_path)and x['candidate_sha256']==sha(PROV)
        for f,h in x['output_sha256'].items():assert sha(f)==h
        assert not(D/f'OWNER_LOCK_{mode}_v2').exists()
        clean_guard(N/f'final_{mode}/guard.json')
    for f,h in r['evidence_sha256'].items():assert sha(f)==h
    ids=set(r['owned_pids_checked_absent']);groups=set(r['owned_pgids_checked_absent'])
    rows=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()]
    assert not[x for x in rows if x[0]in ids or x[1]in ids or x[2]in groups]
    return r

def finalize(grant_path,retirement_path):
    c,g,census=grant(grant_path);root=retired_collectors(retirement_path,grant_path)
    assert not(D/'BOUNDARY_COMPLETE_HELD_v2.json').exists()
    try:
        normal=assess(False);slow=assess(True)
        check=subprocess.run([str(PY),'-B',str(R/'fastgate_pin_v1.py'),'--check',str(R/'fastgate_pin_v1.json'),str(PROV),str(normal),str(slow)],env=ENV,cwd=R,capture_output=True,text=True,check=True,timeout=120)
        assert check.stdout.startswith('PIN OK:')
        with(D/'FASTGATE_PIN_CHECK_v2.txt').open('x')as f:f.write(check.stdout)
        pins();prior_census(g)
        expected=[N/step/'guard.json'for step in('final_normal','final_slow')]
        assert sorted(N.glob('*/guard.json'))==sorted(expected)
        receipts={str(D/f'BOUNDARY_STEP_{s}_DONE_v2.json'):sha(D/f'BOUNDARY_STEP_{s}_DONE_v2.json')for s in('final_normal','final_slow')}
        evidence={str(retirement_path):sha(retirement_path),str(D/'FASTGATE_PIN_CHECK_v2.txt'):sha(D/'FASTGATE_PIN_CHECK_v2.txt'),**receipts}
        for mode in('normal','slow'):
            p=D/f'BOUNDARY_COLLECTED_{mode}_v2.json';evidence[str(p)]=sha(p);evidence.update(read(p)['output_sha256'])
            evidence.update(read(D/f'BOUNDARY_STEP_final_{mode}_DONE_v2.json')['output_sha256'])
        exclusive(D/'BOUNDARY_COMPLETE_HELD_v2.json',dict(utc=stamp(),status='NORMAL_SLOW_COMPLETE_HELD_BEFORE_NATIVE10_SEAL_RELEASE',
                  controller_pids=[x['controller_pid']for x in root['collectors']],managed_sessions=[x['managed_session_id']for x in root['collectors']],
                  candidate_sha256=sha(PROV),controller_sha256=sha(__file__),protocol_sha256=sha(ACTIVE_PROTOCOL),
                  root_grant_sha256=sha(grant_path),census=census,final_source_identity_root_sha256=g['final_source_identity_root_sha256'],
                  done_receipts=receipts,evidence_sha256=evidence,normal_pass=True,slow_pass=True,own20_accepted=True,
                  both_actual_managed_exits_observed=True,all_owned_guards_retired=True,no_followup_authorization=True))
    except BaseException as e:
        exclusive(D/'BOUNDARY_FINALIZE_STOPPED_v2.json',dict(utc=stamp(),error=repr(e),no_model_retry=True));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','finalize']);p.add_argument('root_grant',type=Path)
    p.add_argument('--mode',choices=['normal','slow']);p.add_argument('--root-collectors-retirement',type=Path);a=p.parse_args()
    if a.action=='collect':assert a.mode;collect(a.mode,a.root_grant)
    else:assert a.root_collectors_retirement;finalize(a.root_grant,a.root_collectors_retirement)

if __name__=='__main__':main()


