"""HELD single clean-boundary fresh resume; same48 queue/core, separate segment.
Only explicit prospective root boundary+retirement receipts allow original exit1.
No source edits, unplanned failure reinterpretation, automatic retry or OOS grant.
"""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=D/'finite_energy_science_UNVALIDATED_v2.py'
assert sha(H)=='e0e329440595e708203d79cd9f0e5aaf8b264cc536f12aabd2451bba2a41796c'
spec=importlib.util.spec_from_file_location('private_segment_science',H);SCI=importlib.util.module_from_spec(spec);spec.loader.exec_module(SCI)
h=SCI.h;write=h.write
PROTOCOL=D/'FRESH_BOUNDARY_PROTOCOL_DRAFT_v1.json'
read=lambda p:json.loads(Path(p).read_text())
ORIGINAL_CLEAN=SCI.clean

def configured(review_path):
    p=read(PROTOCOL);review=read(review_path)
    assert review_path.resolve().parent==D and review['reviewed_by']=='root' and review['approved'] is True
    assert review['protocol_sha256']==sha(PROTOCOL) and review['runner_sha256']==sha(__file__)
    assert review['fixed48_sources_requests_science_unchanged'] is True
    for path,digest in p['pins'].items():assert sha(path)==digest
    c=SCI.config(p['base_protocol']['path'],p['base_protocol']['sha256']);c['_digest']=p['base_protocol']['sha256']
    assert p['canonical_run']==str(SCI.run_dir(c,'fresh')) and p['segment_name']=='segment2_v1'
    return p,c

def guard_ok(path,command,digest):
    assert sha(path)==digest;g=read(path)
    assert g['command']==command and g['completed'] is True and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] is True and g['remaining_rows']==[]
    assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
    assert g['rss_backstop_bytes']==g['footprint_trigger_bytes']==7500000000
    return g

def exact_claim(c,item,run):
    found=list((run/'claimed').glob('*'+item['id']+'.w*.json'));assert len(found)==1 and sha(found[0])==item['request_sha256']
    assert read(found[0])==SCI.expected_request(c,'fresh',item)
    return found[0]

def prior_boundary(p,c,m,review_path,require_held=False):
    run=SCI.run_dir(c,'fresh');hold_path=Path(p['hold_path']);drain_path=Path(p['drain_path'])
    hold=read(hold_path);drain=read(drain_path)
    assert hold['approved'] is True and hold['reviewed_by']=='root' and hold['status']=='root_registered_intentional_boundary_before_STOP'
    assert hold['resume_review_sha256']==sha(review_path) and hold['original_protocol_sha256']==c['_digest']
    assert hold['manifest_sha256']==sha(run/'MANIFEST_v1.json') and hold['fixed48_unchanged'] is True and hold['observed_errors']==[]
    assert hold['original_controller_pid']==p['original_controller_pid'] and hold['original_session_id']==p['original_session_id']
    assert hold['original_guard_pids']==p['original_guard_pids']
    assert hold['original_controller_live_at_hold'] is True and hold['original_expected_guards_live_at_hold'] is True
    assert hold_path.stat().st_mtime_ns < (run/'RETIREMENT_v1.json').stat().st_mtime_ns
    assert drain['status']=='root_verified_intentional_clean_boundary_retirement' and drain['reviewed_by']=='root'
    assert drain['hold_sha256']==sha(hold_path) and drain['controller_reaped'] is True and drain['remaining_rows']==[]
    assert drain['controller_pid']==hold['original_controller_pid'] and drain['session_id']==hold['original_session_id']
    assert drain['controller_exit_code']==1  # Only original helper's final incomplete assertion, below.
    assert not (run/'PHASE_LOCK').exists()
    owner=read(run/'OWNER_v1.json');ret=read(run/'RETIREMENT_v1.json');started=read(run/'STARTED_v1.json')
    assert sha(run/'OWNER_v1.json')==hold['original_owner_sha256']==drain['owner_sha256']==ret['owner_sha256']==started['owner_sha256']
    assert sha(run/'RETIREMENT_v1.json')==drain['retirement_sha256']
    assert owner['protocol_sha256']==c['_digest'] and owner['source_pins']==c['source_pins']
    assert started['guard_pids']==hold['original_guard_pids'] and len(started['guard_pids'])==owner['workers']
    assert ret['completed'] is False and ret['exception'] is None and ret['errors']==[]
    assert ret['all_owned_groups_retired'] is True and ret['slot_reservation_retained'] is False and ret['guard_exit_codes']==[0]*owner['workers']
    guards=[run/f'guard_w{i}_v1.json' for i in range(owner['workers'])]
    assert owner['expected_guards']==list(map(str,guards)) and sorted(run.glob('guard_*_v1.json'))==sorted(guards)
    for i,g in enumerate(guards):guard_ok(g,owner['child_commands'][i],ret['guard_sha256'][str(g)])
    ids={x['id'] for x in m['requests']};completed=set(ret['result_sha256'])
    assert completed<ids and set(hold['observed_completed'])<=completed
    for rid,digest in hold['observed_completed'].items():assert ret['result_sha256'][rid]==digest
    for item in m['requests']:
        if item['id'] in completed:
            assert sha(run/'results'/(item['id']+'.json'))==ret['result_sha256'][item['id']]
            exact_claim(c,item,run)
    assert not list((run/'results').glob('*.error'))
    if require_held:
        assert (run/'STOP').exists() and {q.stem for q in (run/'results').glob('*.json')}==completed
        remaining=[x for x in m['requests'] if x['id'] not in completed]
        assert {q.name for q in (run/'queue').glob('*.json')}=={x['queue_name'] for x in remaining}
        for item in remaining:
            q=run/'queue'/item['queue_name'];assert sha(q)==item['request_sha256'] and read(q)==SCI.expected_request(c,'fresh',item)
        assert len(list((run/'claimed').glob('*.json')))==len(completed)
    return dict(completed=completed,hold_sha256=sha(hold_path),drain_sha256=sha(drain_path),retirement_sha256=sha(run/'RETIREMENT_v1.json'),guards=list(map(str,guards)))

def segment_home(p,c):return SCI.run_dir(c,'fresh')/'segments_v1'/p['segment_name']
def clean_fresh(p,c,m,review_path):
    prior=prior_boundary(p,c,m,review_path);run=SCI.run_dir(c,'fresh');seg=segment_home(p,c)
    assert not (seg/'SEGMENT_LOCK').exists()
    owner=read(seg/'OWNER_v1.json');ret=read(seg/'RETIREMENT_v1.json')
    assert owner['protocol_sha256']==c['_digest'] and owner['resume_protocol_sha256']==sha(PROTOCOL) and owner['review_sha256']==sha(review_path)
    assert owner['source_pins']==c['source_pins'] and owner['prior_hold_sha256']==prior['hold_sha256'] and owner['prior_drain_sha256']==prior['drain_sha256']
    assert owner['preserved_result_sha256']==read(run/'RETIREMENT_v1.json')['result_sha256']
    assert ret['owner_sha256']==sha(seg/'OWNER_v1.json') and ret['completed'] is True and ret['exception'] is None and ret['errors']==[]
    assert ret['all_owned_groups_retired'] is True and ret['slot_reservation_retained'] is False and ret['guard_exit_codes']==[0]*owner['workers']
    guards=[seg/f'guard_w{i}_v1.json' for i in range(owner['workers'])]
    assert owner['expected_guards']==list(map(str,guards)) and sorted(seg.glob('guard_*_v1.json'))==sorted(guards)
    assert {str(x) for x in (run/'segments_v1').rglob('guard_*_v1.json')}==set(map(str,guards))
    for i,g in enumerate(guards):guard_ok(g,owner['child_commands'][i],ret['guard_sha256'][str(g)])
    ids={x['id'] for x in m['requests']}
    assert set(ret['result_sha256'])==ids=={q.stem for q in (run/'results').glob('*.json')}
    assert not list((run/'queue').iterdir()) and not list((run/'results').glob('*.error'))
    assert len(list((run/'claimed').glob('*.json')))==len(ids)
    for item in m['requests']:
        assert sha(run/'results'/(item['id']+'.json'))==ret['result_sha256'][item['id']]
        exact_claim(c,item,run)
    return ret

def install(review_path):
    p,c=configured(review_path)
    SCI.clean=lambda cfg,phase,m:clean_fresh(p,c,m,review_path) if phase=='fresh' else ORIGINAL_CLEAN(cfg,phase,m)
    return p,c

def launch(p,c,m,review_path,grant_path):
    prior=prior_boundary(p,c,m,review_path,require_held=True);g=read(grant_path);workers=g['numerical_slots']
    assert grant_path.resolve().parent==D and g['approved'] is True and g['reviewed_by']=='root' and g['purpose']=='gnn_energy_fresh_boundary_resume'
    assert g['base_protocol_sha256']==c['_digest'] and g['resume_protocol_sha256']==sha(PROTOCOL) and g['runner_sha256']==sha(__file__)
    assert g['review_sha256']==sha(review_path) and g['hold_sha256']==prior['hold_sha256'] and g['drain_sha256']==prior['drain_sha256']
    assert g['segment_name']==p['segment_name'] and workers in (1,2,3)
    assert os.getpriority(os.PRIO_PROCESS,0)==0 and all(os.environ.get(k)=='1' for k in h.OUTER_THREADS)
    admission=h.resource_gate();assert admission['heavy_python_ge1GB']+1+workers<=10
    run=SCI.run_dir(c,'fresh');seg=segment_home(p,c);seg.mkdir(parents=True,exist_ok=False);lock=seg/'SEGMENT_LOCK';lock.mkdir()
    env=dict(os.environ);env.update({k:'1' for k in h.OUTER_THREADS})
    for k in ('TRAINER_TRUTH_PLAN','TRAINER_TRUTH_PLAN_SHA256'):env.pop(k,None)
    childenv=h.child_environment(c,run)
    children=[['/usr/bin/env']+[k+'='+v for k,v in childenv.items()]+[c['model_python'],'-B',c['worker_path'],str(run),str(i)] for i in range(workers)]
    owner=dict(protocol_sha256=c['_digest'],resume_protocol_sha256=sha(PROTOCOL),review_sha256=sha(review_path),root_grant_sha256=sha(grant_path),workers=workers,
               source_pins=c['source_pins'],runner_sha256=sha(__file__),prior_hold_sha256=prior['hold_sha256'],prior_drain_sha256=prior['drain_sha256'],
               preserved_result_sha256=read(run/'RETIREMENT_v1.json')['result_sha256'],expected_guards=[str(seg/f'guard_w{i}_v1.json') for i in range(workers)],child_commands=children,admission=admission,controller_pid=os.getpid())
    write(seg/'OWNER_v1.json',owner);(run/'STOP').rename(seg/'STOP.before_resume_v1')
    procs=[];logs=[];error=None;ret=dict(completed=False,all_owned_groups_retired=False,owner_sha256=sha(seg/'OWNER_v1.json'))
    try:
        for i,child in enumerate(children):
            gate=h.resource_gate();assert gate['heavy_python_ge1GB']+1+(workers-i)<=10
            log=(seg/f'worker_w{i}_v1.log').open('x');logs.append(log)
            command=[c['guard_python'],'-B',str(O/'release/guard_footprint_v3.py'),'--record',owner['expected_guards'][i],'--seconds','14400','--footprint-limit','7500000000','--rss-backstop','7500000000']+child
            procs.append(subprocess.Popen(command,cwd=O/'trainer',env=env,stdout=log,stderr=subprocess.STDOUT))
        write(seg/'STARTED_v1.json',dict(guard_pids=[x.pid for x in procs],owner_sha256=sha(seg/'OWNER_v1.json')))
        ids={x['id'] for x in m['requests']};next_check=time.monotonic()+10
        while any(x.poll() is None for x in procs):
            assert not list((run/'results').glob('*.error')) and all(x.poll() in (None,0) for x in procs)
            if {x.stem for x in (run/'results').glob('*.json')}==ids:(run/'STOP').touch(exist_ok=True)
            if time.monotonic()>=next_check:h.resource_gate(running=True);next_check=time.monotonic()+10
            time.sleep(1)
        for path,digest in p['pins'].items():assert sha(path)==digest
    except BaseException as e:
        error=e;(run/'STOP').touch(exist_ok=True)
        for x in procs:
            if x.poll() is None:
                try:x.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
        for x in procs:
            if x.poll() is None:
                try:x.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
    finally:
        (run/'STOP').touch(exist_ok=True);guards={};retired=True
        for i,x in enumerate(procs):
            gp=Path(owner['expected_guards'][i]);ok=False
            if x.poll() is not None and gp.exists():
                gd=read(gp);ok=gd.get('whole_group_retired') is True and gd.get('remaining_rows')==[];guards[str(gp)]=sha(gp)
            retired=retired and ok
        ret.update(all_owned_groups_retired=retired,guard_exit_codes=[x.poll() for x in procs],guard_sha256=guards,errors=[x.name for x in (run/'results').glob('*.error')],exception=repr(error) if error else None,result_sha256={x.stem:sha(x) for x in (run/'results').glob('*.json')},slot_reservation_retained=not retired)
        ret['completed']=error is None and len(procs)==workers and retired and ret['guard_exit_codes']==[0]*workers and not ret['errors'] and set(ret['result_sha256'])=={x['id'] for x in m['requests']}
        write(seg/'RETIREMENT_v1.json',ret)
        for x,log in zip(procs,logs):
            if x.poll() is not None:log.close()
        if retired:lock.rmdir()
    if error:raise error
    assert ret['completed'];clean_fresh(p,c,m,review_path)
    print(json.dumps(dict(complete=True,fixed_fresh48=True,guards_retired=True,scientific_values_read=False)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('launch','grade','prepare-oos','launch-oos'));ap.add_argument('--review',type=Path,required=True);ap.add_argument('--grant',type=Path);ap.add_argument('--phase',choices=('fresh','oos'),default='fresh');args=ap.parse_args()
    p,c=install(args.review)
    if args.action=='grade':print(json.dumps(SCI.grade(c,args.phase),indent=2,allow_nan=False));return
    if args.action=='prepare-oos':return SCI.prepare(c,'oos')
    assert args.grant
    if args.action=='launch-oos':return SCI.launch(c,'oos',SCI.manifest(c,'oos'),args.grant)
    launch(p,c,SCI.manifest(c,'fresh'),args.review,args.grant)
if __name__=='__main__':main()
