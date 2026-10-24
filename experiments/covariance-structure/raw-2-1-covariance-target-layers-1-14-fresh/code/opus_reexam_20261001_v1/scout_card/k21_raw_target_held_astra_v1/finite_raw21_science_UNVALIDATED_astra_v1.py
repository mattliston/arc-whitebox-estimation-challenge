"""HELD finite fresh32/OOS execution and complete-table grading; inert import.
Fresh evaluation may overlap price. The fixed MAX price must be frozen before
any fresh mean. Separate root registration and numerical grants are required.
"""
import argparse,copy,hashlib,importlib.util,json,math,os,signal,statistics,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[1]
H=D/'finite_raw21_smoke_UNVALIDATED_astra_v1.py'
assert hashlib.sha256(H.read_bytes()).hexdigest()=='8a1837c8e3c680f1dd81a238d18370767193e419e9d61248a68e414039d457bf'
spec=importlib.util.spec_from_file_location('raw21_smoke_source_pinned',H);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
sha=h.sha;write=h.write
MODES={'C':0,'RAW21':1}
def config(path,digest):
    assert sha(path)==digest
    c=json.loads(Path(path).read_text());assert c['status']=='root_registered_followup_before_evaluations'
    assert c['approved'] is True and c['reviewed_by']=='root'
    assert len(c['fresh'])==16 and len(set(c['fresh']))==16 and c['oos']==list(range(7100592,7100608))
    allocation=json.loads(Path(c['fresh_allocation_path']).read_text())
    assert sha(c['fresh_allocation_path'])==c['fresh_allocation_sha256']
    assert allocation['approved'] is True and allocation['purpose']=='raw21_target_fresh16' and allocation['seeds']==c['fresh']
    assert allocation['reviewed_by']=='root' and allocation['no_evaluated_seed_reclassified'] is True
    assert c['fresh']==list(range(7102488,7102504))
    assert [r['seed'] for r in allocation['truth_metadata']]==c['fresh']
    assert all(r['N']>=15990784 and sha(r['path'])==r['sha256']for r in allocation['truth_metadata'])
    assert sha(__file__)==c['source_pins'][str(Path(__file__).resolve())]
    h.source_pins(c)
    pc=Path(c['native_price_protocol']);assert sha(pc)==c['native_price_protocol_sha256']
    nc=json.loads(pc.read_text());assert nc['approved'] is True and nc['reviewed_by']=='root' and nc['purpose']=='raw21_fixed_price6'
    assert c['price_candidate_sha256']=={k:v['sha256']for k,v in nc['candidates'].items()}
    base=h.registered(c['base_protocol'],c['base_protocol_sha256'])
    assert c['template_request']==base['template_request'] and c['worker_path']==base['worker_path']
    assert c['adapter_path']==base['adapter_path']
    m=h.manifest(base,c['base_protocol_sha256'])
    assert not (h.home(base)/'PHASE_LOCK').exists()
    r=json.loads((h.home(base)/'PHASE_RETIRED_v1.json').read_text())
    assert r['completed'] is True and r['protocol_sha256']==c['base_protocol_sha256']
    assert h.final_gate(base,m,c['base_protocol_sha256'])['passed']
    root=json.loads((h.home(base)/'ROOT_CONTROLLER_RETIRED_v1.json').read_text())
    assert root['approved'] is True and root['controller_absent'] is True and root['managed_session_exit_code']==0
    assert root['phase_owner_sha256']==sha(h.home(base)/'PHASE_OWNER_v1.json') and root['retirement_sha256']==sha(h.home(base)/'PHASE_RETIRED_v1.json')
    assert root['protocol_sha256']==c['base_protocol_sha256']
    return c
def run_dir(c,phase):return D/c['runs'][phase]
def price(c):
    # Revalidate the exact root-accepted six-call tariff, reading no model arrays.
    p=Path(c['root_native_price_path']);assert p==D/'ROOT_NATIVE_PRICE_FREEZE_astra_v1.json'
    root=json.loads(p.read_text())
    assert root['root_verified'] is True and root['reviewed_by']=='root'
    assert root['native_protocol_sha256']==c['native_price_protocol_sha256']
    assert root['completed_evaluations']==6 and root['guards_retired'] is True
    assert root['actual_managed_exit_observed'] is True and root['actual_managed_exit_code']==0
    assert root['both_OFF_exact_all16'] is True and root['both_ON_original_route_gates_pass'] is True
    terminal=root['terminal_receipt'];assert sha(terminal['path'])==terminal['sha256']
    r=json.loads(Path(terminal['path']).read_text())
    assert r['status']=='price6_complete_all_guards_retired' and r['all_guards_retired'] is True
    order=[(seed,role)for seed in (7100136,7100137)for role in ('parent','OFF','ON')]
    assert [(e['seed'],e['role'])for e in r['evaluations']]==order
    bills={}
    for e in r['evaluations']:
        cp=Path(e['certificate']);gp=Path(e['guard']);pp=cp.with_name(cp.stem+'_pred.npz')
        assert sha(cp)==e['certificate_sha256'] and sha(gp)==e['guard_sha256'] and sha(pp)==e['prediction_npz_sha256']
        g=json.loads(gp.read_text())
        assert g['completed'] is True and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] is True and g['remaining_rows']==[]
        assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
        cert=json.loads(cp.read_text());assert cert['sha256']==c['price_candidate_sha256'][e['role']]
        rows=[x for x in cert['rows']if x['seed']==e['seed']];assert len(rows)==1
        bill=rows[0]['flops'];assert type(bill)is int and .1*2**41<=bill<=2**41
        bills[e['seed'],e['role']]=bill
    logs=[]
    for seed in (7100136,7100137):
        assert bills[seed,'parent']==bills[seed,'OFF']
        logs.append(math.log(bills[seed,'ON']/bills[seed,'parent']))
    cost=max(logs)
    assert math.isclose(cost,root['max_log_bill'],rel_tol=1e-14,abs_tol=1e-16)
    assert r['max_log_bill']==root['max_log_bill']
    assert r['price_cap_log'] is None and r['no_cost_sign_selection'] is True and r['no_displacement_selection'] is True
    return {'RAW21':cost},sha(p)
def frozen_decision(c):
    p=D/'FRESH_DECISION_v1.json';r=json.loads(p.read_text())
    assert r['status']=='root_frozen_complete_fresh_decision' and r['followup_protocol_sha256']==c['_digest']
    actual=grade(c,'fresh')
    assert r['decision']==actual
    return actual
def plan(c,phase):
    if phase=='fresh':arms=['C','RAW21']
    else:
        d=frozen_decision(c);arms=['C']+d['survivors'];assert len(arms)>1
    return [dict(arm=a,seed=s,mode=MODES[a],id=f'raw21_{phase}_{a}_r{s}') for s in c[phase] for a in arms]
def expected_request(c,phase,item):
    r=h.make_request(c,phase+'_'+item['arm'],item['seed'],item['mode']);assert r['id']==item['id'];return r
def prepare(c,phase):
    rows=plan(c,phase);run=run_dir(c,phase);run.mkdir(exist_ok=False)
    for x in ('queue','claimed','results'):(run/x).mkdir()
    (run/'STOP').write_text('Held; separate root phase grant required.\n')
    (run/'allowed_seeds.txt').write_text(','.join(map(str,c[phase])))
    import numpy as np
    truths=[]
    for seed in c[phase]:
        p=O/'analyst/fresh'/f'truth_seed{seed}.npz'
        with np.load(p,allow_pickle=False) as z:
            assert int(z['seed'])==seed and int(z['N'])>=15990784
            truths.append(dict(seed=seed,N=int(z['N']),weights_sha256=str(z['weights_sha256']),path=str(p),sha256=sha(p)))
    for i,item in enumerate(rows):
        q=run/'queue'/('%03d_'%i+item['id']+'.json');write(q,expected_request(c,phase,item));item['request_sha256']=sha(q);item['queue_name']=q.name
    write(run/'MANIFEST_v1.json',dict(phase=phase,followup_protocol_sha256=c['_digest'],requests=rows,truths=truths))
    print(json.dumps(dict(phase=phase,held=True,count=len(rows),evaluations=0)))
def manifest(c,phase):
    run=run_dir(c,phase);m=json.loads((run/'MANIFEST_v1.json').read_text())
    assert m['phase']==phase and m['followup_protocol_sha256']==c['_digest']
    assert [{k:x[k] for k in ('arm','seed','mode','id')} for x in m['requests']]==plan(c,phase)
    for t in m['truths']:assert sha(t['path'])==t['sha256']
    return m
def clean(c,phase,m,require_root=True):
    run=run_dir(c,phase);assert not (run/'PHASE_LOCK').exists()
    owner=json.loads((run/'OWNER_v1.json').read_text());ret=json.loads((run/'RETIREMENT_v1.json').read_text())
    assert owner['protocol_sha256']==c['_digest'] and ret['owner_sha256']==sha(run/'OWNER_v1.json')
    assert owner['source_pins']==c['source_pins'] and ret['exception'] is None
    assert ret['completed'] is True and ret['all_owned_groups_retired'] is True
    assert ret['errors']==[] and ret['guard_exit_codes']==[0]*owner['workers']
    guards=[run/f'guard_w{i}_v1.json' for i in range(owner['workers'])]
    assert owner['expected_guards']==list(map(str,guards))
    assert sorted(run.glob('guard_*_v1.json'))==sorted(guards)
    for i,p in enumerate(guards):
        assert sha(p)==ret['guard_sha256'][str(p)]
        g=json.loads(p.read_text())
        assert g['command']==owner['child_commands'][i] and g['completed'] is True and g['exit_code']==0 and g['error'] is None
        assert g['whole_group_retired'] is True and g['remaining_rows']==[]
        assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
    ids={x['id'] for x in m['requests']}
    assert {p.stem for p in (run/'results').glob('*.json')}==ids and not list((run/'results').glob('*.error'))
    assert not list((run/'queue').iterdir())
    for item in m['requests']:
        p=run/'results'/(item['id']+'.json');assert sha(p)==ret['result_sha256'][item['id']]
    if require_root:
        p=run/'ROOT_CONTROLLER_RETIRED_v1.json';root=json.loads(p.read_text())
        assert root['approved'] is True and root['phase']==phase and root['protocol_sha256']==c['_digest']
        assert root['retirement_sha256']==sha(run/'RETIREMENT_v1.json') and root['owner_sha256']==sha(run/'OWNER_v1.json')
        assert root['controller_absent'] is True and root['managed_session_exit_code']==0
        assert root['result_sha256']==ret['result_sha256']
    return ret
def read_result(c,phase,item):
    run=run_dir(c,phase);p=run/'results'/(item['id']+'.json')
    claims=list((run/'claimed').glob('*'+item['id']+'.w*.json'))
    assert len(claims)==1 and sha(claims[0])==item['request_sha256']
    q=json.loads(claims[0].read_text());assert q==expected_request(c,phase,item)
    r=json.loads(p.read_text());h.validate_values(r,item['seed'])
    assert r['id']==item['id'] and r['params']==q['params'] and r['cutsim']==q['cutsim']
    assert r['worker']==claims[0].name.rsplit('.',2)[1]
    assert type(r['astra_raw21_target']) is int and r['astra_raw21_target']==item['mode']
    a=r['astra_raw21_provenance'];assert a['request_sha256']==sha(claims[0])
    assert a['worker_path']==c['worker_path'] and a['worker_sha256']==c['source_pins'][c['worker_path']]
    assert a['adapter_sha256']==c['source_pins'][c['adapter_path']] and Path(a['executable']).resolve()==Path(c['model_python']).resolve()
    assert a['thread_environment']=={k:'2' for k in ('TRAINER_THREADS',)+h.THREADS}
    assert len(r['prediction'])==16 and all(len(v)==1024 and all(math.isfinite(x) for x in v) for v in r['prediction'])
    assert h.bits(r['prediction'][-1])==h.bits(r['final_prediction'])
    events=[dict(layer=l,shape=[1024,1024],actual_covariance_law=True)for l in range(1,15)]if item['mode']else []
    assert r['astra_raw21_route']==dict(mode=item['mode'],events=events,retired=True)
    return r
def grade(c,phase):
    h.source_pins(c)
    # Price check precedes manifest/complete means and any fresh result JSON read.
    costs,price_sha=price(c)
    m=manifest(c,phase);clean(c,phase,m)
    marker=D/'FRESH_UNBLIND_v1.json'
    if phase=='fresh':
        if marker.exists():assert json.loads(marker.read_text())==dict(price_sha256=price_sha,followup_protocol_sha256=c['_digest'])
        else:write(marker,dict(price_sha256=price_sha,followup_protocol_sha256=c['_digest']))
    else:assert marker.exists() and json.loads(marker.read_text())['price_sha256']==price_sha
    records={(x['seed'],x['arm']):read_result(c,phase,x) for x in m['requests']}
    arms=sorted({x['arm'] for x in m['requests']}-{'C'})
    rows={}
    for arm in arms:
        xs=[math.log(records[s,arm]['mse_layers'][-1]/records[s,'C']['mse_layers'][-1]) for s in c[phase]]
        rows[arm]=dict(n=16,mse_log=statistics.mean(xs),bill_log=costs[arm],net_log=statistics.mean(xs)+costs[arm],
                       se_log=statistics.stdev(xs)/4,paired_log_deltas=xs)
    if phase=='fresh':return dict(complete=True,rows=rows,survivors=[a for a in ('RAW21',) if rows[a]['net_log']<0],native_price_sha256=price_sha)
    fresh=frozen_decision(c);all32={}
    for arm in arms:
        xs=fresh['rows'][arm]['paired_log_deltas']+rows[arm]['paired_log_deltas']
        net=statistics.mean(xs)+costs[arm]
        all32[arm]=dict(n=32,net_log=net,passed=fresh['rows'][arm]['net_log']<0 and rows[arm]['net_log']<0 and net<0)
    passed=[a for a in ('RAW21',) if a in all32 and all32[a]['passed']]
    selected=min(passed,key=lambda a:(all32[a]['net_log'],0 if a=='RAW21' else 1)) if passed else None
    return dict(complete=True,oos=rows,all32=all32,selected_alternative=selected,native_price_sha256=price_sha,never_stack_alternatives=True)
def launch(c,phase,m,grant_path):
    g=json.loads(Path(grant_path).read_text());workers=g['numerical_slots']
    assert g['approved'] is True and g['reviewed_by']=='root' and g['purpose']=='raw21_target_'+phase
    assert g['protocol_sha256']==c['_digest'] and g['runner_sha256']==sha(__file__) and g['run_name']==c['runs'][phase]
    assert workers in (1,2,3) and all(os.environ.get(k)=='1' for k in h.OUTER_THREADS)
    assert os.getpriority(os.PRIO_PROCESS,0)==0
    # Price was frozen before any science; no scientific result is read here.
    run=run_dir(c,phase);assert (run/'STOP').exists() and not list((run/'claimed').iterdir()) and not list((run/'results').iterdir())
    for item in m['requests']:
        p=run/'queue'/item['queue_name'];assert sha(p)==item['request_sha256'] and json.loads(p.read_text())==expected_request(c,phase,item)
    admission=h.resource_gate();assert admission['heavy_python_ge1GB']+1+workers<=10
    lock=run/'PHASE_LOCK';lock.mkdir()
    env=dict(os.environ);env.update({k:'1' for k in h.OUTER_THREADS})
    for k in ('TRAINER_TRUTH_PLAN','TRAINER_TRUTH_PLAN_SHA256'):env.pop(k,None)
    childenv=h.child_environment(c,run)
    children=[['/usr/bin/env']+[k+'='+v for k,v in childenv.items()]+[c['model_python'],'-B',c['worker_path'],str(run),str(i)] for i in range(workers)]
    owner=dict(controller_pid=os.getpid(),utc=h.stamp(),protocol_sha256=c['_digest'],root_grant_sha256=sha(grant_path),workers=workers,source_pins=c['source_pins'],
               expected_guards=[str(run/f'guard_w{i}_v1.json') for i in range(workers)],child_commands=children,admission=admission)
    write(run/'OWNER_v1.json',owner);(run/'STOP').rename(run/'STOP.prelaunch_v1')
    procs=[];logs=[];error=None
    ret=dict(completed=False,all_owned_groups_retired=False,owner_sha256=sha(run/'OWNER_v1.json'))
    try:
        for i,child in enumerate(children):
            gate=h.resource_gate();assert gate['heavy_python_ge1GB']+1+(workers-i)<=10
            log=(run/f'worker_w{i}_v1.log').open('x');logs.append(log)
            cmd=[c['guard_python'],'-B',str(O/'release/guard_footprint_v3.py'),'--record',owner['expected_guards'][i],
                 '--seconds','14400','--footprint-limit','7500000000','--rss-backstop','7500000000']+child
            procs.append(subprocess.Popen(cmd,cwd=O/'trainer',env=env,stdout=log,stderr=subprocess.STDOUT))
        write(run/'STARTED_v1.json',dict(guard_pids=[p.pid for p in procs],owner_sha256=sha(run/'OWNER_v1.json')))
        ids={x['id'] for x in m['requests']};next_check=time.monotonic()+10
        while any(p.poll() is None for p in procs):
            assert not list((run/'results').glob('*.error')),'Worker error, no retry'
            assert all(p.poll() in (None,0) for p in procs),'Guard failure, no retry'
            if {p.stem for p in (run/'results').glob('*.json')}==ids:(run/'STOP').touch(exist_ok=True)
            if time.monotonic()>=next_check:h.resource_gate(running=True);next_check=time.monotonic()+10
            time.sleep(1)
        h.source_pins(c)
    except BaseException as e:
        error=e;(run/'STOP').touch(exist_ok=True)
        for p in procs:
            if p.poll() is None:
                try:p.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
        for p in procs:
            if p.poll() is None:
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
    finally:
        (run/'STOP').touch(exist_ok=True)
        guards={};retired=True
        for i,p in enumerate(procs):
            gp=Path(owner['expected_guards'][i]);ok=False
            if p.poll() is not None and gp.exists():
                gd=json.loads(gp.read_text());ok=gd.get('whole_group_retired') is True and gd.get('remaining_rows')==[]
                guards[str(gp)]=sha(gp)
            retired=retired and ok
        ret.update(all_owned_groups_retired=retired,guard_exit_codes=[p.poll() for p in procs],guard_sha256=guards,
                   errors=[p.name for p in (run/'results').glob('*.error')],exception=repr(error) if error else None,
                   result_sha256={p.stem:sha(p) for p in (run/'results').glob('*.json')},slot_reservation_retained=not retired)
        ret['completed']=error is None and len(procs)==workers and retired and ret['guard_exit_codes']==[0]*workers and not ret['errors'] and set(ret['result_sha256'])=={x['id'] for x in m['requests']}
        write(run/'RETIREMENT_v1.json',ret)
        for p,log in zip(procs,logs):
            if p.poll() is not None:log.close()
        if retired:lock.rmdir()
    if error:raise error
    assert ret['completed'];clean(c,phase,m,require_root=False)
    print(json.dumps(dict(phase=phase,complete=True,guards_retired=True,scientific_values_read=False)))
def _main_body():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','launch','grade'));p.add_argument('phase',choices=('fresh','oos'))
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--grant',type=Path)
    a=p.parse_args();c=config(a.protocol,a.protocol_sha256);c['_digest']=a.protocol_sha256
    if a.action=='prepare':
        assert a.grant, 'Root numerical grant required for request preparation'
        g=json.loads(a.grant.read_text())
        assert g['approved'] is True and g['reviewed_by']=='root' and g['purpose']=='raw21_target_'+a.phase and g['protocol_sha256']==c['_digest']
        assert g['runner_sha256']==sha(__file__) and g['run_name']==c['runs'][a.phase] and g['numerical_slots'] in (1,2,3)
        return prepare(c,a.phase)
    if a.action=='grade':print(json.dumps(grade(c,a.phase),indent=2,allow_nan=False));return
    assert a.grant;launch(c,a.phase,manifest(c,a.phase),a.grant)
def _interrupted(signum,frame):
    raise KeyboardInterrupt('Owned raw21 target controller signal%s'%signum)

def main():
    previous=signal.signal(signal.SIGTERM,_interrupted)
    try:return _main_body()
    finally:signal.signal(signal.SIGTERM,previous)

if __name__=='__main__':raise SystemExit(main())
