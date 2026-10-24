"""HELD finite three-smoke source. No action at import.

prepare requires exact registered protocol. launch additionally requires a
root one-slot grant. No fresh/OOS scientific data are opened by this helper.
Three jobs: exact Czero584, exact full384zero624, complement624. Each has its own finite queue,
one worker, one guard and retirement receipt. No retries or phase continuation.
"""
import argparse,copy,hashlib,json,math,os,re,signal,struct,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent
O=D.parents[1]
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
OUTER_THREADS=THREADS+('VECLIB_NUM_THREADS',)
ORDER=(('zero584',7100584,'control'),('zero384_624',7100624,'rank384_zero'),('recon624',7100624,'recon_prefix'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def bits(x):return tuple(bits(v) for v in x) if isinstance(x,list) else struct.pack('>d',float(x))
def source_pins(cfg):
    for p,h in cfg['source_pins'].items():assert sha(p)==h,p
def registered(path,digest):
    path=Path(path).resolve();assert path.parent==D and sha(path)==digest
    r=json.loads(path.read_text())
    assert r['status']=='root_registered_before_own_evaluations'
    assert r['approved'] is True and r['reviewed_by']=='root' and r['reference_reuse_approved'] is True
    assert r['smoke_order']==[list(x) for x in ORDER] and r['smoke_slots']==1
    allocation=Path(r['fresh_allocation_path']);assert allocation.parent==D and sha(allocation)==r['fresh_allocation_sha256']
    alloc=json.loads(allocation.read_text())
    assert alloc['approved'] is True and alloc['purpose']=='recon_prefix_fresh16'
    assert alloc['seeds']==list(range(7102440,7102456))==r['fresh'] and alloc['no_evaluated_seed_reclassified'] is True and alloc['minimum_N']==15990784
    assert r['native_price_root_frozen'] is True
    price=Path(r['native_price_path']);assert sha(price)==r['native_price_sha256']
    frozen=json.loads(price.read_text());assert frozen['approved'] is True and frozen['all8_clean_retired'] is True
    for p,h in r['source_pins'].items():assert sha(p)==h,p
    assert r['helper_path']==str(Path(__file__).resolve()) and sha(__file__)==r['source_pins'][str(Path(__file__).resolve())]
    for item in r['smoke_references'].values():
        for key in ('request','result'):assert sha(item[key+'_path'])==item[key+'_sha256']
    return r
def core(r):
    # Only the exact cutsim path/hash descriptor varies; every numeric option,
    # learned parameter/head/dial and runtime request field remains part of core.
    c=copy.deepcopy(r)
    for k in ('id','rank','keep_final','keep_all'):c.pop(k,None)
    c['cutsim']={'opts':copy.deepcopy(r['cutsim']['opts'])}
    return c
def make_request(cfg,role,seed,mode):
    assert (role,seed,mode) in ORDER
    p=Path(cfg['template_request']);assert sha(p)==cfg['source_pins'][str(p)]
    r=json.loads(p.read_text())
    assert r['mode']=='eval' and r['cutsim']['opts']=={'enabled':True,'law_response_fit':True,'law_beta':True,'law_restore_right':True,'ru':256,'law_fit_blend':0.0}
    folder=Path(cfg['private_torch_path'])/('cutsim_'+mode)
    names=('torch_cutsim_s46_v11.py','n6y_cutsim_v2.py','torch_rowcuts_v15.py','torch_push_native_v2.py','pbase_cutsim_v1.py')
    sources={n:cfg['source_pins'][str(folder/n)] for n in names}
    r.update(id='rps_'+role+'_r'+str(seed),rank=seed,keep_final=True,keep_all=True)
    r['cutsim']=dict(module=str(folder/names[0]),sha256=sources[names[0]],sources=sources,opts=r['cutsim']['opts'])
    return r
def home(cfg):return D/cfg['smoke_run_name']
def prepare(cfg,protocol,digest):
    run=home(cfg);run.mkdir(exist_ok=False)
    ref=json.loads(Path(cfg['template_request']).read_text())
    for item in cfg['smoke_references'].values():assert core(json.loads(Path(item['request_path']).read_text()))==core(ref),'Reference request C drift'
    import numpy as np
    truths={}
    for seed in (7100584,7100624):
        p=O/'analyst/fresh'/('truth_seed'+str(seed)+'.npz')
        with np.load(p,allow_pickle=False) as z:
            assert int(z['seed'])==seed and int(z['N'])>=15990784
            truths[str(seed)]=dict(path=str(p),sha256=sha(p),N=int(z['N']),weights_sha256=str(z['weights_sha256']))
    jobs=[]
    for role,seed,mode in ORDER:
        job=run/role
        for name in ('queue','claimed','results'):(job/name).mkdir(parents=True,exist_ok=False)
        (job/'STOP').write_text('HELD; numerical root slot grant required.\n')
        (job/'allowed_seeds.txt').write_text(str(seed))
        req=make_request(cfg,role,seed,mode);p=job/'queue'/(req['id']+'.json');write(p,req)
        jobs.append(dict(role=role,seed=seed,mode=mode,id=req['id'],request_sha256=sha(p)))
    write(run/'requests_manifest_v1.json',dict(protocol_path=str(protocol),protocol_sha256=digest,jobs=jobs,truths=truths))
    print(json.dumps(dict(prepared=True,held=True,jobs=3,evaluations=0)))
def manifest(cfg,digest):
    run=home(cfg);p=run/'requests_manifest_v1.json';m=json.loads(p.read_text())
    assert m['protocol_sha256']==digest
    assert [(x['role'],x['seed'],x['mode']) for x in m['jobs']]==list(ORDER)
    for v in m['truths'].values():assert sha(v['path'])==v['sha256']
    return m
def clean_job(cfg,item,digest):
    job=home(cfg)/item['role']
    assert not (job/'LAUNCH_LOCK').exists()
    op=job/'owner_v1.json';rp=job/'retirement_v1.json';gp=job/'guard_v1.json'
    owner=json.loads(op.read_text());ret=json.loads(rp.read_text());g=json.loads(gp.read_text())
    assert owner['protocol_sha256']==digest and owner['role']==item['role']
    assert owner['source_pins']==cfg['source_pins']
    assert owner['expected_guard']==str(gp) and ret['owner_sha256']==sha(op)
    assert ret['guard_path']==str(gp) and ret['guard_sha256']==sha(gp)
    assert ret['completed'] is True and ret['guard_exit_code']==0 and ret['owner_process_reaped'] is True
    assert g['completed'] is True and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] is True and g['remaining_rows']==[]
    assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
    assert g['rss_backstop_bytes']==7500000000 and g['footprint_trigger_bytes']==7500000000
    assert g['command']==owner['guard_child_command']
    assert {p.name for p in job.glob('guard*.json')}=={'guard_v1.json'}
    assert not list((job/'results').glob('*.error'))
    assert {p.name for p in (job/'results').iterdir()}=={item['id']+'.json'}
    assert not list((job/'queue').iterdir())
    return dict(owner_sha256=sha(op),retirement_sha256=sha(rp),guard_sha256=sha(gp))
def validate_values(r,seed):
    assert r['rank']==seed and r['mode']=='eval' and r['threads']==2 and r['data_source']=='generated'
    assert r['card']=='M' and r['f4c'] is True and r.get('truth_plan') is None
    assert len(r['mse_layers'])==16 and all(math.isfinite(x) and x>0 for x in r['mse_layers'])
    assert math.isfinite(r['mse']) and r['mse']>0 and math.isclose(r['mse'],r['mse_layers'][-1],rel_tol=1e-12,abs_tol=0)
    assert len(r['final_prediction'])==1024 and all(math.isfinite(x) for x in r['final_prediction'])
def read_job(cfg,item,digest):
    source_pins(cfg)
    clean=clean_job(cfg,item,digest)
    job=home(cfg)/item['role'];rid=item['id']
    claims=list((job/'claimed').glob('*.json'))
    assert len(claims)==1 and claims[0].name==rid+'.w0.json'
    q=json.loads(claims[0].read_text());assert q==make_request(cfg,item['role'],item['seed'],item['mode'])
    assert sha(claims[0])==item['request_sha256']
    p=job/'results'/(rid+'.json')
    ret=json.loads((job/'retirement_v1.json').read_text())
    assert ret['result_sha256']==sha(p) and ret['request_sha256']==sha(claims[0])
    r=json.loads(p.read_text());validate_values(r,item['seed'])
    assert r['id']==rid and r['worker']=='w0' and r['params']==q['params'] and r['cutsim']==q['cutsim']
    a=r['astra_recon_provenance']
    assert a['request_sha256']==sha(claims[0]) and a['worker_path']==cfg['worker_path']
    assert a['worker_sha256']==cfg['source_pins'][cfg['worker_path']] and a['runtime_path']==cfg['runtime_path'] and a['runtime_sha256']==cfg['source_pins'][cfg['runtime_path']]
    assert Path(a['executable']).resolve()==Path(cfg['model_python']).resolve()
    assert a['thread_environment']=={k:'2' for k in ('TRAINER_THREADS',)+THREADS}
    assert len(r['prediction'])==16 and all(len(v)==1024 and all(math.isfinite(x) for x in v) for v in r['prediction'])
    assert bits(r['final_prediction'])==bits(r['prediction'][-1])
    return r,dict(path=str(p),sha256=sha(p),claim_sha256=sha(claims[0]),**clean)
def read_reference(cfg,name,seed):
    item=cfg['smoke_references'][name]
    source_pins(cfg)
    assert sha(item['request_path'])==item['request_sha256'] and sha(item['result_path'])==item['result_sha256']
    # Exact completed source-specific reference custody is pinned before any run.
    ret=json.loads(Path(item['retirement_path']).read_text())
    assert ret['passed'] is True and ret['managed_exit_code']==0 and ret['guard_controller_child_absent'] is True
    assert ret['exact_expected_guard_set_retired'] is True
    g=json.loads(Path(item['guard_path']).read_text())
    assert g['completed'] is True and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] is True and g['remaining_rows']==[]
    assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
    q=json.loads(Path(item['request_path']).read_text());r=json.loads(Path(item['result_path']).read_text())
    assert core(q)==core(json.loads(Path(cfg['template_request']).read_text()))
    assert q['cutsim']==item['cutsim_descriptor']
    validate_values(r,seed)
    assert q['rank']==seed and r['id']==q['id'] and r['params']==q['params'] and r['cutsim']==q['cutsim']
    assert q['keep_all'] is True and type(q['keep_final']) in (bool,int) and q['keep_final']==1
    assert len(r['prediction'])==16 and all(len(v)==1024 and all(math.isfinite(x) for x in v) for v in r['prediction'])
    assert bits(r['prediction'][-1])==bits(r['final_prediction'])
    assert r['readrank_stats']=={'calls':0,'events':[]}
    return r
def exact(a,b):return {k:bits(a[k])==bits(b[k]) for k in ('mse','mse_layers','final_prediction','prediction')}
def zero_gate(cfg,m,digest):
    r,meta=read_job(cfg,m['jobs'][0],digest)
    ref=read_reference(cfg,'zero584',7100584)
    checks=exact(r,ref)
    return dict(passed=all(checks.values()),checks=checks,evidence=meta)
def final_gate(cfg,m,digest):
    z=zero_gate(cfg,m,digest);assert z['passed']
    rows=[read_job(cfg,x,digest) for x in m['jobs'][1:]]
    c=rows[0][0];ref=read_reference(cfg,'zero384_624',7100624)
    checks=exact(c,ref)
    r=rows[1][0]
    checks['layer0_to9_exact']=all(bits(r['prediction'][j])==bits(c['prediction'][j]) and bits(r['mse_layers'][j])==bits(c['mse_layers'][j]) for j in range(10))
    checks['final_changed']=bits(r['final_prediction'])!=bits(c['final_prediction'])
    allrows=[read_job(cfg,m['jobs'][0],digest)[0],c,r]
    checks['no_shortened_reads']=all(x['readrank_stats']=={'calls':0,'events':[]} for x in allrows)
    checks['zero_recon_counters']=all(x['recon_prefix_stats']=={'calls':0,'events':[]} for x in allrows[:2])
    stats=r['recon_prefix_stats'];events=stats['events']
    checks['four_prefix_events']=stats['calls']==4 and len(events)==4 and [x['layer'] for x in events]==[10,11,12,13]
    checks['event_dimensions']=all(x['full_state_rank']==384 and x['reconstruction_v_rank']==320 and x['common_rank']==256 and x['rows']>0 and x['prefix_pair_columns']>0 for x in events)
    return dict(passed=all(checks.values()),zero=z,checks=checks,evidence=[x[1] for x in rows],smoke_MSE_not_used_for_selection=True)
def resource_gate(running=False):
    p=subprocess.check_output(['/usr/bin/memory_pressure'],text=True,timeout=15)
    free=int(re.search(r'System-wide memory free percentage: (\d+)%',p).group(1))
    rows=subprocess.check_output(['/bin/ps','-axo','rss=,comm='],text=True,timeout=10)
    heavy=sum(int(s.split(None,1)[0])>=1000000 and 'python' in s.lower() for s in rows.splitlines() if s.strip())
    assert free>=50 and heavy+1+(0 if running else 1)<=10,('resource_gate',free,heavy)
    return dict(free_percent=free,heavy_python_ge1GB=heavy,truth_reserve=1,new_slots=(0 if running else 1))
def child_environment(cfg,job):
    e=dict(TRAINER_CARD='M',TRAINER_PL='1',TRAINER_F4C='1',TRAINER_THREADS='2',TRAINER_RELIEF='1',
           TRAINER_STAR_ATTACH='0',TRAINER_BIRTH_ATTACH='0',TRAINER_RSS_LIMIT='7500000000',
           TRAINER_GENERATED_MIN_N='15990784',TRAINER_GENERATED_ALLOW_FILE=str(job/'allowed_seeds.txt'),
           PYTHONPATH=str(O/'trainer'),PYTHONDONTWRITEBYTECODE='1')
    e.update({k:'2' for k in THREADS});return e
def run_one(cfg,item,digest,grant_sha):
    source_pins(cfg)
    job=home(cfg)/item['role'];rid=item['id']
    assert (job/'STOP').exists() and not list((job/'claimed').iterdir()) and not list((job/'results').iterdir())
    q=job/'queue'/(rid+'.json');assert sha(q)==item['request_sha256']
    assert json.loads(q.read_text())==make_request(cfg,item['role'],item['seed'],item['mode']),'Launch request core drift'
    gate=resource_gate();assert os.getpriority(os.PRIO_PROCESS,0)==0,'Guard supplies nice10 once'
    lock=job/'LAUNCH_LOCK';lock.mkdir()
    env=dict(os.environ);env.update({k:'1' for k in OUTER_THREADS})
    for k in ('TRAINER_TRUTH_PLAN','TRAINER_TRUTH_PLAN_SHA256'):env.pop(k,None)
    child=['/usr/bin/env']+[k+'='+v for k,v in child_environment(cfg,job).items()]+[cfg['model_python'],'-B',cfg['worker_path'],str(job),'0']
    owner=dict(utc=stamp(),role=item['role'],protocol_sha256=digest,root_grant_sha256=grant_sha,
               expected_guard=str(job/'guard_v1.json'),guard_child_command=child,resource_gate=gate,source_pins=cfg['source_pins'])
    write(job/'owner_v1.json',owner)
    cmd=[cfg['guard_python'],'-B',str(O/'release/guard_footprint_v3.py'),'--record',str(job/'guard_v1.json'),
         '--seconds','1800','--footprint-limit','7500000000','--rss-backstop','7500000000']+child
    (job/'STOP').rename(job/'STOP.prelaunch_v1')
    p=None
    ret=dict(completed=False,owner_process_reaped=False,owned_group_retired=False,
             owner_sha256=sha(job/'owner_v1.json'),error=None)
    failure=None
    try:
        with (job/'worker_v1.log').open('x') as log:
            p=subprocess.Popen(cmd,cwd=O/'trainer',env=env,stdout=log,stderr=subprocess.STDOUT)
            ret['guard_pid']=p.pid
            next_check=time.monotonic()+10
            while p.poll() is None:
                if (job/'results'/(rid+'.json')).exists() or list((job/'results').glob('*.error')):
                    (job/'STOP').touch(exist_ok=True)
                if time.monotonic()>=next_check:
                    ret['last_running_resource_gate']=resource_gate(running=True)
                    next_check=time.monotonic()+10
                time.sleep(1)
            ret.update(guard_exit_code=p.wait(),owner_process_reaped=True)
        source_pins(cfg)
    except BaseException as error:
        failure=error;ret['error']=repr(error)
        (job/'STOP').touch(exist_ok=True)
        if p is not None and p.poll() is None:
            # Only our exact Popen guard receives SIGINT. Its finally retires
            # its own child process group, including descendants.
            try:p.send_signal(signal.SIGINT)
            except ProcessLookupError:pass
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:ret['cleanup_wait_timed_out']=True
        if p is not None and p.poll() is not None:
            ret.update(guard_exit_code=p.wait(),owner_process_reaped=True)
    finally:
        (job/'STOP').touch(exist_ok=True)
        gp=job/'guard_v1.json'
        if p is None:
            ret['owned_group_retired']=True
        elif ret['owner_process_reaped'] and gp.exists():
            g=json.loads(gp.read_text())
            ret.update(guard_path=str(gp),guard_sha256=sha(gp))
            ret['owned_group_retired']=g.get('whole_group_retired') is True and g.get('remaining_rows')==[]
            ret['completed']=failure is None and ret['guard_exit_code']==0 and g.get('completed') is True and ret['owned_group_retired'] and g.get('error') is None
            result=job/'results'/(rid+'.json')
            claims=list((job/'claimed').glob('*.json'))
            if result.exists():ret['result_sha256']=sha(result)
            if len(claims)==1:ret['request_sha256']=sha(claims[0])
        ret['slot_reservation_retained']=not ret['owned_group_retired']
        write(job/'retirement_v1.json',ret)
        if ret['owned_group_retired']:lock.rmdir()
    if failure is not None:raise failure
    assert ret['completed'],'Owned guard failure: stop phase, no retry'
    clean_job(cfg,item,digest)

def verify_grant(cfg,digest,grant_path):
    grant=json.loads(Path(grant_path).read_text())
    assert grant['approved'] is True and grant['purpose']=='recon_prefix_fixed_three_smoke'
    assert grant['protocol_sha256']==digest and grant['runner_sha256']==sha(__file__)
    assert grant['run_name']==cfg['smoke_run_name'] and grant['numerical_slots']==1
    assert grant['fresh_or_oos_evaluations_authorized'] is False
    return grant

def launch(cfg,m,digest,grant_path):
    grant=verify_grant(cfg,digest,grant_path)
    assert all(os.environ.get(k)=='1' for k in OUTER_THREADS),'Outer seven BLAS variables must be1'
    run=home(cfg);lock=run/'PHASE_LOCK';lock.mkdir()
    write(run/'PHASE_OWNER_v1.json',dict(controller_pid=os.getpid(),utc=stamp(),protocol_sha256=digest,grant_sha256=sha(grant_path),source_pins=cfg['source_pins'],expected_guards=[str(run/x['role']/'guard_v1.json') for x in m['jobs']]))
    done=False
    try:
        for i,item in enumerate(m['jobs']):
            run_one(cfg,item,digest,sha(grant_path))
            if i==0:
                z=zero_gate(cfg,m,digest);write(run/'ZERO_GATE_v1.json',z);assert z['passed'],'Exact C zero failed'
            if i==1:
                c,_=read_job(cfg,item,digest);assert all(exact(c,read_reference(cfg,'zero384_624',7100624)).values()),'Full384zero624 reference drift'
        verdict=final_gate(cfg,m,digest);write(run/'SMOKE_GATE_v1.json',verdict)
        write(run/'PHASE_RETIRED_v1.json',dict(utc=stamp(),completed=True,phase_owner_sha256=sha(run/'PHASE_OWNER_v1.json'),guard_set=[str(run/x['role']/'guard_v1.json') for x in m['jobs']],protocol_sha256=digest))
        done=True
    except BaseException as error:
        attempts=[home(cfg)/x['role']/'retirement_v1.json' for x in m['jobs'] if (home(cfg)/x['role']/'owner_v1.json').exists()]
        retired=all(p.exists() and json.loads(p.read_text()).get('owned_group_retired') is True for p in attempts)
        write(run/'PHASE_STOPPED_v1.json',dict(utc=stamp(),error=repr(error),owned_groups_retired=retired,slot_reservation_retained=not retired,
              note='No retry or later smoke job authorized by this failed run.'))
        if retired:lock.rmdir()
        raise
    finally:
        if done:lock.rmdir()
    print(json.dumps(dict(completed=True,passed=verdict['passed'],evaluations=3)))
    return 0 if verdict['passed'] else 2
def _main_body():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','launch','gate','future-plan'))
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--grant',type=Path)
    a=p.parse_args();cfg=registered(a.protocol,a.protocol_sha256)
    if a.action=='prepare':
        assert a.grant, 'Root numerical grant required even for held request preparation'
        verify_grant(cfg,a.protocol_sha256,a.grant)
        prepare(cfg,a.protocol,a.protocol_sha256);return 0
    if a.action=='future-plan':
        # Planning only. Root separately reviews/authorizes finite scientific/price controllers.
        print(json.dumps(cfg['future_interfaces'],indent=2));return 0
    m=manifest(cfg,a.protocol_sha256)
    if a.action=='launch':assert a.grant;return launch(cfg,m,a.protocol_sha256,a.grant)
    assert not (home(cfg)/'PHASE_LOCK').exists()
    root_retired=json.loads((home(cfg)/'ROOT_PHASE_RETIREMENT_v1.json').read_text())
    assert root_retired['approved'] is True and root_retired['managed_exit_code']==0 and root_retired['controller_absent'] is True
    assert root_retired['owner_sha256']==sha(home(cfg)/'PHASE_OWNER_v1.json') and root_retired['retirement_sha256']==sha(home(cfg)/'PHASE_RETIRED_v1.json')
    retired=json.loads((home(cfg)/'PHASE_RETIRED_v1.json').read_text())
    assert retired['completed'] is True and retired['protocol_sha256']==a.protocol_sha256
    assert retired['guard_set']==[str(home(cfg)/x['role']/'guard_v1.json') for x in m['jobs']]
    print(json.dumps(final_gate(cfg,m,a.protocol_sha256),indent=2));return 0
def _interrupted(signum,frame):
    raise KeyboardInterrupt('Owned reconstruction-prefix controller signal%s'%signum)

def main():
    previous=signal.signal(signal.SIGTERM,_interrupted)
    try:return _main_body()
    finally:signal.signal(signal.SIGTERM,previous)

if __name__=='__main__':raise SystemExit(main())
