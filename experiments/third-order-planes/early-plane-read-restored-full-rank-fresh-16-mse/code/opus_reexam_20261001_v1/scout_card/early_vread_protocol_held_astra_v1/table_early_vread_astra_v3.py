"""HELD fixed early-read grading. Never read partial fresh/OOS scientific values."""
import hashlib,json,math,re,statistics,struct,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;O=HERE.parents[1];S=O/'scout_card'
WEIGHTED=O/'lead/runs/astra_newideas_v1'
HARNESS=S/'capacity_early_vread_held_astra_v1_v3'
ARMS={'early':'early_vread'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prereg():
    p=HERE/'PREREG_v1.json';d=json.loads(p.read_text())
    assert d['status']=='root_registered_before_own_evaluations' and d['joint_allocation_registered'] is True
    assert d['screen_seeds']==list(range(7101096,7101112)) and d['confirmation_seeds']==list(range(7100592,7100608))
    assert d['native_parity_seeds']==[7101096,7101097] and d['native_price_seeds']==[7100136,7100137]
    assert d['arms']==['early'] and d['r_u']==256 and d['law_fit_blend']==0
    assert sha(d['baseline_params'])==d['baseline_params_sha256']
    for key in ('harness_manifest','matched_C_manifest'):
        p=Path(d[key]);assert sha(p)==d[key+'_sha256'];m=json.loads(p.read_text())
        for name,digest in m['files'].items():assert sha(p.parent/name)==digest,name
    for p,h in d['fixed_source_pins'].items():assert sha(p)==h,p
    for name,h in d['helpers'].items():assert sha(HERE/name)==h,name
    for rid,item in d['reference_request_pins'].items():
        run=Path(item['run'])
        found=list((run/'queue').glob('*'+rid+'.json'))+list((run/'claimed').glob('*'+rid+'.w*.json'))
        assert len(found)==1 and sha(found[0])==item['sha256'],rid
    return d

def clean(phase):
    run=HERE/('run_'+phase)
    owners=sorted(run.glob('launch_owners_*.json'))
    if not owners or (run/'LAUNCH_LOCK').exists():return False
    expected=set()
    for owner_path in owners:
        plan=json.loads(owner_path.read_text())
        if plan.get('phase')!=phase or plan.get('prereg_sha256')!=sha(HERE/'PREREG_v1.json'):return False
        token=plan['token'];receipt=run/('launch_receipt_'+token+'.json')
        if not receipt.exists():return False
        r=json.loads(receipt.read_text())
        if r.get('token')!=token or r.get('phase')!=phase or r.get('errors')!=[]:return False
        # The newest owner must finish the full phase. An earlier root boundary
        # hold is allowed only if explicitly recorded as clean while STOP existed.
        if owner_path==owners[-1]:
            if r.get('completed') is not True:return False
        elif not (r.get('completed') is True or (r.get('completed') is False and r.get('clean_boundary_hold') is True)):
            return False
        if r.get('exit_codes')!=[0]*plan['workers'] or len(r.get('runs',[]))!=plan['workers']:return False
        planned={str(run/f'guard_{token}_w{i}.json') for i in range(plan['workers'])}
        if set(plan['guard_records'])!=planned:return False
        if {x['guard_record'] for x in r['runs']}!=planned:return False
        expected.update(planned)
        for name in planned:
            p=Path(name)
            if not p.exists():return False
            g=json.loads(p.read_text())
            if not (g.get('completed') and g.get('exit_code')==0 and g.get('whole_group_retired') and g.get('remaining_rows')==[] and g.get('error') is None and 0<g.get('max_rss_bytes',1e99)<=7500000000 and 0<=g.get('max_footprint_bytes',1e99)<=7500000000):return False
            if str(run) not in g.get('command',[]):return False
    return {str(p) for p in run.glob('guard_*.json')}==expected


def bits(x):return tuple(bits(v) for v in x) if isinstance(x,list) else struct.pack('>d',float(x))
def path(phase,arm,seed):return HERE/('run_'+phase)/'results'/f'evr_{arm}_r{seed}.json'
def read(p):
    cfg=prereg();p=Path(p)
    assert not p.with_suffix('.error').exists(),str(p)
    d=json.loads(p.read_text());assert d['id']==p.stem
    seed=int(re.search(r'_r(\d+)(?:_|$)',p.stem).group(1));assert d['rank']==seed
    assert d['mode']=='eval' and d['threads']==2 and d['data_source']=='generated'
    assert d['params']==cfg['baseline_params'] and d['f4c'] is True
    assert d.get('truth_plan') is None
    claims=list((p.parent.parent/'claimed').glob('*'+d['id']+'.w*.json'))
    assert len(claims)==1,('missing/ambiguous request',p,claims)
    claim=claims[0];request=json.loads(claim.read_text())
    assert request['id']==d['id'] and request['rank']==seed and request['mode']=='eval'
    assert d['cutsim']==request['cutsim'] and d['params']==request['params']
    base=json.loads((O/'physicist/t141_k4l_ext/claimed/ph141_r7100584_c.w0.json').read_text())
    ignored={'id','rank','cutsim','keep_all'}
    assert {k:v for k,v in request.items() if k not in ignored}=={k:v for k,v in base.items() if k not in ignored},'C request core drift'
    if p.parent.parent.parent==HERE:
        run=p.parent.parent;manifest=json.loads((run/'requests_manifest_v1.json').read_text())
        assert manifest['prereg_sha256']==sha(HERE/'PREREG_v1.json')
        assert sha(claim)==manifest['requests'][d['id']]
        phase=run.name.removeprefix('run_')
        arm='control' if phase=='smoke' else 'rank384_zero' if phase=='zero384' else 'early_vread'
        module=HARNESS/('cutsim_'+arm)/'torch_cutsim_s46_v11.py'
        expected_opts=dict(base['cutsim']['opts'],ru=256,law_fit_blend=0.0)
        assert request['cutsim']==dict(module=str(module),sha256=sha(module),sources={x.name:sha(x) for x in module.parent.glob('*.py')},opts=expected_opts)
        assert bool(request.get('keep_all'))==(phase in ('smoke','zero384','effects'))
    else:
        assert sha(claim)==cfg['reference_request_pins'][d['id']]['sha256']
        assert str(p.parent.parent)==cfg['reference_request_pins'][d['id']]['run']
    assert len(d['mse_layers'])==16 and all(math.isfinite(x) and x>0 for x in d['mse_layers'])
    assert math.isfinite(d['mse']) and d['mse']>0
    assert len(d['final_prediction'])==1024 and all(math.isfinite(x) for x in d['final_prediction'])
    return d


def exact(records):
    keys=('mse','mse_layers','final_prediction')
    checks={k:all(bits(x[k])==bits(records[0][k]) for x in records[1:]) for k in keys}
    return dict(passed=all(checks.values()),checks=checks)

def identity():
    prereg();files=[path('smoke','control',7100584),WEIGHTED/'run_smoke/results/wfit_z_r7100584.json',O/'physicist/t141_k4l_ext/results/ph141_r7100584_c.json']
    if not all(p.exists() for p in files) or not clean('smoke'):return dict(complete=False,passed=False)
    ds=list(map(read,files));assert len(ds[0].get('prediction',[]))==16
    return dict(complete=True,**exact(ds),hashes={str(p):sha(p) for p in files})

def zero384():
    prereg();assert identity()['passed']
    files=[path('zero384','rank384_zero',7100624),S/'run_capacity_smoke_astra_v1/results/capacity_astra_v1_r7100624_cjoin384.json']
    if not all(p.exists() for p in files) or not clean('zero384'):return dict(complete=False,passed=False)
    ds=list(map(read,files));assert len(ds[0].get('prediction',[]))==16
    assert ds[0]['readrank_stats']=={'calls':0,'events':[]}
    return dict(complete=True,**exact(ds),hashes={str(p):sha(p) for p in files})

def effects():
    if not zero384()['passed']:return dict(complete=False,passed=False)
    p=path('effects','early',7100624)
    if not p.exists() or not clean('effects'):return dict(complete=False,passed=False)
    a=read(p);b=read(path('zero384','rank384_zero',7100624));assert len(a.get('prediction',[]))==16
    s=a['readrank_stats'];events=s['events']
    ranks=s['calls']==4 and [x['layer'] for x in events]==[10,11,12,13]
    ranks=ranks and all(x['prefix_v_read_rank']==320 and x['full_state_rank']==384 and x['prefix_rank']==384 and x['common_rank']==256 and x['lazy_prefix']==(x['layer']==13) for x in events)
    checks=dict(unchanged_layers0_to9=bits(a['prediction'][:10])==bits(b['prediction'][:10]),final_changed=bits(a['final_prediction'])!=bits(b['final_prediction']),exact_early_ranks=ranks)
    return dict(complete=True,passed=all(checks.values()),checks=checks,result_sha256=sha(p),smoke_gains_not_used=True)

def ref(seed,confirmation=False):return (O/'physicist/t141_k4l_ext/results'/f'ph141_r{seed}_c.json' if confirmation else WEIGHTED/'run_screen/results'/f'wfit_z_r{seed}.json')
def coverage(confirmation=False):
    d=prereg();seeds=d['confirmation_seeds' if confirmation else 'screen_seeds'];phase='confirm' if confirmation else 'screen'
    assert not [str(path(phase,'early',s).with_suffix('.error')) for s in seeds if path(phase,'early',s).with_suffix('.error').exists()]
    count=sum(path(phase,'early',s).exists() for s in seeds);refs=sum(ref(s,confirmation).exists() for s in seeds)
    return dict(complete=count==16 and refs==16 and clean(phase),arm_count=count,reference_count=refs,guards_retired=clean(phase))

def price():
    cfg=prereg();p=HERE/'NATIVE_PRICE_v1.json';d=json.loads(p.read_text())
    assert d['status']=='root_frozen_before_fresh_summary' and d['prereg_sha256']==sha(HERE/'PREREG_v1.json')
    assert d['aggregation']=='max_pair_log_ratio' and d['candidate_source_sha256']==cfg['price_candidate_source_sha256']
    assert [x['seed'] for x in d['pairs']]==[7100136,7100137]
    assert d['all_noop_resource_route_gates_passed'] is True
    vals=[]
    for pair in d['pairs']:
        bills={}
        for role in ('base','arm'):
            p=Path(pair[role+'_path']);assert sha(p)==pair[role+'_sha256'];r=json.loads(p.read_text())
            assert r['sha256']==d['candidate_source_sha256'][role]
            rows=[x for x in r['rows'] if x.get('seed',x.get('rank'))==pair['seed']];assert len(rows)==1
            bills[role]=rows[0]['flops'];assert math.isfinite(bills[role]) and bills[role]>0
        vals.append(math.log(bills['arm']/bills['base']))
    return max(vals),sha(HERE/'NATIVE_PRICE_v1.json')

def delta(confirmation=False):
    assert coverage(confirmation)['complete'],'No partial scientific values'
    d=prereg();seeds=d['confirmation_seeds' if confirmation else 'screen_seeds'];phase='confirm' if confirmation else 'screen'
    return [math.log(read(path(phase,'early',s))['mse_layers'][-1]/read(ref(s,confirmation))['mse_layers'][-1]) for s in seeds]

def stats(xs,b):return dict(n=len(xs),mean_mse_log_pct=100*statistics.mean(xs),bill_log_pct=100*b,mean_net_log_pct=100*(statistics.mean(xs)+b),se_pct=100*statistics.stdev(xs)/math.sqrt(len(xs)),lower_net=sum(x+b<0 for x in xs))
def screen():
    c=coverage()
    if not c['complete']:return dict(**c,passed=False)
    assert effects()['passed'];b,h=price();out=stats(delta(),b)
    return dict(complete=True,passed=out['mean_net_log_pct']<0,stats=out,native_price_sha256=h)

def confirm():
    decision=screen();assert decision['complete'] and decision['passed'];c=coverage(True)
    if not c['complete']:return dict(**c,passed=False)
    b,h=price();fresh,oos=delta(),delta(True);a=stats(fresh+oos,b);z=stats(oos,b)
    return dict(complete=True,passed=a['mean_net_log_pct']<0 and z['mean_net_log_pct']<0,all32=a,oos16=z,native_price_sha256=h)
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('identity','zero384','effects','screen','confirm')
    print(json.dumps(globals()[sys.argv[1]](),indent=2,allow_nan=False))
