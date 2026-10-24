"""Pure-source reachability and setup exception/identity proof; no arrays."""
import ast,hashlib,json,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1]
source=R/'astra_crunch_runtime_v1/failure336640_v1/candidate_setup_hold_v19e.py'
fragment=D/'fragment_all_push_off_v1.py.txt'
tables={}
for node in ast.parse(source.read_text()).body:
    if isinstance(node,ast.Assign):
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id in ('_PEEL_PUSH_PLANS','_PEEL_READ_PLANS','_SM12_RULE'):
                tables[target.id]=ast.literal_eval(node.value)
push,read=tables['_PEEL_PUSH_PLANS'],tables['_PEEL_READ_PLANS']
push.pop((736,768,6400))
for plans in(push,read):
    for key,p in tuple(plans.items()):
        grid=(192,192,384)if p[0]=='signed_executable_0'else(192,384,192)
        leaf=tuple((v+g-1)//g for v,g in zip(p[2:5],grid))
        for old,new,want in [('peel6_p10_5_5_26','peel6_p10_5_5_21_tight',(5,5,21)),
                             ('peel6_p10_5_21_5','peel6_p10_5_16_5_tight',(5,16,5)),
                             ('peel6_p10_4_30_4','peel6_p10_4_23_4_tight',(4,23,4))]:
            if p[1]==old and leaf==want:plans[key]=(p[0],new)+p[2:]
sm12={k:(rec,fam,*k)for k,(rec,fam)in tables['_SM12_RULE'].items()}
families={p[1]:object()for plans in(push,read,sm12)for p in plans.values()}
four=frozenset((('push',(1024,992,8064)),('push',(1024,992,8128)),('read',(1024,4096,960,0)),('read',(992,6144,928,0))))
env=dict(_HW_REPAIR_SLOW_ROUTES_V1=four,_PEEL_PUSH_PLANS=push,_PEEL_READ_PLANS=read,_SMX_FAMILIES=families,_SM12_PLANS=sm12)
exec(compile(fragment.read_text(),str(fragment),'exec'),env)
checks=[]
for fail in(False,True):
    def parent(t0):
        assert all(('push',k)not in env['_HW_REPAIR_SLOW_ROUTES_V1']for k in env['_PEEL_PUSH_PLANS'])
        assert all(('read',k)not in env['_HW_REPAIR_SLOW_ROUTES_V1']for k in env['_PEEL_READ_PLANS'])
        assert not(set(env['_SMX_FAMILIES'])&env['_DPS_DEAD_FAMILIES_V3'])
        assert len(push)+len(read)-len(env['_PEEL_PUSH_PLANS'])-len(env['_PEEL_READ_PLANS'])==21
        if fail:raise ValueError('test')
        return 'ok'
    env['_DPS_PARENT_BIND_V1']=parent
    try:result=env['_dps_bind_v3'](0)
    except ValueError:assert fail;result='expected_exception'
    assert env['_PEEL_PUSH_PLANS']is push and env['_PEEL_READ_PLANS']is read and env['_SMX_FAMILIES']is families
    checks.append(dict(exception=fail,result=result,restored_original_identities=True))
users={name:[(kind,key)for kind,plans in(('push',push),('read',read),('sm12',sm12))for key,p in plans.items()if p[1]==name]for name in env['_DPS_DEAD_FAMILIES_V3']}
assert all((kind,key)in env['_HW_REPAIR_SLOW_ROUTES_V1']for entries in users.values()for kind,key in entries)
tariff=json.loads((D.parent/'rank48_wide_reaudit_v1/PRICE_v1.json').read_text())
by={(x['kind'],tuple(x['shape'])):x for x in tariff['rows']}
census=json.loads((R/'astra_crunch_actual16_tariff_v1/ACTUAL16_WARM_TARIFF_v1.json').read_text())
rows=[]
for row in census['rows']:
    selected=[by[p['kind'],tuple(p['shape'])]for p in row['prices']if(p['kind'],tuple(p['shape']))in env['_HW_REPAIR_SLOW_ROUTES_V1']]
    added=sum(x['base']['flops']-x['eligible_peel']['flops']for x in selected)
    rows.append(dict(rank=row['rank'],added_bill=added,bill_pct=100*added/row['control_cold_reference_flops'],
                     metadata_request_delta=sum(x['base']['requests']-x['eligible_peel']['requests']for x in selected)))
result=dict(status='PASS',models=0,array_operations=0,fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(),
            dead_families=sorted(env['_DPS_DEAD_FAMILIES_V3']),family_users=users,setup_identity_cases=checks,
            total_original_routes=len(push)+len(read),excluded_routes=21,saved16=rows,
            saved16_mean_bill_penalty_pct=statistics.mean(x['bill_pct']for x in rows),
            saved16_mean_metadata_request_delta=statistics.mean(x['metadata_request_delta']for x in rows),
            saved16_max_bill_penalty_pct=max(x['bill_pct']for x in rows),
            saved16_min_metadata_request_delta=min(x['metadata_request_delta']for x in rows),
            saved16_max_metadata_request_delta=max(x['metadata_request_delta']for x in rows))
with(D/'PROOF_PRUNE_ALL_PUSH_v1.json').open('x')as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items()if k!='saved16'}))
