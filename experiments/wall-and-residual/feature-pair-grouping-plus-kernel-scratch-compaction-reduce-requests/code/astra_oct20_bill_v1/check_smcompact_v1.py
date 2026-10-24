"""Tiny current-plan exactness/cache/precompile proof; no estimator import."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

D=Path(__file__).resolve().parent
fixture=D/'check_materialize_v1.py'
tree=ast.parse(fixture.read_text())
cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
         and any(isinstance(t,ast.Name) and t.id=='rng' for t in n.targets))
f={'__file__':str(fixture)}
exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(fixture),'exec'),f)
env,np,fs,fnp=f['env'],f['np'],f['fs'],f['fnp']
old_fn,old_key=env['_astra_sm40_linear_grouped'],env['_sm40_program_key']
tail=D/'tail_smdirect_compact_internal_v1.py'
exec(tail.read_text(),env)
new_fn,new_key=env['_astra_sm40_linear_grouped'],env['_sm40_program_key']
for name in ('_Sm40ProgramReady','_Sm40CapturePrograms','_sm40_precompile_one'):
    f['install'](name)
prepared_cls=env['_Sm40Prepared']
rng=np.random.default_rng(20261022)
receipt=json.loads((D/'SMDIRECT_COMPACT_PREPARED_v1.json').read_text())
cases=[]

def prepare(plan,shape):
    sizes=[len(nodes) for _,nodes in plan[1]]
    total=sum(sizes)
    storage=fnp.empty((total,*shape),dtype=fnp.float32)
    cuts=np.cumsum(sizes)[:-1].tolist()
    return prepared_cls((tuple(fnp.split(storage,cuts,axis=0)),tuple(fnp.unstack(storage,axis=0))))

with fs.BudgetContext(flop_budget=2**34,quiet=True) as ctx:
    for record in env['_SMX_RECORDS'].values():
        for kind in ('L','R'):
            plan=record['prepared'][kind]
            static=next(r for r in receipt['plans'] if r['record']==record['name'] and r['kind']==kind)
            count=max(plan[0].values())+1
            for shape in ((2,3),(3,2),(4,5)):
                for strided in (False,True):
                    inputs=tuple(fnp.asarray(rng.normal(size=shape).astype('float32')) for _ in range(count))
                    prepared=prepare(plan,shape)
                    if strided:
                        owner=fnp.empty((len(plan[2]),shape[0]+2,shape[1]+3),dtype=fnp.float32)
                        out=owner[:,:shape[0],:shape[1]]
                    else:
                        out=fnp.empty((len(plan[2]),*shape),dtype=fnp.float32)
                    stats=dict(temp_peak=0,temp_live=0)
                    env['_sm40_program_key']=old_key
                    before,ops=ctx.flops_used,len(ctx.op_log)
                    old=np.asarray(old_fn(fnp,plan,inputs,shape,stats,out=out,prepared=prepared)).copy()
                    old_bill,old_cold_calls=ctx.flops_used-before,len(ctx.op_log)-ops
                    ko=old_key(plan,inputs,out,prepared)
                    assert len(prepared.programs[ko][2])==static['old_direct_arithmetic_calls']
                    env['_sm40_program_key']=new_key
                    before,ops=ctx.flops_used,len(ctx.op_log)
                    new=np.asarray(new_fn(fnp,plan,inputs,shape,stats,out=out,prepared=prepared)).copy()
                    new_bill,new_cold_calls=ctx.flops_used-before,len(ctx.op_log)-ops
                    kn=new_key(plan,inputs,out,prepared)
                    assert ko!=kn and ko in prepared.programs and kn in prepared.programs
                    assert len(prepared.programs[kn][2])==static['new_compact_arithmetic_calls']
                    assert old_bill==new_bill and np.array_equal(old,new)
                    # Same array owners with changed inputs prove cached operand
                    # references remain live; no stale private-node view survives.
                    for operand in inputs:
                        fnp.copyto(operand,fnp.asarray(rng.normal(size=shape).astype('float32')))
                    before,ops=ctx.flops_used,len(ctx.op_log)
                    new2=np.asarray(new_fn(fnp,plan,inputs,shape,stats,out=out,prepared=prepared)).copy()
                    new_warm_bill,new_warm_calls=ctx.flops_used-before,len(ctx.op_log)-ops
                    env['_sm40_program_key']=old_key
                    before,ops=ctx.flops_used,len(ctx.op_log)
                    old2=np.asarray(old_fn(fnp,plan,inputs,shape,stats,out=out,prepared=prepared)).copy()
                    old_warm_bill,old_warm_calls=ctx.flops_used-before,len(ctx.op_log)-ops
                    assert np.array_equal(old2,new2) and old_warm_bill==new_warm_bill
                    assert old_warm_calls==static['old_direct_arithmetic_calls']
                    assert new_warm_calls==static['new_compact_arithmetic_calls']
                    # Actual setup precompiler must use the new key, retain the
                    # original dict, and stop exactly at insertion before math.
                    env['_sm40_program_key']=new_key
                    p2=prepare(plan,shape)
                    o2=fnp.full((len(plan[2]),*shape),7.25,dtype=fnp.float32)
                    p2dict=p2.programs
                    before=ctx.flops_used
                    assert env['_sm40_precompile_one'](plan,inputs,shape,o2,p2)
                    precompile_flops=ctx.flops_used-before
                    assert precompile_flops==0 and p2.programs is p2dict
                    assert np.all(np.asarray(o2)==np.float32(7.25))
                    assert new_key(plan,inputs,o2,p2) in p2.programs
                    assert old_key(plan,inputs,o2,p2) not in p2.programs
                    assert not env['_sm40_precompile_one'](plan,inputs,shape,o2,p2)
                    assert np.array_equal(old2,np.asarray(new_fn(fnp,plan,inputs,shape,stats,out=o2,prepared=p2)))
                    cases.append(dict(record=record['name'],kind=kind,shape=shape,strided_out=strided,
                        exact_cold_and_changed_input_warm=True,flops_unchanged=True,
                        old_flops=old_bill,new_flops=new_bill,
                        old_cold_calls=old_cold_calls,new_cold_calls=new_cold_calls,
                        old_warm_calls=old_warm_calls,new_warm_calls=new_warm_calls,
                        no_program_collision=True,precompiler_exact_new_key=True,precompile_flops=precompile_flops))

report=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
    status='TINY_OFFICIAL_ALL_ACTUAL_LR_PLANS_EXACT_PASS_NO_ESTIMATOR_IMPORT',official_version=fs.__version__,
    tail_sha256=hashlib.sha256(tail.read_bytes()).hexdigest(),cases=cases,
    total_cases=len(cases),two_numerical_inputs_per_case=True,
    rank35_projection=receipt['rank35_census_projection'],
    limitations=['Tiny public API tests establish exact output, bill and cache/precompile routing on all six actual L/R plans.',
                 'Full native owner lifetime, production requests, memory and wall remain unmeasured.',
                 'No FLOP or adjusted-score gain is claimed; this is exact request consolidation only.'])
out=D/'SMDIRECT_COMPACT_CHECK_v1.json'
assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],cases=len(cases),official_version=fs.__version__,
                     rank35_saved_calls=receipt['rank35_census_projection']['projected_saved_arithmetic_calls'],
                     output=str(out)),indent=2))
