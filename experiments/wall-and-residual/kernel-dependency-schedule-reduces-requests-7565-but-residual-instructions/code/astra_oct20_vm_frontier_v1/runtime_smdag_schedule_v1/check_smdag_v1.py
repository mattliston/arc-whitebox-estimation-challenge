"""Official current-plan cold/warm, ownership and precompiler checks only."""
import ast, hashlib, json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent
D = RUNS/'astra_oct20_bill_v1'
fixture = HERE.parent/'runtime_current337466_v1/check_sm_v1.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
tree = ast.parse(fixture.read_text())
cut = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='rng' for t in n.targets))
f = {'__file__': str(fixture)}
exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(fixture),'exec'),f)
env,np,fs,fnp = f['env'],f['np'],f['fs'],f['fnp']
assert hashlib.sha256(f['source'].encode()).hexdigest() == '434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4'
old_fn, old_key, old_runs = (env[n] for n in ('_astra_sm40_linear_grouped','_sm40_program_key','_o20_smdirect_runs'))
tail = HERE/'tail_smdag_schedule_v1.py'
assert sha(tail) == '4f15f88a0b47a8a3468a5a28c021d5d5141abc3f15671c14ce31f264bdc9f8ec'
old_registry=dict(env['_O20_SMDIRECT_PLANS'])
old_plans={(r['name'],kind):r['prepared'][kind] for r in env['_SMX_RECORDS'].values() for kind in ('L','R')}
import types
env['_SMX_ENV']=types.SimpleNamespace();env['_ASTRA_SM40_ENV']=types.SimpleNamespace()
exec(tail.read_text(),env)
new_registry=dict(env['_O20_SMDIRECT_PLANS'])
new_fn, new_key, new_runs = (env[n] for n in ('_astra_sm40_linear_grouped','_sm40_program_key','_o20_smdirect_runs'))
for name in ('_Sm40ProgramReady','_Sm40CapturePrograms','_sm40_precompile_one'):
    f['install'](name)
prepared_cls = env['_Sm40Prepared']
rng = np.random.default_rng(20261022)
receipt = json.loads((D/'SMDIRECT_COMPACT_AP_PREPARED_v1.json').read_text())
newprice=json.loads((HERE/'PRICE_METADATA_v2.json').read_text())
cases = []

def route(new):
    env['_sm40_program_key'] = new_key if new else old_key
    env['_o20_smdirect_runs'] = new_runs if new else old_runs
    env['_O20_SMDIRECT_PLANS'] = new_registry if new else old_registry

def prepare(plan,shape):
    sizes = [len(nodes) for _,nodes in plan[1]]
    storage = fnp.full((sum(sizes),*shape),-17.25,dtype=fnp.float32)
    cuts = np.cumsum(sizes)[:-1].tolist()
    return prepared_cls((tuple(fnp.split(storage,cuts,axis=0)),tuple(fnp.unstack(storage,axis=0))))

with fs.BudgetContext(flop_budget=2**34,quiet=True) as ctx:
    for record in env['_SMX_RECORDS'].values():
        for kind in ('L','R'):
            plan = record['prepared'][kind]
            oldplan = old_plans[record['name'],kind]
            assert plan[0]==oldplan[0] and plan[2:]==oldplan[2:]
            proposed=next(r for r in newprice['plans'] if r['record']==record['name'] and r['kind']==kind)['proposed_calls']
            static = next(r for r in receipt['plans'] if r['record']==record['name'] and r['kind']==kind)
            count = max(plan[0].values())+1
            for shape in ((2,3),(3,2),(4,5)):
                for strided in (False,True):
                    inputs = tuple(fnp.asarray(rng.normal(size=shape).astype('float32')) for _ in range(count))
                    input_copies = [np.asarray(x).copy() for x in inputs]
                    prepared = prepare(plan,shape)
                    oldprepared = prepare(oldplan,shape)
                    if strided:
                        owner = fnp.full((len(plan[2]),shape[0]+2,shape[1]+3),123.25,dtype=fnp.float32)
                        out = owner[:,:shape[0],:shape[1]]
                    else:
                        out = fnp.empty((len(plan[2]),*shape),dtype=fnp.float32)
                    stats = dict(temp_peak=0,temp_live=0)
                    route(False)
                    before,ops = ctx.flops_used,len(ctx.op_log)
                    old_ret = old_fn(fnp,oldplan,inputs,shape,stats,out=out,prepared=oldprepared)
                    old = np.asarray(old_ret).copy()
                    old_bill,old_cold_calls = ctx.flops_used-before,len(ctx.op_log)-ops
                    assert old_ret is out
                    ko = old_key(oldplan,inputs,out,oldprepared)
                    assert len(oldprepared.programs[ko][2])==static['old_direct_arithmetic_calls']
                    route(True)
                    before,ops = ctx.flops_used,len(ctx.op_log)
                    new_ret = new_fn(fnp,plan,inputs,shape,stats,out=out,prepared=prepared)
                    new = np.asarray(new_ret).copy()
                    new_bill,new_cold_calls = ctx.flops_used-before,len(ctx.op_log)-ops
                    assert new_ret is out
                    kn = new_key(plan,inputs,out,prepared)
                    assert ko!=kn and ko in oldprepared.programs and kn in prepared.programs
                    assert len(prepared.programs[kn][2])==proposed
                    assert old_bill==new_bill and np.array_equal(old,new)
                    assert all(np.array_equal(x,np.asarray(y)) for x,y in zip(input_copies,inputs))
                    for operand in inputs:
                        fnp.copyto(operand,fnp.asarray(rng.normal(size=shape).astype('float32')))
                    input_copies = [np.asarray(x).copy() for x in inputs]
                    before,ops = ctx.flops_used,len(ctx.op_log)
                    new2 = np.asarray(new_fn(fnp,plan,inputs,shape,stats,out=out,prepared=prepared)).copy()
                    new_warm_bill,new_warm_calls = ctx.flops_used-before,len(ctx.op_log)-ops
                    route(False)
                    before,ops = ctx.flops_used,len(ctx.op_log)
                    old2 = np.asarray(old_fn(fnp,oldplan,inputs,shape,stats,out=out,prepared=oldprepared)).copy()
                    old_warm_bill,old_warm_calls = ctx.flops_used-before,len(ctx.op_log)-ops
                    assert np.array_equal(old2,new2) and old_warm_bill==new_warm_bill
                    assert old_warm_calls==static['old_direct_arithmetic_calls']
                    assert new_warm_calls==proposed
                    assert all(np.array_equal(x,np.asarray(y)) for x,y in zip(input_copies,inputs))
                    route(True)
                    p2 = prepare(plan,shape)
                    o2 = fnp.full((len(plan[2]),*shape),7.25,dtype=fnp.float32)
                    p2dict = p2.programs
                    before = ctx.flops_used
                    assert env['_sm40_precompile_one'](plan,inputs,shape,o2,p2)
                    precompile_flops = ctx.flops_used-before
                    assert precompile_flops==0 and p2.programs is p2dict
                    assert np.all(np.asarray(o2)==np.float32(7.25))
                    assert new_key(plan,inputs,o2,p2) in p2.programs
                    assert old_key(plan,inputs,o2,p2) not in p2.programs
                    assert not env['_sm40_precompile_one'](plan,inputs,shape,o2,p2)
                    assert np.array_equal(old2,np.asarray(new_fn(fnp,plan,inputs,shape,stats,out=o2,prepared=p2)))
                    # New private nodes fit inside each ORIGINAL group owner;
                    # their unused suffix and all external input planes survive.
                    output_refs = set(plan[2])
                    for batch,(_,nodes) in zip(p2[0],plan[1]):
                        internal = sum(ref not in output_refs for ref,args in nodes)
                        assert np.all(np.asarray(batch)[internal:]==np.float32(-17.25))
                    assert all(np.array_equal(x,np.asarray(y)) for x,y in zip(input_copies,inputs))
                    if strided:
                        owner_np = np.asarray(owner)
                        assert np.all(owner_np[:,shape[0]:,:]==np.float32(123.25))
                        assert np.all(owner_np[:,:shape[0],shape[1]:]==np.float32(123.25))
                    cases.append(dict(record=record['name'],kind=kind,shape=shape,strided_out=strided,
                        exact_cold_and_changed_input_warm=True,flops_unchanged=True,
                        old_flops=old_bill,new_flops=new_bill,
                        old_cold_calls=old_cold_calls,new_cold_calls=new_cold_calls,
                        old_warm_calls=old_warm_calls,new_warm_calls=new_warm_calls,
                        no_program_collision=True,precompiler_exact_new_key=True,
                        precompile_flops=precompile_flops,output_owner_identity=True,
                        untouched_outside_output_view=True,private_group_suffix_untouched=True,
                        input_bytes_unchanged=True,old_max_group_nodes=max(len(x[1]) for x in oldplan[1]),new_max_group_nodes=max(len(x[1]) for x in plan[1])))
r = dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
         successful=True,official_version=fs.__version__,source_sha256=hashlib.sha256(f['source'].encode()).hexdigest(),
         script_sha256=sha(__file__),fixture_sha256=sha(fixture),tail_sha256=sha(tail),
         total_cases=len(cases),cases=cases,historical_rank35_projection=receipt['rank35_census_projection'],
         limitation='Current six-plan official checks only. Whole-network requests, residual, owner lifetime and tariff still require own quiet3; no native score claim.')
with (HERE/'SMDAG_CHECK_v1.json').open('x') as out: json.dump(r,out,indent=2);out.write('\n')
print(json.dumps(dict(successful=True,cases=len(cases),source_sha256=r['source_sha256'])))
