"""Export only passing frozen TRAIN90 local-innovation head fits.
No fitting, target access or native evaluation; dispatch receives pinned plans.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp
assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_SERVER_URL')
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
V=HERE.parents[1]/'astra_oct20_vm_frontier_v1'
sys.path.insert(0,str(V));import fit_frontier_v1 as f
sha=f.sha;read=lambda p:json.loads(Path(p).read_text());utc=f.utc
def rec(p):return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),effective_sha256=sha(p))
def jwrite(p,r):
 with p.open('x') as out:json.dump(r,out,indent=2,allow_nan=False);out.write('\n')

assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
fitp=RUNS/'local_innovation90_fit_v1/FITTED_v1.json';fit=read(fitp)
assert fit['status']=='PROXY_FITS_COMPLETE'
assert fit['script_sha256']=='e0c431f98a071e240706f07560955ad64ada870c436d0b4da4d19efab6ad9f49'
prereg=HERE/'PREREG_v1.json';assert sha(prereg)=='970f361d7df4d93afac33dff551080ebe6a36257341057cde0ade59382eba143'
assert fit['prereg_sha256']==sha(prereg) and fit['no_protected_truth_read']
assert fit['design_sha256']=='3ade08b96c06461c7cc9d455957b87a8c858d6500ef1227a03e6eff3a769ed44'
assert len(fit['provenance'])==90 and len({r['seed'] for r in fit['provenance']})==90
finish=read(HERE/'FIT_PIPELINE_FINAL_v1.json')
assert finish['exit_code']==0 and finish['fit_sha256']==sha(fitp)
guardp=RUNS/'local_innovation90_fit_v1_GUARD.json';guard=read(guardp)
assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors']
assert guard['guard_sha256']=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
assert guard['max_owned_swap_bytes']==0
parents={role:V/'msg8_loaderfix_v1'/('candidate_msg8_teacher_loaderfix_'+suffix+'_v1.py') for role,suffix in [('science','sci'),('release','rel')]}
pins={'science':'777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510','release':'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'}
for role,p in parents.items():assert sha(p)==pins[role]
lit,_=f.literals_and_functions(parents['science'].read_text());P=f.decode(lit['_GNN_LITERAL'])
oldast={role:{n.name:ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))} for role,p in parents.items()}
out=HERE/'native_v1';out.mkdir(exist_ok=False)
checker=HERE.parent/'shared_innovation_head_v1/check_exploratory_import_v1.py'
assert sha(checker)=='963ef51a4fce2fd29b044cc6e37cde2842bb7c173e82835b84d6cac91a751765'
items=[];rng=np.random.default_rng(20261021091)
for mode in ('shared64','per_layer960'):
  r=fit['results'][mode]
  if not r['passed']:
    assert r['status']=='REJECT_BEFORE_NATIVE'
    continue
  assert r['status']=='PROXY_GATE_PASS_NATIVE_REQUIRED'
  assert r['loo']['n']==90 and r['loo']['mean_relative_pct']<=-.15 and r['loo']['lower']>=54
  assert sum(b['mean_relative_pct']<0 for b in r['descriptive_blocks'])>=4
  assert r['physical']['overall_correction_to_head_rms']<=.25 and r['physical']['max_per_layer_correction_to_head_rms']<=1.
  c=np.asarray(r['coef_by_layer'],dtype='f8');assert c.shape==(15,64) and np.isfinite(c).all()
  name='local_innovation90_'+mode;folder=out/name;folder.mkdir()
  Q={k:a.copy() for k,a in P.items()};Q['head'][1:16]=(P['head'][1:16].astype('f8')+c).astype('f4')
  np.testing.assert_array_equal(Q['head'][1:16].astype('f8')-P['head'][1:16].astype('f8'),c)
  assert all(np.array_equal(P[k],Q[k]) for k in P if k!='head') and np.array_equal(P['head'][0],Q['head'][0])
  tail='\n# Frozen TRAIN90 local-innovation head fit; full native validation still required.\nassert _GNN_PARAMS is None and not _E2_GNN_STATE\n_GNN_LITERAL = '+repr(f.encode(Q))+'\n_FAST_RESCUE_FRAC = {}\n'
  outputs={}
  for role,parent in parents.items():
   p=folder/('candidate_'+name+'_'+role+'_v1.py');s=parent.read_text()+tail;compile(s,str(p),'exec')
   with p.open('x') as h:h.write(s)
   tree=ast.parse(s);newast={n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};assert newast==oldast[role]
   literals,_=f.literals_and_functions(s);decoded=f.decode(literals['_GNN_LITERAL']);assert all(np.array_equal(decoded[k],Q[k]) for k in Q)
   outputs[role]=dict(path=str(p),sha256=sha(p),parent_path=str(parent),parent_sha256=sha(parent))
  jwrite(folder/'BUILD_v1.json',dict(utc=utc(),outputs=outputs,model=mode,fit_sha256=sha(fitp),prereg_sha256=sha(prereg),
   all_function_class_AST_unchanged=True,only_changed_parameter='head[1:16]',runtime_shapes_unchanged=True,no_native_launch=True,requires_native_validation=True))
  cases=[]
  for n in (16,1024):
   state=rng.normal(size=(n,64)).astype('f4');sigma=rng.uniform(.1,1.8,n).astype('f4')
   with fs.BudgetContext(flop_budget=2**32,quiet=True) as ctx:
    st=fnp.asarray(state);si=fnp.asarray(sigma)
    for layer in range(1,16):
     values=[]
     for params in (P,Q):
      h=fnp.asarray(params['head'][layer]);bias=fnp.asarray(params['head_b'][layer]);rms=fnp.asarray(params['rms'][layer]);gain=1+.25*fnp.tanh(fnp.asarray(params['gain'][layer:layer+1]))
      runs=[]
      for repeat in range(2):
       start,count=ctx.flops_used,len(ctx.op_log);got=si.astype(fnp.float32)*rms*(st@h+bias)*gain
       runs.append((np.asarray(got).copy(),ctx.flops_used-start,len(ctx.op_log)-count))
      np.testing.assert_array_equal(runs[0][0],runs[1][0])
      scalar=np.float32(1)+np.float32(.25)*np.tanh(params['gain'][layer])
      oracle=(sigma*params['rms'][layer]*(state@params['head'][layer]+params['head_b'][layer]))*scalar
      np.testing.assert_allclose(runs[1][0],oracle,atol=2e-10,rtol=3e-6);values.append(runs[1])
     assert values[0][1:]==values[1][1:] and np.isfinite(values[1][0]).all()
     cases.append(dict(n=n,layer=layer,flops=values[0][1],calls=values[0][2],tariff_unchanged=True,repeated_bytes_exact=True))
    np.testing.assert_array_equal(np.asarray(st),state);np.testing.assert_array_equal(np.asarray(si),sigma)
  jwrite(folder/'TINY_OFFICIAL_v1.json',dict(utc=utc(),successful=True,official_version=fs.__version__,source_sha256=outputs['science']['sha256'],cases=cases,input_ownership_preserved=True))
  subprocess.run(['/home/arcworker/arc-envs/client/bin/python','-B',str(checker),'--build',str(folder/'BUILD_v1.json'),'--out',str(folder/'NUMPY_FREE_IMPORT_v1.json')],check=True)
  items.append(dict(name=name,model=mode,outputs=outputs,build_sha256=sha(folder/'BUILD_v1.json'),
    tiny_sha256=sha(folder/'TINY_OFFICIAL_v1.json'),import_sha256=sha(folder/'NUMPY_FREE_IMPORT_v1.json')))
basep=V/'PLAN_BANK8_ACTUAL_SHARED_v1.json'
assert sha(basep)=='cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'
ready=[]
for item in items:
  name=item['name'];folder=out/name;plan=read(basep)
  plan.update(utc=utc(),kind='FROZEN_TRAIN90_LOCAL_INNOVATION_HEAD',reference_submission=337422,
    fit_sha256=sha(fitp),prereg_sha256=sha(prereg),model=item['model'])
  sp=Path(item['outputs']['science']['path'])
  plan['candidates']={'bank337393':plan['candidates']['bank337393'],name:dict(rec(sp),role='frozen_candidate')}
  pp=folder/'PLAN_v1.json';jwrite(pp,plan)
  rr=dict(utc=utc(),successful=True,candidate=name,source_sha256=sha(sp),outputs=item['outputs'],
    plan=str(pp),plan_sha256=sha(pp),fit_sha256=sha(fitp),fit_guard_sha256=sha(guardp),
    tiny_sha256=item['tiny_sha256'],import_sha256=item['import_sha256'],build_sha256=item['build_sha256'],
    requested_jobs=12,shared_max_active=36,blas_threads=4,no_native_launch=True,reference_submission=337422)
  jwrite(folder/'LINUX_READY_v1.json',rr);ready.append(rr)
result=dict(utc=utc(),successful=True,qualified=ready,rejected=[k for k,v in fit['results'].items() if not v['passed']],
    fit_sha256=sha(fitp),script_sha256=sha(__file__),no_native_launch=True,no_adjusted_gain_claim=True)
jwrite(out/'QUALIFIED_v1.json',result)
print(json.dumps(dict(utc=result['utc'],qualified=[r['candidate'] for r in ready],rejected=result['rejected'])),flush=True)
