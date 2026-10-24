"""Export and qualify eight fixed head-only native FD endpoints.
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

design=read(HERE/'DESIGN_LOCAL_v1.json');assert sha(HERE/'DESIGN_LOCAL_v1.json')=='3ade08b96c06461c7cc9d455957b87a8c858d6500ef1227a03e6eff3a769ed44'
prereg=read(HERE/'DIRECTION4_PREREG_v1.json');assert sha(HERE/'DIRECTION4_PREREG_v1.json')=='03bd92249d5cbe2915b5f8d102c7f66105efd041bfa7dd235ad0372ac3e539bb'
gp=RUNS/'truth_response90_direction4_geometry_v1_GUARD.json';guard=read(gp)
assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors']
assert guard['guard_sha256']=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
geometryp=RUNS/'truth_response90_direction4_geometry_v1/DIRECTIONS_v1.json';geometry=read(geometryp)
assert geometry['successful'] and geometry['no_truth_arrays_or_fit_outputs_read'] and geometry['prereg_sha256']==sha(HERE/'DIRECTION4_PREREG_v1.json')
assert geometry['fd_train_seeds']==prereg['fd_train_seeds'] and len(geometry['directions'])==4
ready=read(HERE/'CAPTURE_READY_v1.json');cp=read(ready['plan']);assert sha(ready['plan'])==ready['plan_sha256']
parents={role:V/'msg8_loaderfix_v1'/('candidate_msg8_teacher_loaderfix_'+suffix+'_v1.py') for role,suffix in [('science','sci'),('release','rel')]}
pins={'science':'777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510','release':'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'}
for role,p in parents.items():assert sha(p)==pins[role]
lit,_=f.literals_and_functions(parents['science'].read_text());P=f.decode(lit['_GNN_LITERAL'])
oldast={role:{n.name:ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))} for role,p in parents.items()}
out=HERE/'direction4_endpoints_v1';out.mkdir(exist_ok=False)
checker=HERE.parent/'shared_innovation_head_v1/check_exploratory_import_v1.py'
items=[];rng=np.random.default_rng(20261021040)
for direction in geometry['directions']:
 v=np.asarray(direction['head_delta'],dtype='f8');assert v.shape==(64,) and np.isfinite(v).all()
 for sign,word in [(-1,'minus'),(1,'plus')]:
  name='direction4_d%d_%s'%(direction['index'],word);folder=out/name;folder.mkdir()
  Q={k:a.copy() for k,a in P.items()};Q['head'][1:16]=(P['head'][1:16].astype('f8')+sign*v).astype('f4')
  assert all(np.array_equal(P[k],Q[k]) for k in P if k!='head') and np.array_equal(P['head'][0],Q['head'][0])
  tail='\n# Frozen TRAIN-geometry head-only FD endpoint; never a submission.\nassert _GNN_PARAMS is None and not _E2_GNN_STATE\n_GNN_LITERAL = '+repr(f.encode(Q))+'\n_FAST_RESCUE_FRAC = {}\n'
  outputs={}
  for role,parent in parents.items():
   p=folder/('candidate_'+name+'_'+role+'_v1.py');s=parent.read_text()+tail;compile(s,str(p),'exec')
   with p.open('x') as h:h.write(s)
   tree=ast.parse(s);newast={n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};assert newast==oldast[role]
   literals,_=f.literals_and_functions(s);decoded=f.decode(literals['_GNN_LITERAL']);assert all(np.array_equal(decoded[k],Q[k]) for k in Q)
   outputs[role]=dict(path=str(p),sha256=sha(p),parent_path=str(parent),parent_sha256=sha(parent))
  jwrite(folder/'BUILD_v1.json',dict(utc=utc(),outputs=outputs,direction=direction['index'],sign=sign,geometry_sha256=sha(geometryp),
   all_function_class_AST_unchanged=True,only_changed_parameter='head[1:16]',runtime_shapes_unchanged=True,no_native_launch=True,never_submit_endpoint=True))
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
  items.append(dict(name=name,direction=direction['index'],sign=sign,outputs=outputs,build_sha256=sha(folder/'BUILD_v1.json'),tiny_sha256=sha(folder/'TINY_OFFICIAL_v1.json'),import_sha256=sha(folder/'NUMPY_FREE_IMPORT_v1.json')))
seeds=prereg['fd_train_seeds'];assert len(seeds)==30 and not set(seeds)&set(design['protected_seed_ids'])
rows={}
for seed in seeds:
 rows[str(seed)]={'weights':cp['rows'][str(seed)]['weights']}
 truth=design['rows'][str(seed)]['truth']
 for key in ('A','B'):rows[str(seed)]['truth_'+key]=dict(truth,key='mc_post_'+key)
plan=dict(cp);plan.update(utc=utc(),kind='TRAIN30_NATIVE_FOUR_DIRECTION_ENDPOINTS',diagnostic_only=False,groups=[[seed] for seed in seeds],first7=seeds,held25=[],rows=rows,
 candidates={x['name']:rec(Path(x['outputs']['science']['path'])) for x in items},training_only=True,no_submission=True,geometry_sha256=sha(geometryp),prereg_sha256=sha(HERE/'DIRECTION4_PREREG_v1.json'))
plan['candidates']['exact777a_control']=rec(parents['science'])
pp=out/'PLAN_FD_TRAIN30_v1.json';jwrite(pp,plan)
anchors=prereg['baseline_anchor_seeds'];anchorplan=dict(plan);anchorplan.update(kind='TRAIN30_CAPTURE_BYTE_ANCHORS',groups=[[seed] for seed in anchors],first7=anchors,rows={str(s):rows[str(s)] for s in anchors},candidates={'exact777a_control':rec(parents['science'])})
ap=out/'PLAN_BASELINE_ANCHORS_v1.json';jwrite(ap,anchorplan)
result=dict(utc=utc(),successful=True,endpoints=items,plan=str(pp),plan_sha256=sha(pp),anchor_plan=str(ap),anchor_plan_sha256=sha(ap),
 candidate_source_control_sha256=pins['science'],geometry_sha256=sha(geometryp),geometry_guard_sha256=sha(gp),
 endpoint_prediction_count=240,anchor_prediction_count=3,jobs_per_endpoint=4,shared_max_active=36,blas_threads=4,
 native_group_semantics='one cold seed per process, matching TRAIN90[::3] capture references',no_native_launched=True,truth_arrays_opened=False,script_sha256=sha(__file__),never_submit_endpoints=True)
jwrite(out/'FD_READY_v1.json',result);print(json.dumps({k:v for k,v in result.items() if k!='endpoints'}),flush=True)
