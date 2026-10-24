"""Build inactive Smirnov arena + frozen RPE/UVP source; no import/model."""
import ast,hashlib,importlib.util,json
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
source=O/'lead/runs/astra_uvp_split_views_v1/candidate_prefix22_uvp_split_HELD_v1.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='507355ec4d4d76f725ae83ef4fc57fdacc19bb622c90e214453dad50791832a4'
helper_path=D.parent/'astra_smirnov_cfull_v1/smirnov_cfull_v1.py'
assert hashlib.sha256(helper_path.read_bytes()).hexdigest()=='239588b3b17772710779173770259abf4aa644eb82cec0ac3b01d86a066df41a'
spec=importlib.util.spec_from_file_location('private_arena_sm40',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
records=json.loads((O.parent/'astra_phase2_reexamination_v1/smirnov_executable_transpose_certificate_v1.json').read_text())['records'];constants={}
for r in records:
 if r['shape']!=[3,6,3]:continue
 prepared={k:helper.prepare_grouped(helper.prepare(r['circuits'][k]))for k in ('L','R','P')}
 for k in prepared: prepared[k]=prepared[k][:-1]+(dict(prepared[k][-1]),)
 constants[r['name']]=dict(shape=r['shape'],prepared=prepared)
price=json.loads((D/'CURRENT24_FULL_KERNEL_PRICE_v1.json').read_text());plans={tuple(r['shape']):(r['best']['name'],r['best']['L'],r['best']['group'])for r in price['rows']if r['best']and r['best']['saving']>0}
s=helper_path.read_text().replace("self.circuits={k:prepare_grouped(prepare(record['circuits'][k]))for k in ('L','R','P')}","self.circuits=record['prepared']")
t=ast.parse(s);wanted={'linear_grouped','family_for','bind_family','slice_side','bind_universal','bind_padding','padding_views','bind_c_full','Kernel'}
nodes=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in wanted]
mapping={n:'_astra_sm40_'+n for n in wanted};mapping.update(Kernel='_AstraSm40Kernel',FAMILIES='_ASTRA_SM40_FAMILIES',math='_rc_math')
class Rename(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id in mapping:n.id=mapping[n.id]
  return n
 def visit_FunctionDef(self,n):
  if n.name in mapping:n.name=mapping[n.name]
  return self.generic_visit(n)
 def visit_ClassDef(self,n):
  if n.name in mapping:n.name=mapping[n.name]
  return self.generic_visit(n)
body=ast.unparse(ast.fix_missing_locations(Rename().visit(ast.Module(body=nodes,type_ignores=[]))))
prefix='''
# Exact rank40 read circuit, over owner-qualified incumbent scratch views.
_ASTRA_SM40_ON = True
_ASTRA_SM40_STATS = dict(calls=0,shapes=[],max_workspace_bytes=0,owner_misses=0,retired=True)
_ASTRA_SM40_BIND_STATS = dict(bound=False,families=0,error=None)
'''
suffix='''
class _AstraSm40Environment:
    pass
_ASTRA_SM40_ENV = _AstraSm40Environment()
_ASTRA_SM40_ENV.fnp = fnp
_ASTRA_SM40_ENV._IPT_REGION = _IPT_REGION
_ASTRA_SM40_ENV._BH_HOLD = _BH_HOLD
_ASTRA_SM40_ENV._TR_LAST = _TR_LAST
_ASTRA_SM40_ENV._SCRATCH = None
_ASTRA_SM40_ENV._rs_nat_perm = _rs_nat_perm
_ASTRA_SM40_ENV._rs_basis_views = _rs_basis_views
_ASTRA_SM40_ENV._sm40_owner = None
_ASTRA_SM40_ENV._sm40_frames = {}
_ASTRA_SM40_ENV._sm40_cache = {}
_ASTRA_SM40_ENV._sm40_outer={}
_ASTRA_SM40_ENV._sm40_outer_cache={}
_ASTRA_SM40_ENV._sm40_io_cache={}
_ASTRA_SM40_KERNELS = {name:_AstraSm40Kernel(_ASTRA_SM40_ENV,record)for name,record in _ASTRA_SM40_RECORDS.items()}
def _astra_sm40_clear():
    env=_ASTRA_SM40_ENV
    for name in ('_sm40_io_cache','_sm40_outer_cache','_sm40_outer','_sm40_cache','_sm40_frames','_sm40_group_cache','_sm40_padding_grid','_sm40_padding_cache','_sm40_cfull','_sm40_ccrop_cache'):
        value=getattr(env,name,None)
        if value is not None:value.clear()
        setattr(env,name,{})
    for name in ('_sm40_extra_leaf','_sm40_group_base','_sm40_padding'):
        setattr(env,name,None)
    env._sm40_extra_leaf_bytes=0
    env._sm40_outer_bytes=0
    env._sm40_owner=None
    env._SCRATCH=None
    _TR_LAST[0]=None
_ASTRA_SM40_PARENT_BIND = _pb_bind
def _pb_bind(t0):
    _ASTRA_SM40_PARENT_BIND(t0)
    if _SCRATCH is None or not _ASTRA_SM40_ON:return
    _astra_sm40_clear()
    _ASTRA_SM40_ENV._SCRATCH=_SCRATCH
    try:
        _astra_sm40_bind_universal(_ASTRA_SM40_ENV,_SCRATCH[0])
        kernel=next(iter(_ASTRA_SM40_KERNELS.values()))
        for family,args in _ASTRA_SM40_PREBIND:
            kernel.family=family
            kernel.side(*args)
        _ASTRA_SM40_BIND_STATS.update(bound=True,families=len(_ASTRA_SM40_ENV._sm40_frames),prebound=len(_ASTRA_SM40_PREBIND),error=None)
        _PB_LOG.append(('sm40_fixed_views',len(_ASTRA_SM40_ENV._sm40_frames),round(_pb_time.monotonic()-t0,3)))
        _PB_LOG.append(('sm40_hot_views',len(_ASTRA_SM40_PREBIND),round(_pb_time.monotonic()-t0,3)))
    except Exception as exc:
        _astra_sm40_clear()
        _ASTRA_SM40_BIND_STATS.update(bound=False,families=0,prebound=0,error=type(exc).__name__)
        _PB_LOG.append(('sm40_views_deferred:'+type(exc).__name__,0,round(_pb_time.monotonic()-t0,3)))
_ASTRA_SM40_PARENT_SETUP = Estimator.setup
def _astra_sm40_setup(self,context):
    _astra_sm40_clear()
    try:return _ASTRA_SM40_PARENT_SETUP(self,context)
    except BaseException:
        _astra_sm40_clear()
        raise
Estimator.setup = _astra_sm40_setup
_ASTRA_SM40_PARENT_READ = _RTT_PARENT_READ
def _astra_sm40_read(left,young_right,lam,live_indices,state):
    old=int(state['columns']);m,k,n=int(left.shape[0]),int(young_right.shape[1]),int(young_right.shape[0])
    rank=int(state['pre_basis'].shape[1])if old else 0
    key=(m,k,n,rank);plan=_ASTRA_SM40_PLANS.get(key)
    if (not _ASTRA_SM40_ON or plan is None or not _WINOGRAD or not _AB_ON
            or str(left.dtype)!=str(young_right.dtype)or str(left.dtype)!='float32'
            or str(lam.dtype)!='float32' or _ASTRA_RLF_CONTEXT[0]is not None
            or _UBX_CTX[0]is not None or _UBC_CONTEXT[0]is not None or _UBX_RC[0]is not None):
        return _ASTRA_SM40_PARENT_READ(left,young_right,lam,live_indices,state)
    if _SCRATCH is None or _ASTRA_SM40_ENV._sm40_owner is not _SCRATCH[0]:
        _ASTRA_SM40_STATS['owner_misses']+=1
        _astra_sm40_clear()
        return _ASTRA_SM40_PARENT_READ(left,young_right,lam,live_indices,state)
    _ASTRA_SM40_ENV._SCRATCH=_SCRATCH
    hold=_E2Hold();old_product=None
    if old:
        inner=hold(_cp_slr_inner_sq(hold(left[:,:old]),hold(lam[:old]),state))
        right=hold(hold(_xsg_rows(5119,state['pre_basis'],live_indices)).T)
        old_product=(inner,right)
    a=hold(left[:,old:]);young_lam=hold(lam[old:])
    if m<n:a=hold(a*young_lam);b=hold(young_right.T)
    else:b=hold(hold(young_right*young_lam).T)
    name,L,P=plan;kernel=_ASTRA_SM40_KERNELS[name]
    result=kernel(a,b,L,P,old_product)
    _ASTRA_SM40_STATS['calls']+=1;_ASTRA_SM40_STATS['shapes'].append(key)
    _ASTRA_SM40_STATS['max_workspace_bytes']=max(_ASTRA_SM40_STATS['max_workspace_bytes'],kernel.stats['conservative_peak_bytes'])
    return result
_RTT_PARENT_READ = _astra_sm40_read
_ASTRA_SM40_PARENT_PREDICT = Estimator.predict
def _astra_sm40_predict(self,*args,**kwargs):
    _ASTRA_SM40_STATS.clear()
    _ASTRA_SM40_STATS.update(calls=0,shapes=[],max_workspace_bytes=0,owner_misses=0,retired=False)
    try:return _ASTRA_SM40_PARENT_PREDICT(self,*args,**kwargs)
    except BaseException:
        _astra_sm40_clear()
        raise
    finally:
        _TR_LAST[0]=None
        _ASTRA_SM40_STATS['retired']=True
Estimator.predict = _astra_sm40_predict
'''
prebind=json.loads((D/'FINAL_PREBIND_PLAN_v14.json').read_text())
prebind_values=tuple((tuple(x['family']),tuple(x['args']))for x in prebind['chosen'])
prefix+='\n_ASTRA_SM40_PREBIND = '+repr(prebind_values)+'\n'
fragment=prefix+'\n_ASTRA_SM40_RECORDS = '+repr(constants)+'\n_ASTRA_SM40_PLANS = '+repr(plans)+'\n_ASTRA_SM40_FAMILIES = '+repr(helper.FAMILIES)+'\n'+body+'\n'+suffix
# Preserve original PB649 implementation; defer two optional read7 prefetch phases.
base=source.read_text();tree=ast.parse(base);lines=base.splitlines(keepends=True);replacements=[]
for node in tree.body:
 if isinstance(node,ast.FunctionDef)and node.name=='_pb_bind':
  text=ast.get_source_segment(base,node)
  if "'read7_fused'" in text:
   replacement="def _pb_bind(t0):\n    _R7_PARENT_BIND(t0)\n    _PB_LOG.append(('read7_lazy',0,round(_pb_time.monotonic()-t0,3)))\n"
   replacements.append((node.lineno,node.end_lineno,replacement))
  if "'yr7_fixed_views'" in text:
   replacement="def _pb_bind(t0):\n    _YR7_PARENT_BIND(t0)\n    _YR7_BIND_STATS.update(keys=0,retired=True)\n    _PB_LOG.append(('yr7_lazy',0,round(_pb_time.monotonic()-t0,3)))\n"
   replacements.append((node.lineno,node.end_lineno,replacement))
assert len(replacements)==2
cov_dir=D.parent/'astra_cov_output_views_v1'
cov_source=(cov_dir/'candidate_cov_output_views_HELD_v1.py').read_text()
cov_parent=(D.parent/'astra_uvp_view_hold_v1/candidate_prefix22_uvp_views_HELD_v1.py').read_text()
def functions(text):
 return {n.name:ast.get_source_segment(text,n)for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
cf,pf,bf=functions(cov_source),functions(cov_parent),functions(base)
cov_names=[name for name in cf if name in pf and cf[name]!=pf[name]]
assert len(cov_names)==5,cov_names
for node in tree.body:
 if isinstance(node,ast.FunctionDef)and node.name in cov_names:
  assert bf[node.name]==pf[node.name],node.name
  replacements.append((node.lineno,node.end_lineno,cf[node.name]+'\n'))
for start,end,replacement in sorted(replacements,reverse=True):lines[start-1:end]=[replacement]
base=''.join(lines)
cov_fragment=cov_source.split('# Per-prediction covariance output scratch.',1)[1]
cov_fragment='\n# Per-prediction covariance output scratch.'+cov_fragment
base+=cov_fragment
target=D/'candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v14.py';result=base+fragment;compile(result,str(target),'exec')
with target.open('x')as f:f.write(result)
with(D/'fragment_smirnov_arena_HELD_v14.py').open('x')as f:f.write(fragment)
manifest=dict(candidate=str(target),candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),fixed_plans=len(plans),prebind_requests=prebind['spent'],prebind_count=len(prebind_values),cov_functions=cov_names,scope='Inactive build only; frozen exact RPE/UVP prefix plus mathematically same Smirnov outer kernel. Owner-qualified shared arena; lazy bounded structural views; setup-only universal binding; old-inner hook retained once; unknown/invalidowner fallback. No model.')
with(D/'BUILD_COMBINED_ARENA_v14.json').open('x')as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps(manifest,indent=2))
