"""Build inactive Smirnov arena + frozen RPE/UVP source; no import/model."""
import ast,hashlib,importlib.util,json
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
source=O/'lead/runs/astra_uvp_view_hold_v1/candidate_prefix22_uvp_views_HELD_v1.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='b5268d4d55fab95c3929e09bd9956f5092efc7b39184b0bb3faac0dcaef256e4'
helper_path=D/'smirnov_arena_fused_v4.py'
assert hashlib.sha256(helper_path.read_bytes()).hexdigest()=='00832129adca0a48657f348bd8b5cf447b3e9c24db4a4defad84a4f9490bdd36'
spec=importlib.util.spec_from_file_location('private_arena_sm40',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
records=json.loads((O.parent/'astra_phase2_reexamination_v1/smirnov_executable_transpose_certificate_v1.json').read_text())['records'];constants={}
for r in records:
 if r['shape']!=[3,6,3]:continue
 constants[r['name']]=dict(shape=r['shape'],prepared={k:helper.prepare_slots(helper.prepare(r['circuits'][k]))for k in ('L','R','P')})
price=json.loads((D/'CURRENT24_FULL_KERNEL_PRICE_v1.json').read_text());plans={tuple(r['shape']):(r['best']['name'],r['best']['L'],r['best']['group'])for r in price['rows']if r['best']and r['best']['saving']>0}
s=helper_path.read_text().replace("self.circuits={k:prepare_slots(prepare(record['circuits'][k]))for k in ('L','R','P')}","self.circuits=record['prepared']")
t=ast.parse(s);wanted={'linear_slots','family_for','bind_family','slice_side','bind_universal','Kernel'}
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
_ASTRA_SM40_KERNELS = {name:_AstraSm40Kernel(_ASTRA_SM40_ENV,record)for name,record in _ASTRA_SM40_RECORDS.items()}
def _astra_sm40_clear():
    _ASTRA_SM40_ENV._sm40_cache.clear()
    _ASTRA_SM40_ENV._sm40_frames.clear()
    _ASTRA_SM40_ENV._sm40_owner=None
    _ASTRA_SM40_ENV._SCRATCH=None
    _TR_LAST[0]=None
_ASTRA_SM40_PARENT_SETUP = Estimator.setup
def _astra_sm40_setup(self,context):
    _ASTRA_SM40_PARENT_SETUP(self,context)
    if (int(getattr(context,'width',0))!=1024 or int(getattr(context,'depth',0))!=16
            or _SCRATCH is None or not _ASTRA_SM40_ON):return
    if _ASTRA_SM40_ENV._sm40_owner is _SCRATCH[0]:return
    _astra_sm40_clear()
    _ASTRA_SM40_ENV._SCRATCH=_SCRATCH
    try:
        try:_astra_sm40_bind_universal(_ASTRA_SM40_ENV,_SCRATCH[0])
        except Exception as exc:
            if type(exc).__name__!='NoBudgetContextError':raise
            with flops.BudgetContext(flop_budget=2**40,quiet=True,namespace='sm40views'):
                _astra_sm40_bind_universal(_ASTRA_SM40_ENV,_SCRATCH[0])
        _ASTRA_SM40_BIND_STATS.update(bound=True,families=len(_ASTRA_SM40_ENV._sm40_frames),error=None)
    except Exception as exc:
        _astra_sm40_clear()
        _ASTRA_SM40_BIND_STATS.update(bound=False,families=0,error=type(exc).__name__)
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
fragment=prefix+'\n_ASTRA_SM40_RECORDS = '+repr(constants)+'\n_ASTRA_SM40_PLANS = '+repr(plans)+'\n_ASTRA_SM40_FAMILIES = '+repr(helper.FAMILIES)+'\n'+body+'\n'+suffix
target=D/'candidate_smirnov_arena_RPE_HELD_v4.py';result=source.read_text()+fragment;compile(result,str(target),'exec')
with target.open('x')as f:f.write(result)
with(D/'fragment_smirnov_arena_HELD_v4.py').open('x')as f:f.write(fragment)
manifest=dict(candidate=str(target),candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),fixed_plans=len(plans),scope='Inactive build only; frozen exact RPE/UVP prefix plus mathematically same Smirnov outer kernel. Owner-qualified shared arena; lazy bounded structural views; setup-only universal binding; old-inner hook retained once; unknown/invalidowner fallback. No model.')
with(D/'BUILD_COMBINED_ARENA_v4.json').open('x')as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps(manifest,indent=2))
