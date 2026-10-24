"""NumPy-free full imports and exact single-expression source qualification."""
import ast,hashlib,importlib.util,json,subprocess,sys,types
from pathlib import Path
H=Path(__file__).resolve().parent;assert importlib.util.find_spec('numpy') is None
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((H/'BUILD_v1.json').read_text());stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
R=Path('/home/arcworker/arc-workspace')
def local(s):
 p=Path(s)
 if p.exists():return p
 return R/str(p).removeprefix('/Users/matt/arc-white-box-estimation/')
def module(p,name):
 m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m;exec(compile(p.read_text(),str(p),'exec'),m.__dict__);return m
old="_GCZ_STATS.update(predicts=_GCZ_STATS['predicts'] + 1, frozen_objects=_setup_gc.get_freeze_count())";new="_GCZ_STATS.update(predicts=_GCZ_STATS['predicts'] + 1, frozen_objects=None)"
cache={};cases=[]
for key,roles in build['outputs'].items():
 for role,r in roles.items():
  p,q=local(r['path']),local(r['parent_path']);assert sha(p)==r['sha256'] and sha(q)==r['parent_sha256'];assert q.read_text().replace(old,new)==p.read_text()
  ck=(r['parent_sha256'],role)
  if ck not in cache:cache[ck]=module(q,'census_parent_'+key+'_'+role)
  a=cache[ck];b=module(p,'census_candidate_'+key+'_'+role)
  assert a._GCY_PARENT_PREDICT is a._gcz_predict_v1 and b._GCY_PARENT_PREDICT is b._gcz_predict_v1
  assert a._GCZ_PARENT_PREDICT.__code__.co_code==b._GCZ_PARENT_PREDICT.__code__.co_code
  for m in (a,b):
   assert m._FAST_RESCUE_FRAC=={} and m._O20_SMDIRECT_ON is False and m._O20_NPX_COPY_EXACT_ON is False
   assert all(m.__dict__.get(n) is v for n,v in m._ASTRA_DENSE_CONTEXT)
   assert m._K3LB_PARENT_LEFT is m._o20_npx_copy_exact_left_v1
   assert m._GCZ_STATS==dict(predicts=0,frozen_objects=0)
  traces=[]
  for raise_parent in (False,True):
   outcomes=[]
   for m in (a,b):
    trace=[];stats=dict(predicts=2,frozen_objects=123);sentinel=object()
    class G:
     def freeze(self):trace.append('freeze')
     def get_freeze_count(self):trace.append('census');return 456
    def parent(self,mlp,budget):
     trace.append('parent')
     if raise_parent:raise RuntimeError('fixture-original-parent-error')
     return sentinel
    fn=types.FunctionType(m._gcz_predict_v1.__code__,dict(_GCZ_PARENT_PREDICT=parent,_setup_gc=G(),_GCZ_STATS=stats))
    try:result=fn(None,None,None);outcome=('returned',result is sentinel)
    except RuntimeError as e:outcome=('raised',str(e))
    outcomes.append(outcome);assert trace==(['parent','freeze','census'] if m is a else ['parent','freeze']);assert stats['predicts']==3;assert stats['frozen_objects']==(456 if m is a else None)
   assert outcomes[0]==outcomes[1];traces.append(dict(parent_raises=raise_parent,outcomes=outcomes))
  cases.append(dict(key=key,role=role,sha256=sha(p),parent_sha256=sha(q),parent_cache_key=list(ck),full_import=True,captured_wrapper_live=True,empty_rescue=True,exception_and_finally_order=traces))
assert len(cache)==4 and len(cases)==4 and 'numpy' not in sys.modules
x=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),successful=True,numpy_free=True,no_setup=True,no_prediction=True,no_numeric_arrays=True,only_extracted_wrapper_metadata_fixture=True,distinct_parent_cache_entries=len(cache),parent_cache_key='(parent_sha256,role)',cases=cases,build_sha256=sha(H/'BUILD_v1.json'),script_sha256=sha(Path(__file__)))
with(H/'IMPORT_CHECK_v1.json').open('x')as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(x))
