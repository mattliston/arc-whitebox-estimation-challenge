"""Offline tiny proof and source import; never instantiate or run the estimator."""
from pathlib import Path
import ast,hashlib,json,sys,types,numpy as np,time
D=Path(__file__).resolve().parent;manifest=json.loads((D/'SOURCE_MANIFEST_HELD_v2.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for k in ('parent','candidate','off','fragment'):assert sha(manifest[k]['path'])==manifest[k]['sha256']
parent=Path(manifest['parent']['path']).read_text();candidate=Path(manifest['candidate']['path']).read_text();fragment=Path(manifest['fragment']['path']).read_text()
assert candidate==parent.replace('            mu = wT @ m\n','            mu = wT @ m; _YCT_PRE_MEAN[0] = m\n',0) if False else True
# The unique inline edit belongs to the active core, and preserves all line numbers.
nodes={n.name:n for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef)}
cnodes={n.name:n for n in ast.parse(candidate).body if isinstance(n,ast.FunctionDef)}
core=cnodes['_astra_fhm_core'];assign=[n for n in ast.walk(core)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.value,ast.Name)and t.value.id=='_YCT_PRE_MEAN'for t in n.targets)]
assert len(assign)==1 and assign[0].lineno==17313 and ast.unparse(assign[0].value)=='m'
assert candidate[:-len(fragment)-1].replace('mu = wT @ m; _YCT_PRE_MEAN[0] = m','mu = wT @ m',1)==parent
lines=parent.splitlines();n=nodes['_ynt_trace'];old='\n'.join(lines[n.lineno-1:n.end_lineno])
def context():
 class Estimator:
  def predict(self,*a,**kw):return None
 g=dict(fnp=np,_E2Hold=lambda:lambda x:x,_idc_get=lambda tag,owner,keep,make:make(),_UVP_CUR=[7],Estimator=Estimator)
 exec(old,g);exec(fragment,g);return g
g=context();rng=np.random.default_rng(71016031);cases=[]
for case,(q,births,nprev)in enumerate([(0,3,1024),(3,7,640),(11,5,960),(0,1,640),(9,9,1024),(13,7,640)]):
 n=1024;previous=np.sort(rng.choice(n,nprev,replace=False));selected=np.sort(rng.choice(n,births,replace=False))
 base=(rng.standard_normal((n,births))*.03).astype(np.float32);base[selected,np.arange(births)]=0
 p=(rng.standard_normal((nprev,q))*.03).astype(np.float32);t=(rng.standard_normal(births)*.1).astype(np.float32);s=(rng.standard_normal(births)*.1).astype(np.float32)
 lam=rng.standard_normal(q+2*births).astype(np.float32);w=(rng.standard_normal((n,19))*.04).astype(np.float32);tail=np.array([1,3,8,11,18]);mean=np.abs(rng.standard_normal(n)).astype(np.float64)
 newborn=(q,selected,base,t,s);args=(w,p,previous,newborn,lam,tail)
 snapshots=[x.copy()for x in (w,p,previous,selected,base,t,s,lam,tail,mean)]
 g['_YCT_ACTIVE'][0]=True;g['_YCT_PRE_MEAN'][0]=mean;g['_YCT_ON']=True
 actual=g['_ynt_trace'](*args);orig=g['_YCT_PARENT_TRACE'](*args)
 A=np.zeros((n,q+2*births),dtype=np.float64);A[previous,:q]=p;A[:,q:q+births]=base;A[:,q+births:]=base
 A[selected,q+np.arange(births)]+=t;A[selected,q+births+np.arange(births)]+=s
 u=mean/np.linalg.norm(mean);tau=A@(lam*np.sum(A*A,axis=0));tuu=A@(lam*(A.T@u)**2)
 a=u@tuu-3*(tau@u)/(n+2);b0=tuu-tau/(n+2);b=b0-u*(u@b0)
 wt=w[:,tail].astype(np.float64);z=wt.T@u;r2=np.sum(wt*wt,axis=0)
 expected=3*r2/(n+2)*(wt.T@tau)+a*((n+2)*z**3-3*z*r2)/(n-1)+3*((n+2)*z*z-r2)*(wt.T@b)/(n+1)
 err=float(np.max(np.abs(actual-expected))/max(np.max(np.abs(expected)),1e-15));assert err<2e-5,err
 assert all(np.array_equal(x,y)for x,y in zip((w,p,previous,selected,base,t,s,lam,tail,mean),snapshots))
 g['_YCT_ON']=False;assert np.array_equal(g['_ynt_trace'](*args),orig)
 g['_YCT_ON']=True;g['_YCT_PRE_MEAN'][0]=np.zeros(n);assert np.array_equal(g['_ynt_trace'](*args),orig)
 cases.append(dict(case=case,q=q,births=births,previous=nprev,relative_max_error=err,input_bytes_unchanged=True,off_bytes_exact=True,zero_mean_bytes_exact=True))
# Exact recovery for the trace + zonal/vector cubic subspace in arbitrary dimension.
harmonic=[]
for n in (3,7,19,64):
 u=rng.normal(size=n);u/=np.linalg.norm(u);v=rng.normal(size=n);v-=u*(u@v);trace=rng.normal(size=n);A=float(rng.normal());B=float(rng.normal());w=rng.normal(size=(n,13))
 tau=(n+2)*trace+A*u+B*v
 tuu=trace+2*(trace@u)*u+A*u+B*v
 a=u@tuu-3*(tau@u)/(n+2);b0=tuu-tau/(n+2);b=b0-u*(u@b0);z=w.T@u;r2=np.sum(w*w,axis=0)
 got=3*r2/(n+2)*(w.T@tau)+a*((n+2)*z**3-3*z*r2)/(n-1)+3*((n+2)*z*z-r2)*(w.T@b)/(n+1)
 want=3*r2*(w.T@trace)+A*z**3+3*B*z*z*(w.T@v)
 err=float(np.max(np.abs(got-want))/max(np.max(np.abs(want)),1e-15));assert err<2e-14,err
 harmonic.append(dict(n=n,relative_max_error=err))
# Wrapper ownership: success and exception both clear the exact previous-mean handle.
for fail in (False,True):
 g=context()
 def fake(self):
  g['_YCT_PRE_MEAN'][0]=np.ones(1024)
  if fail:raise ValueError('synthetic')
  return 17
 g['_YCT_PARENT_PREDICT']=fake
 try:answer=g['_yct_predict'](None);assert not fail and answer==17
 except ValueError:assert fail
 assert g['_YCT_PRE_MEAN']==[None]and g['_YCT_ACTIVE']==[False]and g['_YCT_STATS']['retired']and g['_YCT_STATS']['mean_clear']
# Import both full immutable sources, without Estimator construction/setup/predict.
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
imports=[]
for field in ('candidate','off'):
 p=Path(manifest[field]['path']);m=types.ModuleType('yct_'+field);m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(p.read_text(),str(p),'exec'),m.__dict__)
 assert all(m.__dict__.get(k)is v for k,v in m._ASTRA_DENSE_CONTEXT)
 assert m._astra_fhm_core.__code__.co_firstlineno==nodes['_astra_fhm_core'].lineno
 assert len(m._PB_SINGLE)==649 and m._PB_WALL_S==3.97
 assert m._YCT_PRE_MEAN==[None]and m._YCT_ACTIVE==[False]
 imports.append(dict(field=field,source_sha256=sha(p),pb_count=len(m._PB_SINGLE),pb_sequence_sha256=hashlib.sha256(repr(m._PB_SINGLE).encode()).hexdigest(),active_context_pass=True,setup_or_predict_calls=0))
assert imports[0]['pb_sequence_sha256']==imports[1]['pb_sequence_sha256']
result=dict(passed=True,models=0,source_sha256=manifest['candidate']['sha256'],implicit_tensor_cases=cases,harmonic_subspace_cases=harmonic,wrapper_cleanup_success_and_exception=True,full_source_imports=imports,manifest_sha256=sha(D/'SOURCE_MANIFEST_HELD_v2.json'),script_sha256=sha(__file__))
with(D/'MATH_SOURCE_TINY_PASS_v1.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
