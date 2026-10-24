"""Current-SDK public-operation pricing of literal L0 radius conversion.

Private metadata backend only; common gaussian_raw/derivatives work is omitted
identically on both sides. No estimator/model or new numeric owner is executed.
"""
import ast, collections, hashlib, importlib.util, json, math, resource, sys
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fn
from flopscope import _budget

D=Path(__file__).resolve().parent; R=D.parents[1]
SOURCE=R/'astra_crunch_runtime_v1/failure336631_v1/candidate_setup_repair_v18c.py'
FRAGMENT=R/'astra_crunch_physics_v1/global_radius_v1/fragment_converter_v1.py.txt'
spec=importlib.util.spec_from_file_location('probe',R/'astra_nbmerge_s56_v1/shape_tariff_probe_v2.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
original=p.backend
def backend(f,*args,**kwargs):
    name=getattr(f,'__name__','')
    if name in ('copy','astype') and hasattr(f,'__self__'):
        p.CALLS[name]+=1;a=f.__self__
        return p.virtual(a.shape,args[0] if name=='astype' else a.dtype).view(type(a))
    if name=='outer':
        p.CALLS[name]+=1;return p.virtual((args[0].size,args[1].size),np.result_type(*args))
    if name in ('diag','diagonal'):
        p.CALLS[name]+=1;a=args[0];return p.virtual((min(a.shape),),a.dtype)
    if name=='fill_diagonal':p.CALLS[name]+=1;return None
    if name in ('sum','trace'):
        p.CALLS[name]+=1;return p.virtual((),args[0].dtype)
    return original(f,*args,**kwargs)
old=_budget._call_numpy
for name,module in list(sys.modules.items()):
    if name.startswith('flopscope') and getattr(module,'_call_numpy',None) is old:module._call_numpy=backend

from flopscope.numpy import ndarray
views=collections.Counter();og=ndarray.__getitem__;ot=ndarray.T
def get(self,k):views['getitem']+=1;return og(self,k)
def trans(self):views['T']+=1;return ot.__get__(self,type(self))
ndarray.__getitem__=get;ndarray.T=property(trans)

def selected(source,names,constants=False):
    tree=ast.parse(source)
    nodes=[x for x in tree.body if (isinstance(x,ast.FunctionDef) and x.name in names) or (constants and isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('_GR_') and t.id!='_GR_PARENT_PREDICT' for t in x.targets))]
    return compile(ast.Module(body=nodes,type_ignores=[]),'<selected literal>','exec')

env=dict(math=math,fnp=fn,_ASTRA_DENSE_ON=True,_ASTRA_DENSE_STATS={'l0_calls':0},_PHYS_LAYER=[0],_E2_LAYER={})
env['_ps32']=lambda x:x if str(x.dtype)=='float32' else x.astype(fn.float32)
env['_ps_pre32']=env['_ps32']
with fs.BudgetContext(flop_budget=2**60,quiet=True):
    vec=[fn.empty((1024,),dtype=fn.float64) for _ in range(8)]
    derivs=([fn.empty((1024,),dtype=fn.float64) for _ in range(7)],[fn.empty((1024,),dtype=fn.float64) for _ in range(7)])
    args=(fn.empty((1024,),dtype=fn.float64),fn.empty((1024,1024),dtype=fn.float64),fn.empty((1024,1024),dtype=fn.float64),fn.empty((1024,),dtype=fn.float64))
    planes=[fn.empty((1024,1024),dtype=fn.float32) for _ in range(3)]
    result=fn.empty((16,1024),dtype=fn.float32)
env['gaussian_raw']=lambda *a:vec
env['derivatives']=lambda *a:derivs
exec(selected(SOURCE.read_text(),{'_nonlinear_first_gaussian'}),env);base=env['_nonlinear_first_gaussian']
exec(selected(FRAGMENT.read_text(),{'_gr_convert_first','_nonlinear_first_gaussian'},True),env);new=env['_nonlinear_first_gaussian']

def measure(call):
    p.CALLS.clear();views.clear()
    with fs.BudgetContext(flop_budget=2**60,quiet=True) as c:call()
    return dict(flops=int(c.flops_used),wrapper_calls=sum(p.CALLS.values())+sum(views.values()),backend=dict(p.CALLS),views=dict(views),charged_operations=len(c._op_log))
baseline=measure(lambda:base(*args,metric='covariance'))
candidate=measure(lambda:new(*args,metric='covariance'))
converter=measure(lambda:env['_gr_convert_first'](*planes,*vec[:4]))
restoration=measure(lambda:fn.multiply(result,env['_GR_C1']))
delta={k:candidate[k]-baseline[k]+restoration[k] for k in ('flops','wrapper_calls','charged_operations')}
out=dict(status='METADATA_PUBLIC_TARIFF',source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),fragment_sha=hashlib.sha256(FRAGMENT.read_bytes()).hexdigest(),baseline_L0_excluding_identical_gaussian_helpers=baseline,candidate_L0_excluding_identical_gaussian_helpers=candidate,isolated_converter=converter,all16_mean_restoration=restoration,total_increment=delta,wholebill_percent=100*delta['flops']/243510883029,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['Wrapper calls are not measured client/server requests.','No birth overrides included; physics/fit own consistency and additional tariff.','Common gaussian_raw and derivatives replaced by identical pre-existing vectors on both sides.','Native metadata-only current-SDK execution; no model.'])
(D/'TARIFF_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
