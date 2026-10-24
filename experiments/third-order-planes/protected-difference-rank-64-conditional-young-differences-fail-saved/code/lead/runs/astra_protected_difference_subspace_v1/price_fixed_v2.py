"""Current FNP tariffs, metadata backend only: fixed protected difference64."""
from pathlib import Path
import os,importlib.util,hashlib,json,resource
assert os.getpriority(os.PRIO_PROCESS,0)>=10
D=Path(__file__).resolve().parent;O=D.parents[2]
sp=importlib.util.spec_from_file_location('tariff',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
parent_backend=p.backend
# Ensure linalg wrappers participate in the backend replacement.
unused=(p.fnp.linalg.qr,p.fnp.linalg.solve,p.fnp.linalg.eigh)
def backend(fn,*args,**kwargs):
 name=getattr(fn,'__name__','')
 if name=='qr':
  p.CALLS[name]+=1;m,n=args[0].shape;r=min(m,n);return p.virtual((m,r),args[0].dtype),p.virtual((r,n),args[0].dtype)
 if name=='solve':p.CALLS[name]+=1;return p.virtual(args[1].shape,args[1].dtype)
 if name=='eigh':p.CALLS[name]+=1;n=args[0].shape[0];return p.types.SimpleNamespace(eigenvalues=p.virtual((n,),args[0].dtype),eigenvectors=p.virtual((n,n),args[0].dtype))
 return parent_backend(fn,*args,**kwargs)
p.backend=backend;p.SOURCE=O/'lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py';p.SOURCE_SHA=hashlib.sha256(p.SOURCE.read_bytes()).hexdigest();M,_=p.load();fnp=p.fnp;fs=p.flops
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32);M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None];M._bind_push6(arena)
M._WINOGRAD=True;M._AB_ON=True;M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
cache={};records=[]
def mm(name,m,k,n):
 key=(m,k,n)
 if key not in cache:
  with fs.BudgetContext(flop_budget=2**60,quiet=True):a=fnp.empty((m,k),dtype=fnp.float32);b=fnp.empty((k,n),dtype=fnp.float32)
  with fs.BudgetContext(flop_budget=2**60,quiet=True)as c:M._shared_old384_mm(a,b)
  cache[key]=int(c.flops_used);M._BH_HOLD.clear()
 records.append(dict(name=name,shape=key,flops=cache[key]));return cache[key]
def lin(name,kind,m,n,dtype):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):a=fnp.empty((m,n),dtype=dtype);b=fnp.empty((m,n),dtype=dtype)
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as c:
  if kind=='qr':fnp.linalg.qr(a,mode='reduced')
  elif kind=='solve':fnp.linalg.solve(a,b)
  else:fnp.linalg.eigh(a)
 value=int(c.flops_used);records.append(dict(name=name,shape=[m,n],dtype=str(dtype),flops=value));return value
m,k,old,oldr,pairs=864,5632,6144,384,5888
parent=sum([mm('parent_Y_join',m,k,320),mm('parent_Y_core1',oldr,oldr,320),mm('parent_Y_core2',m,oldr,320),lin('parent_QR','qr',m,320,fnp.float32),lin('parent_core_solve','solve',320,320,fnp.float32)])
arm=sum([mm('arm_Y_join',m,k,384),mm('arm_Y_core1',oldr,oldr,384),mm('arm_Y_core2',m,oldr,384),lin('arm_QR','qr',m,384,fnp.float32),lin('arm_core_solve','solve',384,384,fnp.float32)])
extras=sum([mm('extra_difference_join_projection',64,m,k//2),mm('extra_old_overlap',64,m,oldr),mm('extra_old_difference_projection',64,oldr,old//2),mm('difference_Gram128',128,pairs,128),lin('difference_eigh128','eigh',128,128,fnp.float64),mm('difference_coeff_rotation',64,128,pairs),mm('basis_tail_rotation',m,128,64),mm('core_cross_rotation',256,128,64),mm('core_tail_rotation1',64,128,128),mm('core_tail_rotation2',64,128,64)])
# Copies, differences, weights, gathers, casts, symmetrization and new canonical
# +/- coefficient packing. Generous lower allowance, separately labeled.
linear=10*m*384+4*m*(k//2)+5*128*pairs+4*320*(old+k)+10*384*384
total=arm-parent+extras+linear
out=dict(status='CURRENT_FNP_FIXED_ALGORITHM_SHAPE_TARIFF',source_sha256=p.SOURCE_SHA,geometry=dict(active_rows=m,joining_columns=k,old_columns=old,pairs=pairs),parent_replaced_flops=parent,temporary384_replacement_flops=arm,additional_fixed_subspace_flops=extras,linear_element_allowance=linear,total_incremental_flops=total,wholebill_overhead_pct=100*total/238299111296,records=records,models=0,captures=0,numerical_payload_bytes=p.PAYLOAD[0],rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits=['Historical saved geometry through current tariffs, not current full-model bill.','Only extra difference coefficients are formed; no unnecessary extra common coefficients or dense full384 factor rotation.','Core is transformed in sparse256+64 block structure, not a dense320x384 map.','Unchanged320 later read/push/reconstruction credited as unchanged; actual route/rounding effects would require native test.','Linear element allowance is not a native request/cold-cache certificate.'])
(D/'FIXED_COST_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='records'}))
