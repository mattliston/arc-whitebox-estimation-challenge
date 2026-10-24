from pathlib import Path
import importlib.util,json,hashlib
O=Path.cwd();D=O/'lead/runs/astra_mixed_radix15_audit_v1'
spec=importlib.util.spec_from_file_location('virtual',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.SOURCE=O/'lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py';p.SOURCE_SHA='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
m,patched=p.load();fnp=p.fnp;flops=p.flops
with flops.BudgetContext(flop_budget=2**60,quiet=True):
 arena=fnp.empty((m._PACKED_ELEMENTS,),dtype=fnp.float32);m._SCRATCH=[arena,m._cp3_bind_arena_views(arena,m._PACKED_PUSH),m._cp3_bind_arena_views(arena,m._PACKED_READ),None,None,None,None];m._bind_push6(arena)
def price(shape):
 a,k,n=shape
 with flops.BudgetContext(flop_budget=2**60,quiet=True):x=fnp.empty((a,k),dtype=fnp.float32);y=fnp.empty((k,n),dtype=fnp.float32)
 vals=[]
 for phase in ['cold','warm']:
  with flops.BudgetContext(flop_budget=2**60,quiet=True) as ctx:z=m._mm(x,y)
  vals.append(dict(phase=phase,flops=ctx.flops_used,ops=getattr(ctx,"ops_used",None)))
 return vals
shape=(736,8640,768);child=(368,2880,256)
r=dict(parent_sha256=p.SOURCE_SHA,kind='actual tariff with shape-only backend; no numerical kernels or model',shape=shape,child=child,parent=price(shape),child_price=price(child),rank=15,lower_bound_ignores_all_linear_combination_copy_setup_debt=True)
r['warm_no_add_ratio']=15*r['child_price'][1]['flops']/r['parent'][1]['flops'];r['cold_no_add_ratio']=15*r['child_price'][0]['flops']/r['parent'][0]['flops'];r['payload_bytes']=p.PAYLOAD[0]
(D/'ACTUAL_SHAPE_BOUND_v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
