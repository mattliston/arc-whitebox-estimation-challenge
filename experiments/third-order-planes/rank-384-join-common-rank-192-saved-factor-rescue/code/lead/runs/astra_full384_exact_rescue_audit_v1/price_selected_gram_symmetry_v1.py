from pathlib import Path
import importlib.util,hashlib,json
D=Path(__file__).resolve().parent;O=D.parents[2]
s=importlib.util.spec_from_file_location('tariff',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
p.SOURCE=O/'lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py';p.SOURCE_SHA=hashlib.sha256(p.SOURCE.read_bytes()).hexdigest();M,_=p.load();fnp=p.fnp;fs=p.flops
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 a=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32);M._SCRATCH=[a,M._cp3_bind_arena_views(a,M._PACKED_PUSH),M._cp3_bind_arena_views(a,M._PACKED_READ),None,None,None,None];M._bind_push6(a)
M._WINOGRAD=True;M._AB_ON=True;M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
cache={}
def price(m,k,n):
 key=(m,k,n)
 if key in cache:return cache[key]
 with fs.BudgetContext(flop_budget=2**60,quiet=True): a=fnp.empty((m,k),dtype=fnp.float32);b=fnp.empty((k,n),dtype=fnp.float32);M._shared_old384_mm(a,b)
 with fs.BudgetContext(flop_budget=2**60,quiet=True) as c:M._shared_old384_mm(a,b)
 M._BH_HOLD.clear();cache[key]={'flops':int(c.flops_used),'ops':int(c.ops_used)};return cache[key]
rows=[]
for layer,m,k in [(7,928,6144),(10,896,5568),(13,832,4800)]:
 full=price(m,k,384);outside=price(m-384,k,384);block=price(128,k,128)
 symmetric={v:outside[v]+6*block[v] for v in ('flops','ops')}
 rows.append(dict(layer=layer,shape=[m,k,384],full=full,unselected_rows=outside,selected128_block=block,symmetry_six_blocks=symmetric,saving_before_debits=full['flops']-symmetric['flops']))
out=dict(status='SHAPE_ONLY_NO_MODELS_NO_NUMERICAL_KERNELS',source_sha256=p.SOURCE_SHA,rows=rows,total_saving_before_debits=sum(r['saving_before_debits'] for r in rows),limitations=['Geometry from fixed prior 136 trace; current kernels, not a current full384 census.','Warm prices; optimistic omission of weighted-index gathers, scatter, triangle mirror, cold binding and runtime debt.','Exact-real-arithmetic symmetry of selected Y Gram block; actual floating product order changes.','Previous lowrank core addition is unchanged and not priced as credit.','No candidate installed.'])
f=D/'SELECTED_GRAM_SYMMETRY_TARIFF_v1.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));print(hashlib.sha256(f.read_bytes()).hexdigest())
