from pathlib import Path
import ast,hashlib,json,resource,time
D=Path(__file__).resolve().parent;R=D.parents[1];t0=time.monotonic()
source=R/'astra_crunch_runtime_v1/failure336657_v1/candidate_setup_freeze_v22b.py';pre=D.parent/'bini_apa_v1/price_outer_232_v3.py'
text=pre.read_text().split('def make_bini')[0].replace("p.SOURCE=O/'lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py'",'p.SOURCE=Path('+repr(str(source))+')')
ns={'__file__':str(pre),'__name__':'read_off_cold_metadata'};exec(text,ns)
M,fn,fs,arena,measure=(ns[k]for k in('M','fn','fs','arena','measure'))
def reset():
 M._astra_sm40_clear();env=M._ASTRA_SM40_ENV;env._SCRATCH=M._SCRATCH;M._astra_sm40_bind_universal(env,arena)
 ker=next(iter(M._ASTRA_SM40_KERNELS.values()))
 for family,args in M._ASTRA_SM40_PREBIND:ker.family=family;ker.side(*args)
 M._mrio_prepare_reshapes();M._mrp_prebind();M._mr_prebind_padding_v1(env);M._mr_prebind_sides_v1()
 for shape in M._MR_GEOMETRY_V1:M._mr_prepare_shape(shape,False,False)
 for shape in((1024,4096,960,0),(992,6144,928,0)):M._mr_prepare_shape(shape,True,True)
# Identify original-mainread lazy costs introduced only by the newly disabled reads.
actual=json.loads((R/'astra_crunch_actual16_tariff_v1/ACTUAL16_WARM_TARIFF_v1.json').read_text());excluded={(1024,4096,960,0),(992,6144,928,0),(992,6144,896,0)}
rows=[]
for net in actual['rows']:
 measure(reset);shapes=[tuple(x['shape'])for x in net['prices']if x['kind']=='read'and tuple(x['shape'])not in excluded]
 costs=[]
 for sh in shapes:costs.append(dict(shape=sh,original_sm40_plan=M._ASTRA_SM40_PLANS.get(sh),cold_original_mainread_views=measure(lambda:M._mr_prepare_shape(sh,True,True))))
 rows.append(dict(rank=net['rank'],shapes=costs,cold_calls=sum(x['cold_original_mainread_views']['requests']for x in costs),cold_flops=sum(x['cold_original_mainread_views']['flops']for x in costs)))
measure(reset);union=[]
for sh in sorted({tuple(x['shape'])for n in actual['rows']for x in n['prices']if x['kind']=='read'and tuple(x['shape'])not in excluded}):
 union.append(dict(shape=sh,plan=M._ASTRA_SM40_PLANS.get(sh),cost=measure(lambda:M._mr_prepare_shape(sh,True,True))))
out=dict(status='METADATA_ONLY',models=0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,union=union,union_calls=sum(x['cost']['requests']for x in union),union_flops=sum(x['cost']['flops']for x in union),seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['Measures only original-mainread topology; generic fallback first-use costs remain separate and are partly covered by expanded fusion prefix.','Each listed rank starts from the same frozen setup, so later sequence costs can be lower.'])
with(D/'READ_OFF_COLD_v1.json').open('x')as f:json.dump(out,f,indent=2)
print(json.dumps({k:out[k]for k in('rows','union_calls','union_flops','seconds','maxrss')}))
