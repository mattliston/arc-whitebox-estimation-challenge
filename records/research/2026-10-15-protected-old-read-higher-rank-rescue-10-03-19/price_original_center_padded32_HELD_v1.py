from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent
code=(D/'price_protected_range_HELD_v1.py').read_text()
exec(compile(code[:code.index('base={')],str(D/'price_protected_range_HELD_v1.py'),'exec'))
rows=[]
for a in shapes:
 l,m,old,dim=[a[k]for k in ('layer','m','old_columns','old_rank')]
 height=((3*m+255)//256)*64;center=height-2;rank=32;actual=center+rank;packed=((actual+63)//64)*64
 base=inner(l,m,old,dim);small=inner(l,packed,old,dim);extra=overhead(m,old,dim,rank,center)
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  cent=fnp.empty((center,old),dtype=fnp.float32);proj=fnp.empty((rank,old),dtype=fnp.float32);zero=fnp.asarray(0,dtype=fnp.float32)
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:
  z=fnp.empty((packed,old),dtype=fnp.float32)
  fnp.copyto(z[:center],cent);fnp.copyto(z[center:actual],proj);fnp.copyto(z[actual:],zero)
 extra['pack']=dict(flops=int(ctx.flops_used),ops=len(ctx._op_log))
 debit=sum(v['flops']for v in extra.values())
 rows.append(dict(layer=l,m=m,center=center,rank=rank,outer=m-center,logical_packed=actual,padded=packed,padrows=packed-actual,baseline=base,reduced=small,overhead=extra,warm_saving=base['warm']['flops']-small['warm']['flops']-debit,cold_saving=base['cold']['flops']-small['cold']['flops']-debit))
warm=sum(x['warm_saving']for x in rows);cold=sum(x['cold_saving']for x in rows)
formation=2*32*(6144*384+11712*320+11712*320+4480*384+320*384)
result=dict(status='SOURCE_ONLY_ORIGINAL_CENTER_PADDED32',rows=rows,warm_saving_flops=warm,warm_saving_pct=100*warm/cert['flops'],after_alignment_warm_saving_pct=100*(warm-formation)/cert['flops'],cold_saving_pct=100*cold/cert['flops'],metric_alignment_formation_flops=formation,whole_bill=cert['flops'],source_sha256=p.SOURCE_SHA,models=0,limits=['Exact original center retained; projected32 rows followed by34 zero rows to next64. Kernel output padding discarded.','Paid empty/copy/fill packing included, outer-only randomized range/project/QR/lift and finalcopy included.','Same conservative frame-level metric alignment debit; random generation/finite/hosted overhead still omitted.','No estimator, numerical fidelity, or hosted runtime test.'])
out=D/'ORIGINAL_CENTER_PADDED32_TARIFF_v1.json';out.open('x').write(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ('rows','limits')}));print(hashlib.sha256(out.read_bytes()).hexdigest())
