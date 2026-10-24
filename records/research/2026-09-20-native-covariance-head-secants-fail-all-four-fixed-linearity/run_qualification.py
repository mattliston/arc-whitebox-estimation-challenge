from common import HERE,D,RUNTIME,read,write,pin,verify,seal_check,guarded
seal_check();guarded('build',60,[RUNTIME,'-B',HERE/'build.py']);guarded('unit',60,[RUNTIME,'-B',HERE/'unit.py'])
assert read(HERE/'unit.json')['passed'];build=read(HERE/'build.json');rows=[]
roster=[(rank,j,label,False) for rank in [36,38] for j in [0,8] for label in ['plus','minus','double']]+[(36,0,'plus',True)]
for rank,j,label,repeat in roster:
    variant=next(x for x in build['variants'] if x['coordinate']==j and x['label']==label);verify(variant['source'])
    name=f'r{rank:04d}_h{j:02d}_{label}'+('_repeat' if repeat else '');out=HERE/name
    guarded(name,120,[RUNTIME,'-B','-W','ignore',D/'weight5_reuse_v1/native.py',variant['source']['path'],out,'--ranks',str(rank)])
    receipt=read(out/'result.json');assert receipt['candidate_sha256']==variant['source']['sha256'] and not receipt['truth_accessed']
    call=receipt['records'][0];assert call['rank']==rank and call['includes_initialization'] and call['weight5_cache']['numeric_cache_retired']
    rows.append(dict(rank=rank,coordinate=j,label=label,repeat=repeat,source=variant['source'],receipt=pin(out/'result.json'),watch=pin(HERE/f'{name}_watch.json'),call=call))
    print(dict(rank=rank,coordinate=j,label=label,repeat=repeat,flops=call['flops'],wall_s=call['wall_s']),flush=True)
write(HERE/'qualification_predictions_sealed.json',dict(rows=rows,parent=build['parent'],sources=pin(HERE/'sources_sealed.json'),truth_arrays_accessed=False))
guarded('qualify_response',60,[RUNTIME,'-B',HERE/'qualify_response.py'])
q=read(HERE/'response_qualified.json');print(dict(passed=q['passed'],cells=q['cells'],repeat_exact=q['repeat_exact']),flush=True)
