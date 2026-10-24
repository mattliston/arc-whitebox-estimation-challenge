"""One fixed full-native secant design; no target-array access."""
from common import HERE,D,PARENT,RUNTIME,TRAIN,read,write,pin,verify,seal_check,guarded,sha
seal_check()
for p in read(HERE/'acquisition_sources_sealed.json')['pins']:verify(p)
assert read(HERE/'response_qualified.json')['passed']
build=read(HERE/'build.json');qual=read(HERE/'qualification_predictions_sealed.json');inputs=read(HERE/'inputs.json');rows=[]
def native(name,source,rank):
    out=HERE/name
    guarded(name,120,[RUNTIME,'-B','-W','ignore',D/'weight5_reuse_v1/native.py',source,out,'--ranks',str(rank)])
    receipt=read(out/'result.json');assert receipt['candidate_sha256']==sha(source) and not receipt['truth_accessed']
    call=receipt['records'][0];assert call['rank']==rank and call['includes_initialization'] and call['weight5_cache']['numeric_cache_retired']
    print(dict(native=name,rank=rank,flops=call['flops'],wall_s=call['wall_s']),flush=True)
    return dict(source=pin(source),receipt=pin(out/'result.json'),watch=pin(HERE/f'{name}_watch.json'),call=call)
for rank in TRAIN:
    parent=next(x for x in inputs['training_parent'] if x['rank']==rank)
    if parent['fresh_packed_parent_call_required']:
        baseline=native(f'base_{rank:04d}',PARENT,rank)
    else:
        verify(parent['receipt']);r=read(parent['receipt']['path']);baseline=dict(source=parent['source'],receipt=parent['receipt'],call=r['records'][0],reused=True)
    derivatives=[]
    for j in range(16):
        variant=next(x for x in build['variants'] if x['coordinate']==j and x['label']=='plus');verify(variant['source'])
        reuse=next((x for x in qual['rows'] if x['rank']==rank and x['coordinate']==j and x['label']=='plus' and not x['repeat']),None)
        if reuse is not None:
            verify(reuse['receipt']);verify(reuse['source']);assert reuse['source']==variant['source']
            record=dict(source=reuse['source'],receipt=reuse['receipt'],watch=reuse['watch'],call=reuse['call'],reused=True)
        else:record=native(f'train_{rank:04d}_h{j:02d}',variant['source']['path'],rank)
        derivatives.append(dict(coordinate=j,epsilon=variant['epsilon'],actual_delta=variant['actual_delta'],**record))
    row=dict(rank=rank,parent=baseline,saved_parent=parent,secants=derivatives)
    write(HERE/f'acquired_rank{rank:04d}.json',row);rows.append(row)
    print(dict(rank=rank,all16_secants_acquired=True,targets_opened=False),flush=True)
write(HERE/'acquisition_completed.json',dict(rows=rows,sources=pin(HERE/'acquisition_sources_sealed.json'),truth_arrays_accessed=False))
guarded('qualify_acquisition',60,[RUNTIME,'-B',HERE/'qualify_acquisition.py'])
