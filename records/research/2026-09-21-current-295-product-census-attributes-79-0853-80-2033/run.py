from common import *

verify()
for rank in (35,100):guarded(f'census_{rank:04d}',120,'census.py',rank)
rows=[]
for rank in (35,100):
    p=HERE/f'rank{rank:04d}/attribution.json';r=read(p)
    assert all(c['input_dtypes']==['float32','float32'] for c in r['outermost_mm_calls'])
    rows.append(dict(rank=rank,expected_total_flops=r['total_flops'],
                     expected_outermost_mm_calls=len(r['outermost_mm_calls']),
                     expected_empty_calls=sum(min(*c['left'],*c['right'])==0 for c in r['outermost_mm_calls']),**pin(p)))
catalog=read(D/'lazy_mixed_fmm_reprice_v3/inputs.json')
inputs=dict(censuses=rows,smirnov_certificate=catalog['smirnov_certificate'],catalog_certificate=catalog['catalog_certificate'],official_adjusted=read(D/'official_best_331684.json')['adjusted'])
write(HERE/'reprice_inputs.json',inputs)
extra=[pin(HERE/'reprice_inputs.json')]+rows+[inputs['smirnov_certificate'],inputs['catalog_certificate']]
write(HERE/'reprice_inputs_sealed.json',dict(pins=read(HERE/'sources_sealed.json')['pins']+extra,targets_opened=False))
guarded('reprice',120,'reprice.py');guarded('verify_paths',90,'verify_paths.py')
write(HERE/'completed.json',dict(completed=True,result=pin(HERE/'reprice_result.json'),paths=pin(HERE/'paths_verified.json'),new_candidates=0,targets_opened=False,goal_achieved=False))
