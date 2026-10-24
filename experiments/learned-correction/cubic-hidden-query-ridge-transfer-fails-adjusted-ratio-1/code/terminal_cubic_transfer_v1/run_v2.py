from common import *
seal_check()
for repair_pin in read(HERE/'repair_sources_sealed.json')['pins']:check(repair_pin)
done('unit')
guarded('solver_unit_v2',90,'solver_unit_v2.py')
for name,rank in [('qual35a',35),('qual100',100),('qual35b',35)]+[(f'train{r}',r) for r in read(HERE/'inputs.json')['train']]:
    guarded(name,150,'capture_v2.py','--rank',rank,'--name',name)
guarded('qualify',90,'qualify_v2.py')
guarded('fit',120,'fit.py')
guarded('score',90,'score.py')
write(HERE/'completed.json',dict(completed=True,passed=read(HERE/'cv_result.json')['passed'],new_full_native_calls=23))
print(read(HERE/'completed.json'),flush=True)
