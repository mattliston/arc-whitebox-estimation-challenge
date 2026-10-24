from common import *
seal_check()
guarded('build',90,'build.py');guarded('unit',90,'unit.py')
guarded('bank_triple',240,'native.py','--out','bank_triple','--ranks','35,100,35','--bank')
guarded('candidate_triple',240,'native.py','--out','candidate_triple','--ranks','35,100,35')
guarded('cold100',150,'native.py','--out','cold100','--ranks','100')
guarded('qualify',90,'qualify.py')
for rank in CONFIRM:guarded(f'confirmation_{rank}',150,'native.py','--out',f'confirmation_{rank}','--ranks',str(rank))
guarded('seal_predictions',90,'qualify.py','--confirmation');guarded('score',90,'score.py')
write(HERE/'completed.json',dict(completed=True,passed=read(HERE/'score.json')['passed'],full_native_calls=15,new_fit_count=0,
    guarded_names=['build','unit','bank_triple','candidate_triple','cold100','qualify']+[f'confirmation_{r}' for r in CONFIRM]+['seal_predictions','score']))
print(dict(completed=True,passed=read(HERE/'score.json')['passed']),flush=True)
