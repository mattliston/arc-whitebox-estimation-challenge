from common import *
seal_check()
guarded('unit',90,'unit.py')
guarded('disabled',120,'native.py','--out','disabled','--ranks','35','--disabled')
guarded('enabled',240,'native.py','--out','enabled','--ranks','35,100,35')
guarded('cold100',120,'native.py','--out','cold100','--ranks','100')
guarded('qualify',90,'qualify.py')
guarded('score',90,'score.py')
write(HERE/'completed.json',dict(completed=True,passed=read(HERE/'score.json')['passed'],
    stages=[pin(HERE/f'{n}_watch.json') for n in ['unit','disabled','enabled','cold100','qualify','score']]))
print(dict(completed=True,passed=read(HERE/'score.json')['passed']),flush=True)
