from common import *
seal_check();guarded('unit',120,'unit.py')
guarded('disabled',300,'native.py','--out','disabled','--ranks','35,100,35','--disabled')
guarded('enabled',300,'native.py','--out','enabled','--ranks','36,38,36')
guarded('cold38',300,'native.py','--out','cold38','--ranks','38')
guarded('qualify',90,'qualify.py');guarded('score',60,'score.py')
write(HERE/'completed.json',dict(completed=True,guards=['unit','disabled','enabled','cold38','qualify','score'],
    full_native_calls=7,passed=read(HERE/'score.json')['passed']))
print(dict(completed=True,passed=read(HERE/'score.json')['passed']),flush=True)
