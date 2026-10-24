from common import *
seal_check()
guarded('build',60,'build.py');guarded('unit',60,'unit.py')
guarded('bank_triple',240,'native.py','--out','bank_triple','--ranks','35,100,35','--bank-parameters')
guarded('enabled',240,'native.py','--out','enabled','--ranks','36,38,36')
guarded('cold38',120,'native.py','--out','cold38','--ranks','38')
guarded('qualify',90,'qualify.py');guarded('score',90,'score.py')
write(HERE/'completed.json',dict(completed=True,passed=read(HERE/'score.json')['passed'],
    guarded_names=['build','unit','bank_triple','enabled','cold38','qualify','score']))
print(dict(completed=True,passed=read(HERE/'score.json')['passed']),flush=True)
