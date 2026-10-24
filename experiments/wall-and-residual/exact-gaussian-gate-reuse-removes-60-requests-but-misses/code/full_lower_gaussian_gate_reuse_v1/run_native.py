from common import HERE, RUNTIME, NATIVE, guarded, seal_check, read, write

seal_check()
guarded('unit',90,[RUNTIME,'-B','-W','ignore',HERE/'unit.py'])
assert read(HERE/'unit.json')['passed']
guarded('disabled35',120,[RUNTIME,'-B','-W','ignore',NATIVE,HERE/'disabled.py',HERE/'disabled35','--ranks','35'])
guarded('enabled_triple',180,[RUNTIME,'-B','-W','ignore',NATIVE,HERE/'candidate.py',HERE/'enabled_triple','--ranks','35,100,35'])
guarded('enabled_cold100',120,[RUNTIME,'-B','-W','ignore',NATIVE,HERE/'candidate.py',HERE/'enabled_cold100','--ranks','100'])
guarded('native_qualification',60,[RUNTIME,'-B',HERE/'native_qualification.py'])
write(HERE/'native_completed.json',dict(completed=True,passed=read(HERE/'native_qualified.json')['passed']))
