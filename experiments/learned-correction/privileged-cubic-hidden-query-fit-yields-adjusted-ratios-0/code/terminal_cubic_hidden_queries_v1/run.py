from common import *
verify();guarded('unit',90,[RUNTIME,'-B','-W','ignore',HERE/'unit.py'])
guarded('capture',240,[RUNTIME,'-B','-W','ignore',HERE/'capture.py'])
rr=read(HERE/'capture/result.json');pins=[pin(HERE/'capture/result.json')]+[r['capture'] for r in rr['records'][:2]]
assert sum(r['bytes'] for r in pins)<=300_000_000
write(HERE/'captures_sealed.json',dict(pins=pins,targets_opened=False))
guarded('features',120,[RUNTIME,'-B','-W','ignore',HERE/'features.py'])
f=read(HERE/'features_result.json');write(HERE/'features_sealed.json',dict(pins=[pin(HERE/'features_result.json')]+[r['file'] for r in f['rows']],targets_opened=False))
guarded('oracle',60,[RUNTIME,'-B','-W','ignore',HERE/'oracle.py'])
write(HERE/'completed.json',dict(completed=True,passed=read(HERE/'result.json')['passed'],result=pin(HERE/'result.json')))
