from common import *
verify()
guarded('unit',90,[RUNTIME,'-B','-W','ignore',HERE/'unit.py'])
guarded('qualification',240,[RUNTIME,'-B','-W','ignore',HERE/'capture.py','qualification','35,100,35'])
q=read(HERE/'qualification/result.json');rows=q['records'][:2];pins=[pin(HERE/'qualification/result.json')]
for rank in TRAIN[2:]+VALID:
    name=f'capture_{rank:04d}';guarded(name,120,[RUNTIME,'-B','-W','ignore',HERE/'capture.py',name,str(rank)])
    rr=read(HERE/name/'result.json');rows.extend(rr['records']);pins.append(pin(HERE/name/'result.json'))
pins.extend(r['capture'] for r in rows)
assert sum(p['bytes'] for p in pins)<2_000_000_000
write(HERE/'captures_sealed.json',dict(rows=rows,pins=pins,all_captures_before_targets=True,parent_sha256=PARENT_SHA,source_seal=pin(HERE/'sources_sealed.json')))
guarded('wide_qualification',180,[RUNTIME,'-B','-W','ignore',HERE/'qualify_wide.py'])
for rank in TRAIN+VALID:guarded(f'statistics_{rank:04d}',240,[RUNTIME,'-B','-W','ignore',HERE/'statistics.py',str(rank)])
write(HERE/'statistics_sealed.json',dict(pins=[pin(HERE/f'statistics_{r:04d}.{ext}') for r in TRAIN+VALID for ext in ('json','npz')],source_seal=pin(HERE/'sources_sealed.json')))
guarded('fit',60,[RUNTIME,'-B','-W','ignore',HERE/'fit.py'])
write(HERE/'completed.json',dict(completed=True,result=pin(HERE/'result.json'),passed=read(HERE/'result.json')['passed']))
