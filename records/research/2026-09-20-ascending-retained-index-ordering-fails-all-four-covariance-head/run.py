from common import HERE,RUNTIME,read,write,pin,guarded,seal_check

seal_check();build=read(HERE/'build.json')
def variant(coordinate,label,kind):
    return next(r[kind]['path'] for r in build['variants'] if r['coordinate']==coordinate and r['label']==label)
guarded('unit',90,[RUNTIME,'-B','-W','ignore',HERE/'unit.py'])
for coordinate,label in [(None,'base'),(0,'plus'),(0,'minus'),(0,'double')]:
    name='original36_'+label
    guarded(name,120,[RUNTIME,'-B','-W','ignore',HERE/'capture.py',variant(coordinate,label,'original'),name,36])
guarded('qualify_capture',60,[RUNTIME,'-B',HERE/'qualify.py','capture'])
rows=[]
for rank in [36,38]:
    name=f'canonical{rank}_base'
    guarded(name,120,[RUNTIME,'-B','-W','ignore',HERE/'capture.py',variant(None,'base','canonical'),name,rank])
    rows.append(dict(rank=rank,coordinate=None,label='base',repeat=False,receipt=pin(HERE/name/'result.json')))
for rank in [36,38]:
    for coordinate in [0,8]:
        for label in ['plus','minus','double']:
            name=f'canonical{rank}_h{coordinate:02d}_{label}'
            guarded(name,120,[RUNTIME,'-B','-W','ignore',HERE/'capture.py',variant(coordinate,label,'canonical'),name,rank])
            rows.append(dict(rank=rank,coordinate=coordinate,label=label,repeat=False,receipt=pin(HERE/name/'result.json')))
name='canonical36_h00_plus_repeat'
guarded(name,120,[RUNTIME,'-B','-W','ignore',HERE/'capture.py',variant(0,'plus','canonical'),name,36])
rows.append(dict(rank=36,coordinate=0,label='plus',repeat=True,receipt=pin(HERE/name/'result.json')))
write(HERE/'predictions_sealed.json',dict(rows=rows,targets_accessed=False,source_seal=pin(HERE/'sources_sealed.json')))
guarded('response_qualification',90,[RUNTIME,'-B',HERE/'qualify.py','response'])
write(HERE/'completed.json',dict(completed=True,outcome=read(HERE/'outcome.json')))
