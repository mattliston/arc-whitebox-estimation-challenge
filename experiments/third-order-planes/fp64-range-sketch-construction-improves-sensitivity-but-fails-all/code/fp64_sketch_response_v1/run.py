from common import HERE,RUNTIME,NATIVE,read,write,pin,guarded,seal_check

seal_check();build=read(HERE/'build.json')
def source(coordinate,label):
    return next(x['candidate']['path'] for x in build['variants'] if x['coordinate']==coordinate and x['label']==label)
def call(rank,coordinate,label,repeat=False):
    name=f'r{rank:04d}_h{coordinate if coordinate is not None else "base"}_{label}'+('_repeat' if repeat else '')
    guarded(name,120,[RUNTIME,'-B','-W','ignore',NATIVE,source(coordinate,label),HERE/name,'--ranks',str(rank)])
    result=read(HERE/name/'result.json');assert len(result['records'])==1
    return dict(rank=rank,coordinate=coordinate,label=label,repeat=repeat,receipt=pin(HERE/name/'result.json'),call=result['records'][0])
guarded('unit',120,[RUNTIME,'-B','-W','ignore',HERE/'unit.py'])
rows=[]
for rank in [36,38]:
    row=call(rank,None,'base');rows.append(row)
    control=next(x for x in build['controls'] if x['rank']==rank and x['label']=='base')
    if row['call']['flops']/control['flops']>1.20:
        write(HERE/'outcome.json',dict(passed=False,stage='base_cost',rows=rows,closed=True,targets_opened=False))
        raise SystemExit(0)
for rank in [36,38]:
    for c in [0,8]:
        for label in ['plus','minus','double']:rows.append(call(rank,c,label))
rows.append(call(36,0,'plus',True))
write(HERE/'predictions_sealed.json',dict(rows=rows,targets_opened=False,source_seal=pin(HERE/'sources_sealed.json')))
guarded('qualify',90,[RUNTIME,'-B',HERE/'qualify.py'])
write(HERE/'completed.json',dict(completed=True,outcome=read(HERE/'outcome.json')))
