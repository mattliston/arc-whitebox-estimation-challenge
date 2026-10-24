from common import HERE,PARENT,RUNTIME,NATIVE,TRAIN,read,write,pin,seal_check,guarded
seal_check()
for label,ranks in [('parent_triple','35,100,35'),('parent_cold100','100')]:
    guarded(label,240 if ',' in ranks else 120,[RUNTIME,'-B','-W','ignore',NATIVE,PARENT,HERE/label,'--ranks',ranks])
for label,ranks in [('capture_triple','35,100,35'),('capture_cold100','100')]:
    guarded(label,240 if ',' in ranks else 120,[RUNTIME,'-B','-W','ignore',HERE/'capture.py',label,ranks])
guarded('qualification',60,[RUNTIME,'-B',HERE/'qualify_capture.py','qualification'])
assert read(HERE/'capture_qualification.json')['passed']
known=read(HERE/'inputs.json')['known_training_parents'];rows=[]
for rank in TRAIN:
    if str(rank) not in known:
        label=f'parent_{rank}';guarded(label,120,[RUNTIME,'-B','-W','ignore',NATIVE,PARENT,HERE/label,'--ranks',str(rank)])
        p=read(HERE/label/'result.json')['records'][0]
        known[str(rank)]=dict(output=pin(p['output']),flops=p['flops'],weights_sha256=p['weights_sha256'],receipt=pin(HERE/label/'result.json'))
    label=f'capture_{rank}';guarded(label,120,[RUNTIME,'-B','-W','ignore',HERE/'capture.py',label,str(rank)])
    c=read(HERE/label/'result.json')['records'][0]
    rows.append(dict(rank=rank,record=c,parent=known[str(rank)],watch=pin(HERE/f'{label}_watch.json'),receipt=pin(HERE/label/'result.json')))
write(HERE/'train_captures_completed.json',dict(rows=rows,truth_accessed=False,source_seal=pin(HERE/'capture_sources_sealed.json')))
guarded('all_capture_qualification',60,[RUNTIME,'-B',HERE/'qualify_capture.py','train'])
print(dict(all_train_captures_qualified=read(HERE/'captures_sealed.json')['passed'],train_networks=len(rows)),flush=True)
