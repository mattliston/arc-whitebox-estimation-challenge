from common import *
verify();assert read(HERE/'qualification.json')['passed']
rows=[];pins=[]
for rank in TRAIN:
    name='qualification' if rank==35 else f'capture_{rank}'
    clean_watch(name)
    result=read(HERE/name/'result.json')
    row=result['records'][0]
    assert row['rank']==rank and row['includes_initialization']
    assert result['source']==pin(HERE/'capture_candidate.py')
    base=baseline(rank)
    check(base['source']);check(base['receipt'])
    assert base['source']['sha256']==pin(SOURCE)['sha256']
    assert row['prediction']['sha256']==base['record']['prediction']['sha256']
    assert row['domain']==base['record']['domain'] and row['selectors']==base['record']['selectors']
    assert row['flops']>base['record']['flops']
    for r in (row,base['record']):
        for key in ('prediction','bill','weights','input_bill','input_reset','lifecycle'):
            check(r[key]);pins.append(r[key])
    check(row['packet']);pins.append(row['packet'])
    retirement=read(HERE/name/'call_00_retirement.json')
    assert retirement['all_tracked_call_objects_released'] and retirement['tracked_objects']==26 and not retirement['live_objects']
    rows.append(dict(rank=rank,capture=row,baseline=base,capture_result=pin(HERE/name/'result.json')))
write(HERE/'captures_sealed.json',dict(completed=True,rows=rows,pins=pins,targets_decoded=False,
    all_positive_predictions_equal_exact_parent=True,own_complete_bills=True,train=TRAIN,
    reused_research_networks=True,parent=pin(SOURCE),capture_source=pin(HERE/'capture_candidate.py')))
print(dict(sealed=True,rows=len(rows),pins=len(pins),targets_decoded=False),flush=True)
