from common import HERE, PARENT, RUNTIME, PROBE, read, write, pin, guarded, seal_check

seal_check('boundary_sources_sealed.json')
assert read(HERE/'native_qualified.json')['passed']
g = read(HERE/'native_controller_watch.json')
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes'] == 0
rows = []
for index, rank in enumerate([35,100,37,39,42]):
    roles = ['parent','candidate'] if index % 2 == 0 else ['candidate','parent']
    for role in roles:
        label = f'pair{index:02d}_{rank}_{role}'
        source = PARENT if role == 'parent' else HERE/'candidate.py'
        out = HERE/label
        guarded(label,120,[RUNTIME,'-B',PROBE/'driver.py',source,out,str(rank)])
        receipt = read(out/'completed.json')
        assert receipt['completed'] and receipt['numpy_available'] is False
        call = receipt['records'][0]
        rows.append(dict(pair=index,rank=rank,role=role,receipt=pin(out/'completed.json')))
        print(dict(pair=index,rank=rank,role=role,residual_s=call['summary']['residual_wall_time_s'],
                   wall_s=call['wall_s']),flush=True)
write(HERE/'pairs_completed.json',dict(completed=True,rows=rows,source_seal=pin(HERE/'boundary_sources_sealed.json')))
guarded('pairs_qualification',60,[RUNTIME,'-B',HERE/'qualify_boundary.py','pairs'])
if read(HERE/'pairs_qualified.json')['passed']:
    guarded('client_triple',240,[RUNTIME,'-B',PROBE/'driver.py',HERE/'candidate.py',HERE/'client_triple','35,100,35'])
    guarded('warm_qualification',60,[RUNTIME,'-B',HERE/'qualify_boundary.py','warm'])
write(HERE/'boundary_completed.json',dict(completed=True,outcome=read(HERE/'warm_outcome.json')
    if (HERE/'warm_outcome.json').exists() else read(HERE/'outcome.json')))
