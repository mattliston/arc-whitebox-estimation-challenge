from common import *
seal_check()
watch=read(HERE/'audit_watch.json');assert watch['whole_group_retired'] and watch['exit_code']==1 and watch['max_swap_growth_bytes']==0
assert 'Block() takes no arguments' in (HERE/'audit.log').read_text()
assert not (HERE/'audit_result.json').exists() and not list(HERE.glob('placement_*.json.gz'))
edits={
 'loader':[("lines[nodes[name].lineno-1:nodes[name].end_lineno]","lines[min([nodes[name].lineno]+[d.lineno for d in nodes[name].decorator_list])-1:nodes[name].end_lineno]")],
 'audit':[("from loader import environment","from loader_v2 import environment")],
}
inverses=[]
for name,changes in edits.items():
    before=(HERE/(name+'.py')).read_text();after=before
    for old,new in changes:assert after.count(old)==1;after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):assert inverse.count(new)==1;inverse=inverse.replace(new,old)
    assert inverse==before
    path=HERE/(name+'_v2.py')
    with path.open('x') as f:f.write(after)
    inverses.append(dict(original=pin(HERE/(name+'.py')),repaired=pin(path),edits=changes,inverse_exact=True))
write(HERE/'ast_decorator_repair.json',dict(reason='AST source extraction started at class Block, omitting its @dataclass decorator. The inherited old-price replay failed before any new normalized schedule was priced. Include decorators in source spans.',inverses=inverses,failed_guard=pin(HERE/'audit_watch.json'),failed_log=pin(HERE/'audit.log'),schedule_protocol_and_gates_unchanged=True))
write(HERE/'repair_sources_sealed.json',dict(pins=[pin(HERE/n) for n in ['admit_v2.py','loader_v2.py','audit_v2.py','ast_decorator_repair.json','run_v2.py']]))
print(dict(repair_admitted=True,inverses=len(inverses)))
