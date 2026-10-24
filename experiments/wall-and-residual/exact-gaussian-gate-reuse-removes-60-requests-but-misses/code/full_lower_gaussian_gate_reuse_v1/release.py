from common import HERE, CLI, read, write, pin, guarded, seal_check

seal_check('release_sources_sealed.json')
assert read(HERE/'native_qualified.json')['passed'] and read(HERE/'boundary_qualified.json')['passed']
g = read(HERE/'boundary_controller_watch.json')
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
guarded('contract',60,[CLI.with_name('whest'),'validate','--estimator',HERE/'candidate.py','--json'])
guarded('package',60,[CLI,'-B',HERE/'prepare_package.py'])
write(HERE/'release_completed.json',dict(completed=True,package=pin(HERE/'package.json'),
    native=pin(HERE/'native_qualified.json'),boundary=pin(HERE/'boundary_qualified.json'),uploaded=False))
