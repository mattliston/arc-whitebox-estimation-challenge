from records import *
admission()
rows, pins = verify_run('control','control','parent',[35,100,35])
reference_path = FIT/'native_null/result.json'
reference = read(reference_path)
assert reference['completed'] and [r['rank'] for r in reference['records']] == [35,100,35]
assert read(FIT/'native_null_qualification.json')['passed']
clean_watch(FIT/'native_null_watch.json')
for actual,old in zip(rows,reference['records']):
    for key in ['flops','domain','selectors','includes_initialization','weights']:
        assert actual[key] == old[key], (actual['rank'],key)
    check(old['prediction'])
    assert actual['prediction']['sha256'] == old['prediction']['sha256']
    pins.extend([old['prediction'],old['bill']])
pins.extend([pin(reference_path),pin(FIT/'native_null_watch.json'),pin(FIT/'native_null_qualification.json')])
write(HERE/'control_qualification.json',dict(passed=True,calls=3,
    all16_output_bytes_and_complete_flop_counts_exact=True,all16_domains_and45_selectors_exact=True,
    input_copy_and_closed_context_reset_changed_participant_flops=False,truth_accessed=False,
    pins=list({p['path']:p for p in pins}.values())))
print(dict(qualified=True,calls=3,targets_accessed=False),flush=True)
