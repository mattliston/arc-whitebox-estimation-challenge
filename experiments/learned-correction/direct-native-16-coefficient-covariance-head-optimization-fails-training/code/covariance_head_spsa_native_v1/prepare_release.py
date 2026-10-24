from common import HERE,D,read,write,pin,verify,seal_check
seal_check();assert read(HERE/'outcome.json')['passed']
for stage in ['train','check','panel','confirmation']:assert read(HERE/f'{stage}_score.json')['passed']
assert read(HERE/'controller_watch.json')['whole_group_retired']
R=D/'readout_residual_runtime_v1'
files=[HERE/n for n in ['RELEASE_PROTOCOL.md','prepare_release.py','run_release.py','qualify_release.py','package.py',
    'common.py','candidate.py','fit.json','outcome.json','check_repeat_qualified.json','sources_sealed.json']]
files += [R/n for n in ['plain_driver.py','plain_client.py','predict_only_probe_v1/driver.py','predict_only_probe_v1/client.py']]
files += [HERE/f'{s}_score.json' for s in ['train','check','panel','confirmation']]
write(HERE/'release_sources_sealed.json',dict(pins=[pin(p) for p in files],release_not_started=True))
print(dict(release_source_pins=len(files)),flush=True)
