import datetime
from common import HERE,read,write,pin,seal_check
seal_check()
files=[HERE/n for n in ['PROTOCOL.md','common.py','observer_math.py','unit.py','design.py','fit.py','prepare_fit.py','capture_sources_sealed.json','inputs.json']]
write(HERE/'fit_sources_sealed.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pins=[pin(p) for p in files],target_arrays_decoded=False,fit_not_started=True))
print(dict(fit_source_pins=len(files)),flush=True)
