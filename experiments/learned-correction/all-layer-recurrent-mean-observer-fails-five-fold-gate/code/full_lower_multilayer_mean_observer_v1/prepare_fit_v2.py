import datetime
from common import HERE,write,pin,seal_check
seal_check();seal_check('fit_sources_sealed.json')
files=[HERE/n for n in ['FIT_IMPLEMENTATION_REPAIR_V2.md','observer_math_v2.py','unit_v2.py','fit_v2.py','prepare_fit_v2.py','fit_sources_sealed.json','observer_math.py','common.py','PROTOCOL.md']]
write(HERE/'fit_sources_sealed_v2.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pins=[pin(p) for p in files],target_arrays_decoded=False,fit_not_started=True,model_and_gates_unchanged=True))
print(dict(fit_v2_source_pins=len(files)),flush=True)
