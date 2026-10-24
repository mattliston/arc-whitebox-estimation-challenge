from datetime import datetime, timezone
import os
import time
from common import *

admission()
assert not (HERE / 'run_claim.json').exists() and not (HERE / 'result.json').exists()
rows = subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,command='],text=True,timeout=5)
numeric = {str(RUNTIME),str(NATIVE_RUNTIME),str(P2/'data_gate/.client-venv/bin/python')}
active = [line for line in rows.splitlines() if len(s:=line.strip().split(None,2))==3 and s[2].split()[0] in numeric]
assert not active, active
write(HERE / 'run_claim.json',dict(pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.now(timezone.utc).isoformat(),
    sources=pin(HERE/'sources_sealed.json'),candidate=pin(SOURCE),maximum_native_calls=4,
    maximum_full_torch_calls=6,no_automatic_fit=True,no_upload=True))
start = time.monotonic()
result = dict(completed=False,passed=False,error=None,no_retry=True,training_started=False,uploaded=False)
try:
    guarded('unit_native',180,NATIVE_RUNTIME,'unit_native.py')
    guarded('unit_torch',180,RUNTIME,'unit_torch.py')
    guarded('independent_trained',480,NATIVE_RUNTIME,'native.py','independent','trained')
    guarded('confirmation_100',240,NATIVE_RUNTIME,'native.py','confirmation','trained','--rank','100')
    guarded('qualify_native',600,NATIVE_RUNTIME,'qualify_native.py')
    assert read(HERE/'native_qualification.json')['passed']
    guarded('qualify_torch',1800,RUNTIME,'qualify_torch.py')
    assert read(HERE/'torch_qualification.json')['passed']
    result.update(completed=True,passed=True,native=pin(HERE/'native_qualification.json'),
        torch=pin(HERE/'torch_qualification.json'))
except BaseException as error:
    result['error'] = repr(error)
finally:
    result.update(wall_s=time.monotonic()-start,sources=pin(HERE/'sources_sealed.json'))
    write(HERE/'result.json',result)
    print(result,flush=True)
raise SystemExit(0 if result['passed'] else 1)
