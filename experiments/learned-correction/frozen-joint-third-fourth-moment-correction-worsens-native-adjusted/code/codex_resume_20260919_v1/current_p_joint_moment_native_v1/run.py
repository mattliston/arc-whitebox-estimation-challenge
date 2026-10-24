"""Single owner, fixed nine-call experiment. Never launches further work."""
import time
from common import *

verify()
write(HERE/'run_claim.json',dict(launched=True,protocol=pin(HERE/'PROTOCOL.md'),sources=pin(HERE/'sources_sealed.json')))
started=time.monotonic()
result=dict(completed=False,passed=False,error=None,no_automatic_downstream=True,no_upload=True)
try:
    guarded('build',180,'build.py')
    for role in ('parent','null','trained'):
        guarded('native_'+role,480,'native.py','development',role)
    guarded('qualify',180,'qualify.py')
    guarded('score',180,'score.py')
    s=read(HERE/'score.json')
    result.update(completed=True,passed=s['passed'],score=pin(HERE/'score.json'))
except BaseException as error:
    result['error']=repr(error)
    raise
finally:
    result['wall_s']=time.monotonic()-started
    result['source_seal']=pin(HERE/'sources_sealed.json')
    write(HERE/'controller_result.json',result)
    print(result,flush=True)
