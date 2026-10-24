import time
from common import *
verify()
write(HERE/'run_claim.json',dict(started=True,sources=pin(HERE/'sources_sealed.json'),protocol=pin(HERE/'PROTOCOL.md')))
started=time.monotonic();result=dict(completed=False,passed=False,error=None,no_upload=True,no_automatic_downstream=True)
try:
    guarded('qualification',480,'native.py','capture')
    guarded('qualify',180,'qualify.py')
    for rank in TRAIN[1:]:
        if rank in PARENT_NEW:guarded(f'parent_{rank}',180,'native.py','parent','--rank',rank)
        guarded(f'capture_{rank}',180,'native.py','capture','--rank',rank)
    guarded('seal',180,'seal.py')
    guarded('information',180,'information.py')
    score=read(HERE/'information_result.json')
    result.update(completed=True,passed=score['passed'],score=pin(HERE/'information_result.json'))
except BaseException as error:
    result['error']=repr(error)
    raise
finally:
    result['wall_s']=time.monotonic()-started;result['sources']=pin(HERE/'sources_sealed.json')
    write(HERE/'controller_result.json',result)
    print(result,flush=True)
