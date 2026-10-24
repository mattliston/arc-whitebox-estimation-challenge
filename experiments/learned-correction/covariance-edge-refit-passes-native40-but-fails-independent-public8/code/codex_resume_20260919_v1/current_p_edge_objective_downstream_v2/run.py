"""One terminal candidate, one sequential downstream run, no retry or upload."""
from datetime import datetime,timezone
import os
import time
from common import *


def main():
    verify()
    assert not (HERE/'run_claim.json').exists() and not (HERE/'admission.json').exists()
    prior_workers_retired()
    admit()
    write(HERE/'run_claim.json',dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),pgid=os.getpgrp(),
        admission=pin(HERE/'admission.json'),no_other_numerical_worker=True,maximum_new_calls=55))
    started = time.monotonic()
    result = dict(completed=False,passed=False,completed_stages=[],error=None,official_changed=False,
        no_automatic_release_or_upload=True,no_retry=True)
    try:
        guarded('control',480,'native.py','control','parent')
        guarded('qualify_control',180,'qualify_control.py')
        for stage,ranks in [('panel',PANEL),('confirmation',CONFIRM),('independent',INDEPENDENT)]:
            if stage=='independent':
                for role in ['parent','trained']:
                    guarded(f'independent_{role}',2400,'native.py',stage,role)
            else:
                for rank in ranks:
                    guarded(f'{stage}_{rank}',240,'native.py',stage,'trained','--rank',rank)
            guarded(f'seal_{stage}',180,'seal_stage.py',stage)
            guarded(f'score_{stage}',180,'score_stage.py',stage)
            score = read(HERE/f'{stage}_score.json')
            assert score['completed'] and score['integrity_passed']
            result['completed_stages'].append(dict(stage=stage,passed=score['passed'],score=pin(HERE/f'{stage}_score.json')))
            if not score['passed']:
                result.update(completed=True,scientific_stop_stage=stage)
                break
        else:
            result.update(completed=True,passed=True)
    except BaseException as error:
        result['error']=repr(error)
    finally:
        result['wall_s']=time.monotonic()-started
        result['admission']=pin(HERE/'admission.json')
        write(HERE/'result.json',result)
        print(result,flush=True)
    return 0 if result['completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
