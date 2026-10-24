from datetime import datetime,timezone
import os
import time
from endpoint_common import *


def main():
    endpoint_verify()
    assert not (HERE/'run_claim.json').exists()
    qdir = D/'current_p_native_allocator_qualification_v1'
    qualification = read(qdir/'controller_result.json')
    assert qualification['completed'] and qualification['passed'] and qualification['completed_calls'] == 8
    check(qualification['qualification'])
    groups = {96361,99027,34871,48700,61621}
    for name in ['native_baseline','qualify_baseline','native_relief','qualify_comparison']:
        w = read(qdir/f'{name}_watch.json')
        assert w['completed'] and w['exit_code'] == 0 and w['whole_group_retired'] and not w['remaining_rows']
        assert w['max_swap_growth_bytes'] == 0
        groups.add(w['pid'])
    rows = []
    for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True,timeout=3).splitlines():
        a = line.strip().split(None,3)
        if len(a) == 4:
            rows.append(dict(pid=int(a[0]),pgid=int(a[1]),rss_bytes=int(a[2])*1024,command=a[3]))
    assert not [r for r in rows if r['pid'] in groups or r['pgid'] in groups]
    assert not [r for r in rows if r['command'].split(None,1)[0] in {str(RUNTIME),str(NATIVE_RUNTIME)}]
    write(HERE/'run_claim.json',dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),pgid=os.getpgrp(),
        source_seal=pin(HERE/'sources_sealed.json'),candidate=read(HERE/'build.json')['candidate'],
        previous_groups_retired=sorted(groups),planned_new_calls=41,reused_qualified_calls=44,
        no_other_numerical_worker=True))
    started = time.monotonic()
    result = dict(completed=False,scientific_passed=None,error=None,fit_or_candidate_changed=False,
        no_automatic_downstream=True,no_external_actions=True,reused_qualified_calls=44)
    try:
        for name,seconds,script,args in [
            ('native_trained',2400,'native.py',['trained']),
            ('qualify_trained_lifecycle',180,'qualify_trained.py',[]),
            ('seal_predictions',180,'seal_with_lifecycle.py',[]),
            ('score_native40',180,'score_with_lifecycle.py',[])]:
            assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) <= 100_000_000
            guarded(name,seconds,NATIVE_RUNTIME,script,*args)
        score = read(HERE/'native40_score.json')
        assert score['completed'] and score['integrity_passed']
        write(HERE/'endpoint_complete.json',dict(completed=True,passed=score['passed'],native_records=85,
            new_native_calls=41,reused_qualified_calls=44,score=pin(HERE/'native40_score.json'),
            no_automatic_downstream_launch=True,official_changed=False,original_fit_and_candidate_unchanged=True,
            closed_context_lifecycle_qualified=True,all_new_retirements_qualified=True,historical_trained_prefix_exact_calls=30))
        result.update(completed=True,scientific_passed=score['passed'],endpoint=pin(HERE/'endpoint_complete.json'))
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        result['wall_s'] = time.monotonic()-started
        result['completed_new_native_calls'] = len(list((HERE/'native_trained').glob('call_[0-9][0-9].json')))
        write(HERE/'controller_result.json',result)
        print(result,flush=True)
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
