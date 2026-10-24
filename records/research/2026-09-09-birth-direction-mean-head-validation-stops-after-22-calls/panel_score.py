"""Fixed original independent4M NC arithmetic; separate from native acquisition.

Integrity completion is not scientific retention. BOTH preset groups must win.
No candidate import, fitting, target selection or startup subtraction.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from types import SimpleNamespace
import panel_common as c


def verify_acquisition(args,context):
    pins = context['pins']
    c.verify_watch(args,context,pins)
    record,end = c.sealed(c.OUTPUT,args.acquisition_sha256,pins)
    outputs = c.verify_native(record,end,args.acquisition_sha256,args,context,pins)
    c.bind(pins)
    return record,outputs


def score_function(h):
    original = c.function_source(c.TL/'panel.py','score')
    # Identical operations and grouping: labels, comparator role, and NODE-owned
    # startup reporting are the only changes. No derivative/surrogate objective.
    labels = original.replace("'current_d50'","'P1'")
    c.require(labels.replace("'P1'","'current_d50'") == original, 'Comparator-label inverse')
    edits = [
        ("context['control_score']['per_role']['candidate'][call]", "context['control_score']['per_role']['terminal'][call]"),
        ("context['control_score']['summaries']['candidate']", "context['control_score']['summaries']['terminal']"),
        ('        earlier15_rows_byte_exact=True,own_complete_F_and_opcounts_exact=True,\n',
         '        layer0_byte_exact=True,own_complete_bills=True,all16_domains_qualified=True,all47_selectors_qualified=True,\n'),
        ('        initialization_flops=INITIALIZATION,repeat_excluded=True,\n',
         "        initialization_flops=acquisition['initialization_flops'],repeat_excluded=True,\n"),
    ]
    source = c.splice(labels,edits)
    namespace = dict(globals(),require=c.require,sha=h.digest,read=c.read,
        CANDIDATE_SHA=c.CANDIDATE_SHA,SCORE_SHA=c.SCORE_SHA,CONTROL_SHA=c.CONTROL_SHA,
        CONTROL_SCORE_SHA=c.CONTROL_SCORE_SHA)
    exec(compile(source,str(__file__)+':unchanged_original_NC_arithmetic','exec'),namespace)
    return namespace['score'],dict(score_arithmetic_inverse_verified=True,
        original_score_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_score_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        changes=['comparator label P1 and archived terminal score-role key',
                 'report actual NODE startup and first-layer/own-cost qualification, not TL-only invariants'])


def main():
    args = c.arguments('scorer')
    if args is None: return 0
    began = time.monotonic(); c.environment()
    c.require(not c.SCORE_OUT.exists() and not c.ROOT.is_symlink(), 'Exclusive independent14 score')
    c.SCORE_OUT.mkdir(exist_ok=False)
    result = dict(schema=c.SCORE_SCHEMA,mode='score',passed=False,completed=False,qualification_passed=False,
        integrity_passed=False,hashes_unchanged=False,sources_unchanged=False,requested_seals=vars(args),
        terminal_score_sha256=c.SCORE_SHA,truth_files_accessed=False,no_refit=True,
        estimator_executed=False,native_gradient_claim=False,old_NODE_pilot_still_failed=True,
        production_qualified=False,official_score=False,milestone_achieved=False,retain_local_component=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    def guard(*unused):
        c.require(time.monotonic()-began < 120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
                  'Whole offline score120s/2GB including admission and sealing')
    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        context = c.read_context(args,args.context_sha256,guard); c.retired_admission(args,context,guard)
        pins = {**context['source_bindings'],**context['input_bindings']}
        c.verify_watch(args,context,pins,guard)
        control,_ = c.sealed(c.CONTROL,c.CONTROL_SHA,pins,c.CONTROL_END_SHA,tick=guard)
        control_score,_ = c.sealed(c.CONTROL_SCORE,c.CONTROL_SCORE_SHA,pins,tick=guard)
        h,p,v = c.panel_modules()
        control_outputs = c.five_artifact_verifier(context,c.P1_CANDIDATE,c.P1_SHA,1383913786)(
            control,c.CONTROL,c.RANKS,c.pin_to(pins,guard))
        old = SimpleNamespace(TRUTH_INDEX=c.BD.parent/'response_rows_panel_v1/score/result.json')
        c.pin_to(pins,guard)(old.TRUTH_INDEX,'a31981681305de1828c48d8b6e8830c87ed979789c7d600b949217814f479cdd')
        context.update(h=h,p=p,old=old,pins=pins,control=control,control_score=control_score,
            control_outputs=control_outputs,tl=SimpleNamespace(BANK=c.P1_CANDIDATE,BANK_SHA=c.P1_SHA))
        c.require(not any(x in sys.modules for x in ('numpy','torch','flopscope')), 'All acquisition seals before numeric imports')
        scorer,proof = score_function(h)
        result.update(proof=proof,prerequisites=context['prerequisites'])
        c.write(c.SCORE_OUT/'started.json',result)
        scorer(args,context,c.SCORE_OUT,result,guard)
        c.require(result['baseline_recomputed_exact'] and len(result['per_role']['P1']) == 14 and
                  result['summaries']['P1'] == control_score['summaries']['terminal'], 'Exact all14 P1 rows/all3 summaries')
        guard()
    except BaseException:
        result.update(passed=False,completed=False,qualification_passed=False,integrity_passed=False,
            hashes_unchanged=False,sources_unchanged=False,retain_local_component=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        result.update(wall_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            finished_utc=datetime.now(timezone.utc).isoformat())
        if result['wall_s'] >= 120 or result['peak_rss_bytes'] >= 2_000_000_000:
            result.update(passed=False,completed=False,qualification_passed=False,integrity_passed=False,
                          retain_local_component=False)
        c.write(c.SCORE_OUT/'result.json',result)
        c.write(c.SCORE_OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            integrity_passed=result['integrity_passed'],retain_local_component=result['retain_local_component'],
            result_sha256=c.sha(c.SCORE_OUT/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','integrity_passed','retain_local_component',
        'adjusted_candidate_over_parent','error')},allow_nan=False),flush=True)
    return 0 if result['completed'] and result['integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
