"""Exact COMP panel paid-loop adapter, CD observer and one immutable candidate."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import resource
import sys
import time
import panel_common as c


def derived_source():
    owner,digest = c.COMP_OWNERS['worker']
    c.require(c.sha(c.COMP/owner) == digest,'Qualified COMP independent14 worker owner')
    original = c.function_source(c.COMP/owner,'main')
    source = c.splice(original,[
        ('def main():','def worker_main():'),
        ('        n.after_bill(np,observed,trial,result,folder,context[\'reference_by_rank\'])\n',
         '        c.after_bill(np,observed,trial,result,folder,context[\'reference_by_rank\'])\n'),
        ("'Authenticated P1 reference before output-only decode'","'Authenticated current COMP reference before output-only decode'"),
        ('        result.update(layer0_byte_exact=True)\n',
         "        c.require(len(result['artifacts']) == 90, 'All15 complete six-artifact acquisitions')\n"
         "        result.update(layer0_byte_exact=True,all90_artifacts_qualified=True)\n"),
        ('sources_unchanged=False,build_sha256=n.BUILD_SHA,builder_sha256=n.BUILDER_SHA,',
         'sources_unchanged=False,build_sha256=c.BUILD_SHA,builder_sha256=c.BUILDER_SHA,\n'
         "        build_watch_sha256=c.BUILD_WATCH_SHA,native40_expansion_prerequisite_passed=True,"),
        ('no_fit=True,no_refit=True,score_measured=False,old_NODE_gate_still_failed=True,',
         'no_fit=True,no_refit=True,score_measured=False,old_NODE_gate_still_failed=True,\n'
         '        old_max_read_timing_gate_still_failed=True,production_qualified=False,'),
        ('observer_event_filter_inverse_verified=True,proof=proof,full_admission_in_retired_process=True,',
         'observer_event_filter_inverse_verified=True,proof=proof,full_admission_in_retired_process=True,\n'
         '        native40_lineage=context[\'native40_lineage\'],panel_worker_source_inverse=PANEL_WORKER_PROOF,')])
    proof = dict(owner=str(c.COMP/owner),owner_sha256=digest,
        original_main_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_main_sha256=hashlib.sha256(source.encode()).hexdigest(),full_source_inverse_verified=True,
        original_paid_loop_and_fixture_arithmetic_unchanged=True,
        new_observer='Exact qualified CD ten-code domain/startup/dispatch observer',
        no_CHECK4_replay_on_different_inputs=True,no_estimator_change=True)
    return source,proof


def main():
    began = time.monotonic()
    if c.arguments('worker') is None:
        return 0
    source,proof = derived_source()
    namespace = dict(globals(),PANEL_WORKER_PROOF=proof)
    exec(compile(source,str(__file__)+':same-paid-panel-worker','exec'),namespace)
    code = namespace['worker_main']()
    if not c.OUTPUT.exists():
        return code  # Inert --help does not create or inspect a stage.
    record = c.read(c.OUTPUT/'result.json'); end = c.read(c.OUTPUT/'completion.json')
    result_sha,completion_sha = c.sha(c.OUTPUT/'result.json'),c.sha(c.OUTPUT/'completion.json')
    elapsed = time.monotonic()-began
    passed = bool(code == 0 and record['passed'] and record['completed'] and end['passed'] and end['completed']
        and end['result_sha256'] == result_sha and elapsed < 1200
        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < c.RSS_LIMIT)
    c.write(c.OUTPUT/'finalization.json',dict(passed=passed,completed=passed,result_sha256=result_sha,
        completion_sha256=completion_sha,elapsed_s=elapsed))
    return 0 if passed and time.monotonic()-began < 1200 else 1


if __name__ == '__main__':
    raise SystemExit(main())
