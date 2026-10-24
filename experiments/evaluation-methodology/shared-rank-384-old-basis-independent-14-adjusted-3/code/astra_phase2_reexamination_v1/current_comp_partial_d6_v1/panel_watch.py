"""Fixed unchanged-CD independent14 acquisition; two retired owned phases."""
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from types import SimpleNamespace
import panel_common as c

_OWNER = c.COMP/'panel_watch.py'
_OWNER_SHA = '324bcd2a43f931f1a38f0f472e99cf593c9a08d8e2f1f63c0ca5ff5ef0d5c904'
_NATIVE_FLAGS = ('all16_domains_qualified', 'all47_selectors_qualified',
    'own_complete_bills', 'repeat_output_domain_byte_exact',
    'initialization_ownership_passed', 'additional_bindings_paid_once',
    'same_existing_arena_qualified', 'dispatch_tariffs_qualified',
    'COMP_parameter_initialization_owned', 'layer0_byte_exact',
    'all90_artifacts_qualified', 'native40_expansion_prerequisite_passed',
    'full_admission_in_retired_process')


def third_seal(e, folder, result_sha, completion_sha, limit, tick):
    path = folder/'finalization.json'
    item = e.read(path)
    e.require(item.get('passed') is True and item.get('completed') is True
        and item.get('result_sha256') == result_sha
        and item.get('completion_sha256') == completion_sha
        and 0 <= item.get('elapsed_s', -1) < limit,
        'Actual new producer third seal, prior hashes and full clock')
    return e.sha(path, tick)


def native_contract(e, record):
    e.require(record['full_model_calls'] == record['retired_call_count'] == 15
        and len(record['artifacts']) == 90
        and all(record.get(k) is True for k in _NATIVE_FLAGS),
        'All15 own calls/all90 artifacts, D6/startup/domain/fixture qualification')


def final_watch_seal(e, result, began, limit, deadline):
    result_sha = e.sha(e.WATCH_OUT/'result.json')
    completion_sha = e.sha(e.WATCH_OUT/'completion.json')
    elapsed = time.monotonic()-began
    passed = bool(result['passed'] and result['completed'] and elapsed < limit)
    path = e.WATCH_OUT/'finalization.json'
    e.write(path, dict(passed=passed, completed=passed, result_sha256=result_sha,
        completion_sha256=completion_sha, elapsed_s=elapsed))
    e.sha(path)
    passed = bool(passed and time.monotonic()-began < limit)
    if passed:
        deadline()
    result.update(passed=passed, completed=passed)


EXTRA_MAIN_EDITS = (
    ("            result.update(admission_result_sha256=e.sha(rp,deadline),admission_completion_sha256=e.sha(cp,deadline),\n",
     "            result.update(admission_result_sha256=e.sha(rp,deadline),admission_completion_sha256=e.sha(cp,deadline),\n"
     "                admission_finalization_sha256=panel_third_seal(e,e.ADMISSION,e.sha(rp,deadline),e.sha(cp,deadline),60,deadline),\n"),
    ("            result.update(child_result_sha256=e.sha(rp,deadline),child_completion_sha256=e.sha(cp,deadline))",
     "            result.update(child_result_sha256=e.sha(rp,deadline),child_completion_sha256=e.sha(cp,deadline),\n"
     "                child_finalization_sha256=panel_third_seal(e,e.OUTPUT,e.sha(rp,deadline),e.sha(cp,deadline),1200,deadline))"),
    ("            e.verify_native(record,end,result['child_result_sha256'],args,context)\n",
     "            e.verify_native(record,end,result['child_result_sha256'],args,context)\n"
     "            panel_native_contract(e,record)\n"),
    ("(e.ADMISSION,('result.json','completion.json','context.json')),(e.OUTPUT,('result.json','completion.json'))",
     "(e.ADMISSION,('result.json','completion.json','finalization.json','context.json')),\n"
     "                (e.OUTPUT,('result.json','completion.json','finalization.json'))"),
    ("                          'Final native result/completion immutable')\n",
     "                          'Final native result/completion immutable')\n"
     "                e.require(e.sha(e.OUTPUT/'finalization.json',deadline)==result['child_finalization_sha256']\n"
     "                    and e.sha(e.ADMISSION/'finalization.json',deadline)==result['admission_finalization_sha256'],\n"
     "                    'Both new child third seals remain immutable through final observation')\n"),
    ("            result_sha256=e.sha(e.WATCH_OUT/'result.json')))\n",
     "            result_sha256=e.sha(e.WATCH_OUT/'result.json')))\n"
     "        panel_final_watch_seal(e,result,began,WALL_LIMIT,deadline)\n"),
)


def watch_owner():
    path = c.COMP/'watch_native_v2.py'
    c.require(c.sha(path) == c.WATCH_OWNER_SHA, 'Exact original underlying watch owner')
    source = c.splice(path.read_text(), [('import endpoint_v2 as e\n', 'e = ORIGINAL_ENDPOINT_OWNER\n')])
    namespace = dict(__file__=str(path), __name__='cd_panel_private_native_watch_owner',
        ORIGINAL_ENDPOINT_OWNER=c.e)
    exec(compile(source, str(path)+':private-metadata-import', 'exec'), namespace)
    return SimpleNamespace(**namespace)


def native_watch_source():
    c.require(c.sha(_OWNER) == _OWNER_SHA, 'Exact qualified COMP independent14 supervisor owner')
    original = c.function_source(_OWNER, 'native_watch')
    source = c.splice(original, [
        ("original = c.function_source(c.HERE/'watch_native_v2.py','main')",
         "original = c.function_source(c.COMP/'watch_native_v2.py','main')"),
        ("owner = c.load(c.HERE/'watch_native_v2.py','node_panel_qualified_native_supervisor',c.WATCH_OWNER_SHA)",
         "owner = watch_owner()"),
        ("\"schema=e.PREFIX+'admission_watch_v2'\"", "\"schema=e.RETIRED_SCHEMA\""),
        ('    edits = [\n', '    edits = list(EXTRA_MAIN_EDITS)+[\n'),
        ('metadata = SimpleNamespace(**dict(vars(n),VALID=c.RANKS))',
         "metadata = SimpleNamespace(**dict(vars(n),VALID=c.RANKS,\n"
         "        CANDIDATES={'terminal':c.CANDIDATE_SHA},BUILD_SHA=c.BUILD_SHA))"),
        ('        monitor=monitor,inventory_owner=inventory,WATCH_PROOF=proof)',
         '        monitor=monitor,inventory_owner=inventory,WATCH_PROOF=proof,\n'
         '        panel_third_seal=third_seal,panel_native_contract=native_contract,\n'
         '        panel_final_watch_seal=final_watch_seal)'),
    ])
    ast.parse(source)
    return source, dict(parent=str(_OWNER), parent_sha256=_OWNER_SHA,
        original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        full_source_inverse_verified=True, original_native_resource_loop_unchanged=True,
        only_new_CD_lineage_and_three_seal_postconditions=True,
        old_COMP_two_seal_receipts_not_modified=True, no_scoring_branch=True)


def main():
    args = c.arguments('watcher')
    if args is None:
        return 0
    c.environment()
    c.owners(args)
    c.require(not c.ROOT.is_symlink(), 'Fixed nonsymlink panel root')
    c.ROOT.mkdir(exist_ok=True)
    source, proof = native_watch_source()
    namespace = dict(globals())
    exec(compile(source, str(_OWNER)+':fixed-CD-native15-only', 'exec'), namespace)
    return namespace['native_watch'](args)


if __name__ == '__main__':
    raise SystemExit(main())
