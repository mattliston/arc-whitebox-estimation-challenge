"""Read-only independent audit of the completed, once-only f938 release artifacts."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
O = HERE.parents[4]
PROJECT = O.parents[4]
PINS = {}


def sha(path):
    path = Path(path).resolve()
    result = hashlib.sha256(path.read_bytes()).hexdigest()
    PINS[str(path)] = result
    return result


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


plan_path = HERE/'RELEASE_ISOLATED_PLAN_v2.json'
assert sha(plan_path) == 'ead7e8bb2fde1ec46a8b6ddf325ab9e9cbec32469019fe75953c854c981e2942'
plan = read(plan_path)
iso = read(HERE/'ISOLATION_v2.json')
candidate = Path(plan['candidate'])
source_sha = 'f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6'
assert sha(candidate) == plan['candidate_sha256'] == source_sha
assert plan['one_unchanged_release_authorized'] is True and plan['automatic_fire'] is False
expected = ['bash', str(O/'lead/runs/fable_oct19_l13rank_v1/release_min_v1.sh'), str(candidate),
            'astra_first128_paid_nocensus_f9381d09_v1', str(HERE/'qualification_v2'),
            str(O/'release/runs/astra_edge8_hidden_complement90_4685e145_v1/submission.tar.gz')]
assert plan['command'] == iso['command'] == expected
assert plan['archive_directory'] == str(O/'release/runs'/expected[3])
assert iso['manifest_sha256'] == sha(plan_path)
assert iso['source_pins'] == plan['source_pins']
for path, digest in plan['source_pins'].items():
    assert sha(path) == digest, ('Changed release input', path)
assert sha(plan['release_min']) == 'ec26493b98f4cae46e2f20d2745a2af96be0def430500770678e315c0f566258'
assert sha(HERE/'run_release_isolated_root_v2.py') == plan['runner_sha256'] == iso['runner_sha256'] == '75f0c976d29da6ea94c72281376fbdeccb2a7cc6bd2e9a6b8de0cb45c5ddcd45'
assert sha(plan['reference_archive']) == plan['reference_archive_sha256'] == '3848cfaed85175d0f0f817e99234201cc5736bad51e071eabc553b719145947a'
assert sha(plan['runtime_verdict']) == plan['runtime_verdict_sha256'] == '4ddc678aa15a57053f8045fca26ef4d2214226b10eb1a4023c09b19232d9418d'
verdict = read(plan['runtime_verdict'])
assert verdict['successful'] is True and verdict['material_measured_margin_accepted'] is True
assert verdict['release_source_sha256'] == source_sha
assert iso['status'] == 'UNCHANGED_RELEASE_SCRIPT_COMPLETED_ROOT_ARTIFACT_REVIEW_REQUIRED'
assert iso['exitcode'] == iso['release_returncode'] == 0
assert iso['no_fire'] is True and iso['ownership_retirement_proven'] is True
assert iso['resumed'] == [77740] and not iso['restore_errors'] and not iso['not_resumed']
assert iso['research_resumption_withheld'] is False and iso['free_pct_before'] >= 60
retired = iso['ownership_retirement']
assert retired['proven'] is True and not retired['initially_live_after_command'] and not retired['uncertainty']
assert not retired['individually_checked_signals'] and retired['broad_group_signals_used'] is False
assert len(retired['checks']) == 2
assert all(not x['owned_survivors'] and not x['unknown_project_or_group_processes'] for x in retired['checks'])

q = Path(plan['qualification_output'])
guard = read(q/'SETUP_GUARD_v1.json')
setup_plan = read(q/'setup5/PLAN_v1.json')
setup = read(q/'setup5/SUMMARY_v1.json')
assert guard['guard_version'] == 3 and guard['completed'] is True and guard['whole_group_retired'] is True
assert guard['exit_code'] == 0 and guard['error'] is None and not guard['remaining_rows']
assert guard['footprint_trigger_bytes'] == guard['rss_backstop_bytes'] == 7500000000
assert guard['max_rss_bytes'] < 7500000000 and guard['max_footprint_bytes'] < 7500000000
assert guard['elapsed_s'] < 120 and guard['min_pressure_free_pct'] >= 50
assert not guard['failed_readings'] and not guard['tolerated_failures'] and guard['max_swap_growth_bytes'] == 0
assert guard['command'] == [str(O/'lead/runs/astra_crunch_runtime_v1/failure336631_v1/client310_v1/bin/python'),
    '-B', str(O/'lead/runs/astra_oct18_accuracy_v1/official_setup_five_v2.py'), str(candidate),
    str(q/'setup5'), '--setup-timeout', '5']
assert setup_plan['candidate'] == str(candidate) and setup_plan['sha256'] == source_sha
assert setup_plan['workers'] == 5 and setup_plan['setup_timeout_s'] == 5
assert setup_plan['setup_budget'] is True and setup_plan['no_predict'] is True
assert setup_plan['official_controller'] == 'whestbench.runner.SubprocessRunner'
assert setup_plan['versions'] == dict(whestbench='0.16.1', flopscope_client='0.12.3', flopscope_server='0.12.3')
assert setup['status'] == 'PASS' and setup['error'] is None and setup['server_processes_retired'] is True
assert setup['server_returncodes'] == [-15]*5
assert [r['worker'] for r in setup['records']] == list(range(5))
workers, rlimit_notes = [], []
for record in setup['records']:
    j = record['worker']
    assert read(q/f'setup5/worker{j}.json') == record
    assert record['status'] == 'PASS' and record['worker_retired'] is True
    assert not record.get('error') and not record.get('close_error')
    assert record['setup_timeout_s'] == 5 and 0 < record['start_envelope_s'] < 5
    assert len(record['start_responses']) == 1 and record['start_responses'][0]['status'] == 'ok'
    diagnostic = record['start_responses'][0]['diagnostic']
    assert diagnostic['setup_completed'] is True and diagnostic['setup_budget'] is True and not diagnostic['bind_errors']
    assert diagnostic['setup_flops'] == 27687270724
    event_path = q/f'setup5/worker{j}_events.jsonl'
    sha(event_path)
    events = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
    assert events[0]['stage'] == 'boundary_bootstrap' and events[-1]['stage'] == 'setup_end'
    assert events[0]['numpy_available'] is False and events[0]['flopscope'] == '0.12.3'
    assert events[0]['setup_budget'] is True and events[-1]['setup_completed'] is True
    assert events[-1]['setup_flops'] == diagnostic['setup_flops'] and not events[-1]['bind_errors']
    errors = [e for e in events if e.get('error')]
    assert len(errors) == 1 and errors[0]['stage'] == 'rlimit_note'
    assert errors[0]['error'] == "ValueError('current limit exceeds maximum limit')"
    rlimit_notes.append(dict(worker=j, note=errors[0]['error']))
    workers.append(dict(worker=j, start_envelope_s=record['start_envelope_s'], setup_flops=diagnostic['setup_flops'],
                        worker_retired=True, completed=True, numpy_available=False, setup_budget=True))
    sha(q/f'setup5/server{j}.log')

a = Path(plan['archive_directory'])
stage = read(a/'stage_a.json')
assert stage['stage'] == 'A' and stage['passed'] is True and stage['uploaded'] is False
assert stage['candidate'] == str(candidate) and stage['candidate_sha256'] == source_sha
assert set(stage['gates']) == {'contract', 'package', 'validate_package', 'archive_contents', 'no_numpy_import',
    'einsum_signatures_proven', 'einsum_no_new_repeated_operand', 'python310'}
assert all(stage['gates'].values()) and not stage['new_einsum_signatures'] and not stage['new_repeated_operand_einsums']
assert not stage['reviewed_einsum_signatures']
archive = a/'submission.tar.gz'
assert stage['archive'] == str(archive)
archive_sha = sha(archive)
assert archive_sha == 'c63a799cc9402f3fb9aecf9b3fb5bf059db65362b9c8dc63803a83d7176f3d1d'
with tarfile.open(archive, 'r:gz') as handle:
    members = handle.getmembers()
    assert len(members) == 2 and sorted(x.name for x in members) == ['estimator.py', 'manifest.json']
    assert all(x.isfile() for x in members)
    body = handle.extractfile('estimator.py').read()
    manifest = json.loads(handle.extractfile('manifest.json').read())
assert body == candidate.read_bytes() == (a/'payload/estimator.py').read_bytes()
assert sha(a/'payload/estimator.py') == source_sha
assert hashlib.sha256(body).hexdigest() == source_sha
assert manifest['entrypoint'] == {'module': 'estimator', 'class': 'Estimator'}
assert manifest['files'] == [{'name': 'estimator.py', 'sha256': source_sha}]
with tarfile.open(plan['reference_archive'], 'r:gz') as handle:
    old_body = handle.extractfile('estimator.py').read()
    old_manifest = json.loads(handle.extractfile('manifest.json').read())
assert hashlib.sha256(old_body).hexdigest() == '4685e1454ebd2d9607fcd88b64cc1d8435050761212efd1902b04d3f83151a46'
assert {k:v for k,v in manifest.items() if k not in ('files','created_at_utc')} == {
    k:v for k,v in old_manifest.items() if k not in ('files','created_at_utc')}
whest = str(PROJECT/'.venv/bin/whest')
expected_commands = {
    'validate.json': [whest, 'validate', '--estimator', str(candidate), '--class', 'Estimator', '--json'],
    'package.json': [whest, 'package', '--estimator', str(a/'payload/estimator.py'), '--class', 'Estimator',
                     '--output', str(archive), '--yes', '--json'],
    'validate_package.json': [whest, 'validate-package', str(archive), '--json'],
}
for name, command in expected_commands.items():
    record = read(a/name)
    assert record['command'] == command and record['returncode'] == 0 and not record['stderr']
    answer = json.loads(record['stdout'])
    assert answer['ok'] is True
    if name == 'validate.json':
        assert answer['class_name'] == 'Estimator' and answer['output_shape'] == [2,4]
    if name == 'validate_package.json':
        assert not answer['issues']

source = body.decode()
old_source = old_body.decode()
imports = set(re.findall(r'^\s*(?:import|from)\s+([\w.]+)', source, re.M))
assert imports == {'base64','flopscope','flopscope.numpy','gc','math','time','whestbench','zlib'}
banned = ('exec(', 'eval(', 'compile(', '__import__', 'ctypes', 'subprocess', 'threading',
    'multiprocessing', 'import numpy', 'from numpy', 'pickle', 'marshal', 'importlib', 'socket')
assert not any(token in source for token in banned)
signature = lambda src:set(re.findall(r'''einsum\(\s*['"]([^'"]*)['"]''', src))
assert signature(source) == signature(old_source) and len(signature(source)) == 23
tree = ast.parse(source)
assert not any(isinstance(node, ast.Name) and node.id in ('torch','os') for node in ast.walk(tree))
prefire = (q/'prefire.log').read_text()
assert [line for line in prefire.splitlines() if line.startswith('FAIL')] == ["FAIL imports outside the allowed set: ['gc']"]
assert 'new signatures: 0' in prefire and source_sha in prefire and archive_sha in prefire
release_log = (HERE/'release_min_v2.log').read_text()
assert sha(HERE/'release_min_v2.log') == iso['log_sha256']
for marker in (' SETUP5 START ', ' STAGE A START ', ' PREFIRE START', ' GATES CLEAR: archive '):
    assert release_log.count(marker) == 1
assert 'SETUP5 PASS' in release_log and 'PREFIRE other FAIL lines: 0, new einsum signatures: 0' in release_log
assert archive_sha[:12] in release_log
for path in (q/'prefire.log', q/'stage_a.log', q/'setup5.log'):
    sha(path)
sha(__file__)
report = dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
    status='INDEPENDENT_UNCHANGED_RELEASE_ARTIFACT_AUDIT_PASS_NO_FIRE',
    independent_method='Standalone raw artifact and archive verification; no release script, prefire script, supplied analyzer or candidate execution.',
    candidate=str(candidate), source_sha256=source_sha, archive=str(archive), archive_sha256=archive_sha,
    archive_bytes=archive.stat().st_size, estimator_bytes=len(body), reference_submission=337473,
    reference_archive_sha256=plan['reference_archive_sha256'], reference_estimator_sha256=hashlib.sha256(old_body).hexdigest(),
    complete_six_argument_invocation_exact=True, unchanged_release_script=True, all_input_pins_rehashed=True,
    official_setup_workers=workers, maximum_start_envelope_s=max(x['start_envelope_s'] for x in workers),
    official_setup_versions=setup_plan['versions'], setup_guard_original=True,
    setup_guard_max_rss_bytes=guard['max_rss_bytes'], setup_guard_max_footprint_bytes=guard['max_footprint_bytes'],
    setup_guard_min_free_pct=guard['min_pressure_free_pct'], setup_guard_swap_growth_bytes=guard['max_swap_growth_bytes'],
    all_setup_workers_servers_and_owned_process_groups_retired=True, no_cleanup_signals_needed=True,
    research_queue_77740_restored=True, stage_a_all_eight_gates_pass=True,
    archive_exact_regular_members=['estimator.py','manifest.json'], archive_estimator_bytes_exact=True,
    package_metadata_matches_last_fired_except_source_and_timestamp=True,
    packager_versions_are_separate_from_official_setup=True, manifest_metadata=manifest,
    prefire_only_accepted_gc_import=True, new_einsum_signatures=0, no_banned_source_tokens=True,
    existing_nonfatal_rlimit_notes=rlimit_notes,
    rlimit_note_interpretation='The unchanged boundary worker could not raise its existing hard limit. All five official setup budgets and the independent original7.5GB group guard passed; no limit or guard was modified.',
    no_new_release_run=True, no_candidate_import_or_native=True, no_fire=True, no_BEST_change=True,
    fire_authority='Root only under existing authorization; this audit verifies artifacts and does not submit.', pins=PINS)
destination = HERE/'INDEPENDENT_RELEASE_ARTIFACT_AUDIT_v1.json'
assert not destination.exists()
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(path=str(destination),sha256=sha(destination),pins=len(PINS),archive_sha256=archive_sha,
                     maximum_start_envelope_s=report['maximum_start_envelope_s'],status=report['status'])))
