"""Separate, opt-in exact d50 packaging and one bank-confirmation submission.

Default/help is stdlib-only and performs no numerical import, estimator
call, package, key read or upload. --package requires the reviewed helper
and completed runtime SHA. --submit additionally requires the package
receipt SHA; it rechecks every input and the actual archive before writing
an exclusive attempt marker and invoking unmodified Whest0.16 ONCE.

No wrapper retry, --watch, login, key/config write, source edit or forum
operation exists. The pinned Whest client itself has bounded transient
HTTP retries (SUBMIT_RETRY:5 attempts/45s per logical request); one CLI
invocation is NOT a promise of one HTTP attempt. A timeout/ambiguous result
must be investigated externally, never retried by this helper.

Submission response persistence is an approved-field projection, not raw
stdout/stderr. Credentials enter only the child's in-memory environment.
Neither an accepted submission nor this helper claims an official score.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
BD, NEW, P2 = HERE.parent, HERE.parent.parent, HERE.parent.parent.parent
ENV = P2 / 'data_gate/.venv'
SITE = ENV / 'lib/python3.11/site-packages/whestbench'
CLI = ENV / 'bin/whest'
SOURCE = BD / 'build_terminal_v1/estimator.py'
CANDIDATE_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
PANEL_SHA = '6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade'
SCORE_SHA = '767cf92415a0391f7b7596c9681adede144bd10b3e5d36f354ec2dce7ce52c1b'
RUNTIME_RESULT, RUNTIME_COMPLETION = HERE / 'runtime_result.json', HERE / 'runtime_completion.json'
STAGE, PACKAGE_RESULT = HERE / 'package_stage', HERE / 'package_result.json'
ARCHIVE = HERE / 'birth_direction_d50bb5c5_whest016.tar.gz'
ATTEMPT, RESPONSE = HERE / 'submit_attempt.json', HERE / 'submit_response.json'
KEY_FILE = Path('/Users/matt/arc-white-box-estimation/key.txt')
CHALLENGE = 'arc-white-box-estimation-challenge-2026'
DESCRIPTION = ('Phase 2 birth-orientation bank confirmation: fixed learned newborn directions; '
               'validated independent14 panel and five NumPy-free client/OpenBLAS runtime checks')
VERSIONS = {'whestbench': '0.16.0', 'flopscope': '0.12.0', 'numpy': '2.4.6'}
PINS = {
    CLI: '3dac97e9083bda232206eb95fc77e4e33f8e7bc14fecdf2f1a441759d98b6b63',
    SITE / 'cli.py': 'f6b975fd39a1dfda5e71efada4f7f4da0b5fde0b0130816484a660f91dd2b58d',
    SITE / 'packaging.py': '64f38dfe46951e1f49e40b8872e137a9338b7b33b115d8edd6058ddd3431b0a6',
    SITE / 'validation.py': '239ceb596742baf534743a64ea9bf539011677eb7d0cc13f31b22593660a236b',
    SITE / 'aicrowd_config.py': 'fcf5175963f0a905b05a7daa4419c964376f4c11ed761b35595dd53bc253a188',
    SITE / 'aicrowd_client.py': '441cf36acb61266a960164de80fc66b21c5015773a898c76d1807b9c46bae6a6',
}


class CheckFailure(RuntimeError):
    """Only fixed, non-secret diagnostic codes are passed to this class."""


def require(value, code):
    if not value:
        raise CheckFailure(code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def bind(bindings, path, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    require(expected is None or actual == expected, 'input_sha_mismatch')
    require(str(path) not in bindings or bindings[str(path)] == actual, 'conflicting_input_binding')
    bindings[str(path)] = actual
    return actual


def unchanged(bindings):
    require(all(sha(path) == digest for path, digest in bindings.items()), 'sealed_input_changed')


def environment():
    require(Path(sys.prefix).resolve() == ENV.resolve(), 'OFFPY_required')
    require({k: importlib.metadata.version(k) for k in VERSIONS} == VERSIONS, 'installed_versions_changed')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10_and_no_bytecode_required')
    require(all(os.environ.get(k) == '1' for k in
                ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')), 'BLAS1_required')
    require('numpy' not in sys.modules and 'torch' not in sys.modules, 'stdlib_parent_required')


def verify_runtime(args):
    require(valid_sha(args.source_sha256) and sha(__file__) == args.source_sha256, 'reviewed_helper_sha_required')
    require(valid_sha(args.runtime_sha256), 'completed_runtime_sha_required')
    bindings = {}
    bind(bindings, __file__, args.source_sha256)
    for path, expected in PINS.items():
        bind(bindings, path, expected)
    bind(bindings, SOURCE, CANDIDATE_SHA)
    bind(bindings, RUNTIME_RESULT, args.runtime_sha256)
    bind(bindings, RUNTIME_COMPLETION)
    runtime, completion = load(RUNTIME_RESULT), load(RUNTIME_COMPLETION)
    require(all(runtime.get(k) is True for k in ('passed', 'completed', 'runtime_gates_passed',
                'native_fidelity_passed', 'hashes_unchanged')), 'runtime_and_native_fidelity_must_pass')
    require(completion.get('passed') is True and completion.get('completed') is True
            and completion.get('result_sha256') == args.runtime_sha256, 'runtime_completion_seal')
    require(runtime['candidate_sha256'] == CANDIDATE_SHA
            and runtime['native_panel_result_sha256'] == PANEL_SHA
            and runtime['native_panel_score_sha256'] == SCORE_SHA, 'same_d50_native_panel_frame')
    require(runtime.get('all5_complete') is True
            and [x['rank'] for x in runtime['checks']] == [35, 100, 500, 3016500, 35]
            and [x['rank'] for x in runtime['per_net']] == [35, 100, 500, 3016500],
            'complete_fixed_five_context_runtime_and_four_unique_fidelity_records')
    for field in ('source_bindings', 'artifacts'):
        require(isinstance(runtime[field], dict) and runtime[field], 'runtime_bindings_required')
        for path, expected in runtime[field].items():
            require(Path(path).is_absolute() and valid_sha(expected), 'flat_absolute_runtime_binding_required')
            bind(bindings, path, expected)
    # Trust the reviewed, explicitly pinned complete runtime verdict for its
    # five-context numerical gates; do not rerun a model or decode outputs.
    for subdir, expected in (('terminal', PANEL_SHA), ('score', SCORE_SHA)):
        folder = BD / 'panel_v2' / subdir
        bind(bindings, folder / 'result.json', expected)
        bind(bindings, folder / 'completion.json')
        record, seal = load(folder / 'result.json'), load(folder / 'completion.json')
        require(record['passed'] and record['completed'] and record['hashes_unchanged']
                and record['candidate_sha256'] == CANDIDATE_SHA and seal['passed'] and seal['completed']
                and seal['result_sha256'] == expected, 'completed_native14_panel_seal')
        if subdir == 'score':
            require(record['retain_local_component'], 'native14_component_not_retained')
    return bindings


def inspect_archive():
    require(ARCHIVE.is_file() and not ARCHIVE.is_symlink(), 'regular_archive_required')
    source = SOURCE.read_bytes()
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        members = tar.getmembers()
        require(sorted(m.name for m in members) == ['estimator.py', 'manifest.json']
                and all(m.isfile() for m in members), 'exact_two_regular_archive_members')
        values = {}
        for member in members:
            require(member.size == len(source) if member.name == 'estimator.py'
                    else 0 < member.size < 262144, 'archive_member_size')
            stream = tar.extractfile(member)
            require(stream is not None, 'readable_archive_member')
            with stream:
                values[member.name] = stream.read()
    require(values['estimator.py'] == source and hashlib.sha256(source).hexdigest() == CANDIDATE_SHA,
            'packaged_estimator_bytes_must_be_exact_d50')
    manifest = json.loads(values['manifest.json'])
    require(manifest['schema_version'] == '1.0' and manifest['api_version'] == '2.0'
            and manifest['entrypoint'] == {'module': 'estimator', 'class': 'Estimator'}
            and manifest['files'] == [{'name': 'estimator.py', 'sha256': CANDIDATE_SHA}]
            and manifest['whestbench_version'] == VERSIONS['whestbench']
            and manifest['flopscope_version'] == VERSIONS['flopscope']
            and manifest['numpy_version'] == VERSIONS['numpy'], 'exact_submission_manifest')
    return dict(archive=str(ARCHIVE), archive_sha256=sha(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size,
        members={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                 for name, data in values.items()}, manifest=manifest)


def safe_failure(exc):
    # Never format a subprocess exception, traceback, environment or key.
    return dict(type=type(exc).__name__, code=str(exc) if isinstance(exc, CheckFailure) else 'stage_exception')


def approved_response(value, secret=None):
    """Whitelist public scalar receipt fields; no raw API/upload payload."""
    if not isinstance(value, dict):
        return dict(json_object=False)
    out = dict(json_object=True, ok=value.get('ok') is True)
    sid = value.get('submission_id')
    if type(sid) is int and sid > 0:
        out['submission_id'] = sid
    for key in ('error_code', 'status'):
        item = value.get(key)
        if type(item) is int or (isinstance(item, str) and re.fullmatch('[A-Za-z0-9_.-]{1,64}', item)
                                and not (secret and secret in item)):
            out[key] = item
    submission = value.get('submission')
    if isinstance(submission, dict):
        if isinstance(submission.get('data'), dict):
            submission = submission['data']
        public = {}
        for key in ('id', 'challenge_id', 'score', 'score_secondary'):
            item = submission.get(key)
            if type(item) in (int, float) and math.isfinite(item):
                public[key] = item
        for key in ('grading_status_cd', 'status', 'created_at', 'updated_at'):
            item = submission.get(key)
            if (isinstance(item, str) and re.fullmatch('[A-Za-z0-9_.:+ -]{1,80}', item)
                    and not (secret and secret in item)):
                public[key] = item
        out['submission'] = public
    if secret and secret in json.dumps(out, sort_keys=True):
        return dict(json_object=True, ok=value.get('ok') is True, redacted=True)
    return out


def run_cli(arguments, env, timeout, result, name, secret=None):
    proc = subprocess.run([str(CLI)] + arguments, cwd=STAGE, env=env,
                          capture_output=True, text=True, timeout=timeout)
    try:
        payload = json.loads(proc.stdout)
    except (ValueError, TypeError):
        payload = None
    record = dict(command=name, returncode=proc.returncode,
                  stdout_bytes=len(proc.stdout.encode()), stderr_bytes=len(proc.stderr.encode()),
                  response=approved_response(payload, secret))
    result['commands'].append(record)
    require(proc.returncode == 0 and isinstance(payload, dict) and payload.get('ok') is True,
            'whest_command_failed')
    return record['response']


def package(args, bindings):
    require(not STAGE.exists() and not ARCHIVE.exists() and not PACKAGE_RESULT.exists(), 'exclusive_package_outputs')
    STAGE.mkdir()
    start = time.monotonic()
    result = dict(stage='package', passed=False, completed=False, candidate_sha256=CANDIDATE_SHA,
        runtime_result_sha256=args.runtime_sha256, reviewed_source_sha256=args.source_sha256,
        source_bindings=bindings, commands=[], uploaded=False, official_score=False, started_utc=utc())
    try:
        with (STAGE / 'estimator.py').open('xb') as stream:
            stream.write(SOURCE.read_bytes())
        require(sha(STAGE / 'estimator.py') == CANDIDATE_SHA, 'staged_source_bytes')
        env = dict(os.environ)
        env.pop('AICROWD_API_KEY', None)  # Packaging needs no credential.
        run_cli(['validate', '--estimator', str(STAGE / 'estimator.py'), '--json'], env, 120, result, 'validate')
        run_cli(['package', '--estimator', str(STAGE / 'estimator.py'), '--output', str(ARCHIVE), '--yes', '--json'],
                env, 120, result, 'package')
        run_cli(['validate-package', str(ARCHIVE), '--json'], env, 120, result, 'validate-package')
        result.update(inspect_archive())
        unchanged(bindings)
        require(sha(STAGE / 'estimator.py') == CANDIDATE_SHA, 'staged_source_changed')
        result.update(passed=True, completed=True, hashes_unchanged=True,
            validation_scope='Generic Whest guard and exact archive checks; separately sealed native14/runtime5 qualification.')
    except BaseException as exc:
        result['error'] = safe_failure(exc)
    result.update(elapsed_s=time.monotonic()-start, ended_utc=utc())
    write(PACKAGE_RESULT, result)
    print(json.dumps(dict(stage='package', passed=result['passed'], receipt=str(PACKAGE_RESULT),
                         receipt_sha256=sha(PACKAGE_RESULT), uploaded=False)), flush=True)
    return 0 if result['passed'] else 1


def submit(args, bindings):
    require(valid_sha(args.package_sha256), 'explicit_package_receipt_sha_required')
    bind(bindings, PACKAGE_RESULT, args.package_sha256)
    package_record = load(PACKAGE_RESULT)
    require(package_record['passed'] and package_record['completed'] and package_record['hashes_unchanged']
            and package_record['uploaded'] is False and package_record['candidate_sha256'] == CANDIDATE_SHA
            and package_record['runtime_result_sha256'] == args.runtime_sha256
            and package_record['reviewed_source_sha256'] == args.source_sha256, 'same_successful_package_and_runtime')
    for path, digest in package_record['source_bindings'].items():
        bind(bindings, path, digest)
    archive = inspect_archive()
    require(all(package_record[k] == v for k, v in archive.items()), 'packaged_archive_changed')
    bind(bindings, ARCHIVE, package_record['archive_sha256'])
    bind(bindings, STAGE / 'estimator.py', CANDIDATE_SHA)
    require(not ATTEMPT.exists() and not RESPONSE.exists(), 'submission_attempt_already_exists_no_retry')
    unchanged(bindings)
    marker = dict(status='attempt_started', candidate_sha256=CANDIDATE_SHA,
        archive_sha256=archive['archive_sha256'], package_result_sha256=args.package_sha256,
        runtime_result_sha256=args.runtime_sha256, reviewed_source_sha256=args.source_sha256,
        source_bindings=bindings, challenge=CHALLENGE, description=DESCRIPTION,
        authorized_scope='Explicit user bank-confirmation exception for validated d50 component',
        maximum_wrapper_cli_invocations=1, cli_internal_HTTP_retries='Pinned Whest0.16: at most5 attempts/45s per logical request',
        no_automatic_wrapper_retry=True, official_score=False, started_utc=utc())
    write(ATTEMPT, marker)  # Before any key access or external call; never removed.
    start = time.monotonic()
    result = dict(stage='submit', passed=False, completed=False, commands=[],
        candidate_sha256=CANDIDATE_SHA, package_result_sha256=args.package_sha256,
        runtime_result_sha256=args.runtime_sha256, attempt_sha256=sha(ATTEMPT),
        external_call_attempted=False, official_score=False, no_automatic_retry=True)
    env, key = None, None
    try:
        key = KEY_FILE.read_text().strip()
        require(bool(key) and '\n' not in key and '\r' not in key, 'existing_key_unavailable')
        env = dict(os.environ)
        env['AICROWD_API_KEY'] = key
        unchanged(bindings)
        result['external_call_attempted'] = True
        response = run_cli(['submit', str(ARCHIVE), '--challenge', CHALLENGE,
                            '--description', DESCRIPTION, '--json'], env, 180, result, 'submit', secret=key)
        require(type(response.get('submission_id')) is int and response['submission_id'] > 0,
                'submission_identity_missing_do_not_retry')
        result.update(submission_id=response['submission_id'],
            tracking_url=f'https://www.aicrowd.com/challenges/{CHALLENGE}/submissions/{response["submission_id"]}',
            submitted=True)
        unchanged(bindings)
        result.update(passed=True, completed=True, hashes_unchanged=True)
    except BaseException as exc:
        result['error'] = safe_failure(exc)
        result['submission_status_may_be_unknown'] = result['external_call_attempted']
    finally:
        if env is not None:
            env.pop('AICROWD_API_KEY', None)
        key = None
    result.update(elapsed_s=time.monotonic()-start, ended_utc=utc())
    write(RESPONSE, result)
    print(json.dumps(dict(stage='submit', passed=result['passed'], submission_id=result.get('submission_id'),
                         receipt=str(RESPONSE), receipt_sha256=sha(RESPONSE), official_score=False,
                         automatic_retry=False)), flush=True)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--package', action='store_true')
    stages.add_argument('--submit', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--runtime-sha256')
    parser.add_argument('--package-sha256', help='SHA of successful package_result.json; submit only')
    args = parser.parse_args()
    if not (args.package or args.submit):
        parser.print_help()
        return 0
    try:
        environment()
        bindings = verify_runtime(args)
        return package(args, bindings) if args.package else submit(args, bindings)
    except BaseException as exc:
        print(json.dumps(dict(passed=False, preflight_error=safe_failure(exc),
                              automatic_retry=False, official_score=False)), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
