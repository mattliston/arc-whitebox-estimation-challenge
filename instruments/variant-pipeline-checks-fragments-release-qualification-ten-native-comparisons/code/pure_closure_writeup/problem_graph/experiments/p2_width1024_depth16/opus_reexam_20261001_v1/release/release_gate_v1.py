"""Read-only pre-fire audit. Never submits, starts a session, or reads credentials.

Versioned replacement for release_chain42's permissive grep check. Stage D can
only be omitted with --allow-missing-worker; that yields a release-only verdict,
never a completed designation verdict. Run after designation_judge.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


A_GATES = {'contract', 'package', 'validate_package', 'archive_contents',
           'no_numpy_import', 'einsum_signatures_proven',
           'einsum_no_new_repeated_operand', 'python310'}
C_GATES = {'memory', 'swap', 'nonfinite', 'clean', 'setup_ok', 'residual', 'wall'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def gate_errors(label, document, required, sha):
    errors = []
    if document.get('candidate_sha256') != sha:
        errors.append(f'{label}: candidate SHA mismatch')
    gates = document.get('gates') or {}
    for key in sorted(required | set(gates)):
        if gates.get(key) is not True:
            context = (' (local contention: remeasure; not evidence of a platform memory failure)'
                       if key in {'swap', 'clean'} else
                       ' (measured per-session footprint cap failed)' if key == 'memory' else '')
            errors.append(f'{label}: gate {key} is {gates.get(key)!r}{context}')
    if document.get('passed') is not True:
        errors.append(f'{label}: passed is not true')
    return errors


def judge_errors(body, tag, allow_missing_worker):
    errors = []
    if re.search(r'^\s*FAIL\b', body, re.M):
        errors.append('judge: one or more numeric criteria failed')
    prefix = f'[designation {tag}] '
    verdicts = [line for line in body.splitlines() if line.startswith('[designation ')]
    accepted = {prefix + 'PASS: every criterion met'}
    if allow_missing_worker:
        accepted.add(prefix + 'INCOMPLETE: no result for: worker')
    if len(verdicts) != 1 or verdicts[0] not in accepted:
        errors.append('judge: missing, mismatched, failed, spoiled, or incomplete verdict')
    return errors


def audit(stage_a, pair, slow, judge, tag, worker=None, allow_missing_worker=False):
    a = json.loads((stage_a / 'stage_a.json').read_text())
    source = Path(a['candidate'])
    sha = digest(source.read_bytes())
    errors = gate_errors('stage A', a, A_GATES, sha)
    archive = stage_a / 'submission.tar.gz'
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if sorted(m.name for m in members) != ['estimator.py', 'manifest.json'] or not all(m.isfile() for m in members):
            errors.append('archive: expected exactly two regular files, estimator.py and manifest.json')
        else:
            if digest(tar.extractfile('estimator.py').read()) != sha:
                errors.append('archive: estimator.py SHA mismatch')
            manifest = json.loads(tar.extractfile('manifest.json').read())
            if manifest.get('entrypoint') != {'module': 'estimator', 'class': 'Estimator'}:
                errors.append('archive: wrong entrypoint')
    if digest((stage_a / 'payload/estimator.py').read_bytes()) != sha:
        errors.append('stage A payload: estimator.py SHA mismatch')
    for label, path, extra in [('stage C', pair, {'binding'}), ('slow setup', slow, set())]:
        d = json.loads((path / 'stage_c.json').read_text())
        errors.extend(gate_errors(label, d, C_GATES | extra, sha))
        if d.get('projection') is None or not d.get('sessions', {}).get('C'):
            errors.append(f'{label}: projection or candidate session missing')
        if d.get('ranks') != '35,100':
            errors.append(f'{label}: expected both boundary networks 35,100')
        if label == 'slow setup' and d.get('stress_slowdown') != 1.3:
            errors.append('slow setup: expected 1.3x stress')
    if worker is not None:
        d = json.loads((worker / 'stage_d.json').read_text())
        errors.extend(gate_errors('stage D', d, {'finite', 'memory', 'native_parity'}, sha))
    elif not allow_missing_worker:
        errors.append('stage D: missing; explicit --allow-missing-worker required for release-only qualification')
    body = judge.read_text()
    errors.extend(judge_errors(body, tag, allow_missing_worker and worker is None))
    if worker is None and 'PASS: every criterion met' in body:
        errors.append('judge: PASS cannot be bound to an unspecified worker result')
    return {'passed': not errors, 'scope': 'release only; stage D NOT RUN' if worker is None else 'release and designation gates',
            'candidate_sha256': sha, 'archive': str(archive.resolve()), 'archive_sha256': digest(archive.read_bytes()),
            'errors': errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag')
    p.add_argument('--stage-a', type=Path, required=True)
    p.add_argument('--pair', type=Path, required=True)
    p.add_argument('--slow', type=Path, required=True)
    p.add_argument('--judge', type=Path, required=True)
    p.add_argument('--worker', type=Path)
    p.add_argument('--allow-missing-worker', action='store_true')
    args = p.parse_args()
    try:
        result = audit(args.stage_a, args.pair, args.slow, args.judge, args.tag,
                       args.worker, args.allow_missing_worker)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        result = {'passed': False, 'errors': [f'incomplete or invalid evidence: {type(exc).__name__}: {exc}']}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
