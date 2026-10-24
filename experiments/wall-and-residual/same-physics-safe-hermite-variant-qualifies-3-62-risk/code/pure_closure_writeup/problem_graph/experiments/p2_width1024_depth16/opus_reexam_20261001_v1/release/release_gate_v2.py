"""Version 2: recompute the numeric judge from the exact audited evidence.

Read-only; no sessions, submissions or credentials. A supplied --judge report is
optional and must exactly match the recomputed stdout. Version 1 is preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import release_gate_v1 as raw


R = Path(__file__).resolve().parent


class FreshReport:
    def __init__(self, body):
        self.body = body

    def read_text(self):
        return self.body


def report_errors(expected, actual):
    return [] if expected is None or expected == actual else [
        'supplied judge report does not exactly match a fresh judge of these paths and mode']


def audit(stage_a, pair, slow, tag, worker=None, allow_missing_worker=False,
          primary=False, judge=None):
    stage_a, pair, slow = [p.resolve() for p in (stage_a, pair, slow)]
    worker_path = worker.resolve() if worker else R / 'runs/_missing_worker_for_gate_v2'
    if worker is None and (worker_path / 'stage_d.json').exists():
        raise ValueError('missing-worker sentinel unexpectedly contains evidence')
    judge_source = R / 'designation_judge.py'
    inputs = [stage_a / 'stage_a.json', pair / 'stage_c.json', slow / 'stage_c.json', judge_source]
    if worker is not None:
        inputs.append(worker_path / 'stage_d.json')
    fingerprints = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    before = fingerprints()
    command = [sys.executable, '-B', str(judge_source), tag, '--pair', str(pair),
               '--slow', str(slow), '--worker', str(worker_path)] + (['--primary'] if primary else [])
    result = subprocess.run(command, cwd=R, capture_output=True, text=True, timeout=60, check=True)
    after = fingerprints()
    checked = raw.audit(stage_a, pair, slow, FreshReport(result.stdout), tag,
                        worker_path if worker is not None else None, allow_missing_worker)
    checked['errors'].extend(report_errors(judge.read_text() if judge else None, result.stdout))
    if before != after:
        checked['errors'].append('evidence changed during numeric judgment; rerun on immutable artifacts')
    checked['passed'] = not checked['errors']
    checked['numeric_judge'] = {
        'recomputed': True, 'command': command, 'primary': primary,
        'stdout': result.stdout, 'stdout_sha256': raw.digest(result.stdout.encode()),
        'evidence_sha256': after,
        'supplied_report': str(judge.resolve()) if judge else None,
    }
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag')
    p.add_argument('--stage-a', type=Path, required=True)
    p.add_argument('--pair', type=Path, required=True)
    p.add_argument('--slow', type=Path, required=True)
    p.add_argument('--judge', type=Path, help='optional old report; must equal recomputed stdout exactly')
    p.add_argument('--worker', type=Path)
    p.add_argument('--allow-missing-worker', action='store_true')
    p.add_argument('--primary', action='store_true')
    a = p.parse_args()
    try:
        result = audit(a.stage_a, a.pair, a.slow, a.tag, a.worker,
                       a.allow_missing_worker, a.primary, a.judge)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError,
            subprocess.SubprocessError) as exc:
        result = {'passed': False, 'errors': [f'incomplete or invalid evidence: {type(exc).__name__}: {exc}']}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
