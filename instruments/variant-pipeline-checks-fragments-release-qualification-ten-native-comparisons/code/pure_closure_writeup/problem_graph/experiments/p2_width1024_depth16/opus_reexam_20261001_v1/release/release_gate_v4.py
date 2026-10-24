"""Version 4 (release lane, 2026-10-11 18:15 UTC): release_gate_v3 + the lead's S48 override. Read-only; no sessions, submissions or
credentials. With --first-priced the fresh judge recompute runs `designation_judge.py --first-priced`: the first-network request cap becomes
a priced risk line (P(loss) per 100 networks from loss_model_v3, first networks priced separately, <= 10%); the warm cap and every other
gate stay as in v3 (exact hashes, archive members, every stage A / C gate; this machine's swap / paging only warn). Versions 1-3 are kept.
usage: release_gate_v4.py <tag> --stage-a DIR --pair DIR --slow DIR [--judge FILE] [--allow-missing-worker] [--primary | --first-priced]
exit 0 = fire-ready (release-only when stage D is skipped), 1 = blocked."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import release_gate_v1 as raw
import release_gate_v2 as v2

R = Path(__file__).resolve().parent
WARN_ONLY = ("gate swap is", "gate clean is")


def audit(stage_a, pair, slow, tag, worker, allow_missing_worker, extra, judge):
    stage_a, pair, slow = [p.resolve() for p in (stage_a, pair, slow)]
    worker_path = worker.resolve() if worker else R/'runs/_missing_worker_for_gate_v2'
    judge_source = R/'designation_judge.py'
    inputs = [stage_a/'stage_a.json', pair/'stage_c.json', slow/'stage_c.json', judge_source]
    if worker is not None:
        inputs.append(worker_path/'stage_d.json')
    fingerprints = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    before = fingerprints()
    command = [sys.executable, '-B', str(judge_source), tag, '--pair', str(pair), '--slow', str(slow), '--worker', str(worker_path)] + extra
    result = subprocess.run(command, cwd=R, capture_output=True, text=True, timeout=120, check=True)
    after = fingerprints()
    checked = raw.audit(stage_a, pair, slow, v2.FreshReport(result.stdout), tag, worker_path if worker is not None else None, allow_missing_worker)
    checked['errors'].extend(v2.report_errors(judge.read_text() if judge else None, result.stdout))
    if before != after:
        checked['errors'].append('evidence changed during numeric judgment; rerun on immutable artifacts')
    checked['numeric_judge'] = {'recomputed': True, 'command': command, 'stdout': result.stdout, 'evidence_sha256': after}
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag')
    p.add_argument('--stage-a', type=Path, required=True)
    p.add_argument('--pair', type=Path, required=True)
    p.add_argument('--slow', type=Path, required=True)
    p.add_argument('--judge', type=Path)
    p.add_argument('--worker', type=Path)
    p.add_argument('--allow-missing-worker', action='store_true')
    p.add_argument('--primary', action='store_true')
    p.add_argument('--first-priced', action='store_true')
    a = p.parse_args()
    extra = ['--primary'] if a.primary else ['--first-priced'] if a.first_priced else []
    try:
        result = audit(a.stage_a, a.pair, a.slow, a.tag, a.worker, a.allow_missing_worker, extra, a.judge)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, subprocess.SubprocessError) as exc:
        result = {'passed': False, 'errors': [f'incomplete or invalid evidence: {type(exc).__name__}: {exc}']}
    errors = result.get('errors', [])
    result['warnings'] = [e for e in errors if any(w in e for w in WARN_ONLY)]
    result['errors'] = [e for e in errors if not any(w in e for w in WARN_ONLY)]
    if result['warnings'] and result['errors'] == ['judge: missing, mismatched, failed, spoiled, or incomplete verdict']:
        body = (result.get('numeric_judge') or {}).get('stdout', '')
        if '  FAIL ' not in body and 'no result for:' not in body.replace('no result for: worker', ''):
            result['warnings'].append(result['errors'].pop())
    result['passed'] = not result['errors']
    result['gate_version'] = 4
    result['judge_mode'] = extra
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
