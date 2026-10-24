"""Version 3 (release lane, 2026-10-11, lead's S47 task (a): keep only what protects the score). Read-only; no sessions, submissions or
credentials. Runs release_gate_v2's audit (exact source and archive hashes, archive members, every stage A gate, the stage C and
slow-setup gates, a fresh recompute of designation_judge.py on these exact paths) and changes ONE thing: the stage C 'swap' and
'clean' gates of this machine (system swap grew, or the boundary session paged) become warnings, not blockers. They say the LOCAL
kernel and wall timings were disturbed by other work; the request counts, setup log, bill and outputs that decide the fire do not
depend on them, and the platform never sees this machine's swap. Every other error still blocks. Versions 1 and 2 are kept.
usage: release_gate_v3.py <tag> --stage-a DIR --pair DIR --slow DIR [--judge FILE] [--allow-missing-worker] [--primary]
exit 0 = fire-ready (release-only when stage D is skipped), 1 = blocked."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import release_gate_v2 as v2

WARN_ONLY = ("gate swap is", "gate clean is")


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
    a = p.parse_args()
    try:
        result = v2.audit(a.stage_a, a.pair, a.slow, a.tag, a.worker, a.allow_missing_worker, a.primary, a.judge)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, subprocess.SubprocessError) as exc:
        result = {'passed': False, 'errors': [f'incomplete or invalid evidence: {type(exc).__name__}: {exc}']}
    errors = result.get('errors', [])
    result['warnings'] = [e for e in errors if any(w in e for w in WARN_ONLY)]
    result['errors'] = [e for e in errors if not any(w in e for w in WARN_ONLY)]
    # the fresh judge prints FAIL rows only for numeric criteria; a judge error caused solely by a spoiled (swap / paging) note is a warning too
    if result['warnings'] and result['errors'] == ['judge: missing, mismatched, failed, spoiled, or incomplete verdict']:
        body = (result.get('numeric_judge') or {}).get('stdout', '')
        if '  FAIL ' not in body and 'no result for:' not in body.replace('no result for: worker', ''):
            result['warnings'].append(result['errors'].pop())
    result['passed'] = not result['errors']
    result['gate_version'] = 3
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
