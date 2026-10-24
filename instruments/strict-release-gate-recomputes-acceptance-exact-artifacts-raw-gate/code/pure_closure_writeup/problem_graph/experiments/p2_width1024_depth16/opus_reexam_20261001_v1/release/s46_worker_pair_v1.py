"""Prepare/run exactly one guarded 20-network worker per fired S46 file.

Default is a read-only plan. Root must coordinate quiet load before --execute.
Never submits, retries, overwrites or changes any calibration reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import release_gate_v2


R = Path(__file__).resolve().parent
WORKER20 = [35, 100, 37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 76, 80, 84, 88, 92, 96, 104, 108]
FILES = [
    ('safe', 's46_safe_pz', 'a_s46_63f89e8d_safe_pz',
     '70f7857fdabbbca6398d44bb11dbf608268a7199c1f22e67d2e9ca31a514b540', False),
    ('twin', 's46_eprimary', 'a_s46e_f89b8951_primary',
     'f89b89511e62eb5474348bcbfbd7cfedce1a4bbd03de8efa5a9a4fa230266fe5', True),
]
D_GATES = {'finite', 'handles', 'memory', 'swap', 'binding', 'setup_ok', 'native_parity', 'clean', 'residual', 'wall'}


def source_and_plan(label, tag, atag, sha, primary, suffix):
    stage_a = R / 'runs' / atag
    source = Path(json.loads((stage_a / 'stage_a.json').read_text())['candidate']).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
        raise ValueError(f'{label}: fired source no longer matches pinned SHA')
    dtag = f'd_s46_{label}_{suffix}'
    command = [sys.executable, '-B', str(R / 'stage_d_serving.py'), str(source), dtag,
               '--short-setup', '--ranks', ','.join(map(str, WORKER20))]
    return dict(label=label, tag=tag, atag=atag, source=str(source), sha256=sha,
                primary=primary, dtag=dtag, command=command)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('suffix', help='unique run suffix, e.g. astra_20261011T010000Z')
    ap.add_argument('--execute', action='store_true', help='root-coordinated launch; otherwise print plan only')
    a = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+', a.suffix):
        raise ValueError('suffix must contain only letters, numbers and underscores')
    plan = [source_and_plan(*record, a.suffix) for record in FILES]
    out = R / 'runs' / f's46_worker_pair_{a.suffix}'
    destinations = [out] + [R / 'runs' / p['dtag'] for p in plan]
    if any(p.exists() for p in destinations):
        raise FileExistsError('new suffix required; an output directory already exists')
    print(json.dumps(dict(execute=a.execute, output=str(out), sessions=plan,
                         worker_guard_seconds=12900, native_parity_guard_seconds=1100,
                         resource_gates='7 GB own budget; free+file-backed >=4 GB; 30s flat swap',
                         session_guards='7.5 GB footprint, 10 GB RSS, +256 MiB swap growth, 70,000 handles'), indent=2), flush=True)
    if not a.execute:
        return
    # Revalidate all existing release evidence before either heavy job begins.
    for p in plan:
        existing = release_gate_v2.audit(R / 'runs' / p['atag'], R / 'runs' / ('c_' + p['tag']),
                   R / 'runs' / (p['tag'] + '_slow'), p['tag'],
                   allow_missing_worker=True, primary=p['primary'])
        if not existing['passed']:
            raise ValueError(f"{p['label']}: existing release gates failed: {existing['errors']}")
    out.mkdir()
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (out / 'launcher.pid').write_text(str(os.getpid()) + '\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', BLIS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for p in plan:
        if (R / 'runs' / p['dtag']).exists():
            raise FileExistsError(f"run appeared before launch: {p['dtag']}")
        with (out / (p['label'] + '_stage_d.log')).open('x') as log:
            child = subprocess.Popen(['/usr/bin/nice', '-n', '10', *p['command']],
                                     cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT)
            with (out / (p['label'] + '_stage_d.pid')).open('x') as pid:
                pid.write(str(child.pid) + '\n')
            print(f"{p['label']} stage D pid {child.pid}; log {log.name}", flush=True)
            code = child.wait()
        if code:
            raise SystemExit(f"{p['label']} stage D exited {code}; chain stopped, evidence preserved")
        worker = R / 'runs' / p['dtag']
        d = json.loads((worker / 'stage_d.json').read_text())
        if d.get('ranks') != WORKER20 or len(d.get('per_network', [])) != 20:
            raise ValueError(f"{p['label']}: worker did not cover exactly the required 20 networks")
        if any(d.get('gates', {}).get(k) is not True for k in D_GATES):
            raise ValueError(f"{p['label']}: stage D gate failed or absent; no next session")
        checked = release_gate_v2.audit(R / 'runs' / p['atag'], R / 'runs' / ('c_' + p['tag']),
                  R / 'runs' / (p['tag'] + '_slow'), p['tag'], worker=worker, primary=p['primary'])
        with (out / (p['label'] + '_complete_gate.json')).open('x') as report:
            json.dump(checked, report, indent=2)
            report.write('\n')
        if not checked['passed']:
            raise ValueError(f"{p['label']}: complete judge failed; no next session: {checked['errors']}")
        print(f"{p['label']}: full local designation gates PASS", flush=True)
    with (out / 'completed.json').open('x') as completed:
        json.dump({'passed': True, 'files': [p['sha256'] for p in plan]}, completed, indent=2)
        completed.write('\n')
    print('Both S46 files passed the full local designation gates; external designation is root’s step.', flush=True)


if __name__ == '__main__':
    main()
