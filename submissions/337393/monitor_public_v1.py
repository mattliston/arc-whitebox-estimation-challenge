"""Anonymous, read-only grading monitor for the two explicitly assigned submissions."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import requests

HERE = Path(__file__).resolve().parent
O = HERE.parents[4]
reader = O / 'fire/read_eval_summary_astra_v1.py'
assert reader.exists(), reader
spec = importlib.util.spec_from_file_location('existing_public_reader', reader)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
p = argparse.ArgumentParser()
p.add_argument('--count', type=int, default=1)
p.add_argument('--interval', type=float, default=45)
a = p.parse_args()
assert 1 <= a.count <= 100 and a.interval >= 30
done = set()
for turn in range(a.count):
    for sid in (337421, 337422):
        if sid in done:
            continue
        utc = subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
        try:
            base = f'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{sid}'
            response = requests.get(base, params={'arc_audit': utc}, timeout=25,
                                    headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'})
            response.raise_for_status()
            page = module.Page()
            page.feed(response.content.decode('utf-8'))
            assert len(page.records) == 1
            submission = page.records[0]
            evaluation = submission.get('evaluation')
            record = dict(submission_id=sid, observed_utc=utc, url=response.url,
                          submission_status=submission.get('status'), submission_score=submission.get('score'),
                          evaluation=evaluation)
            out = HERE / ('public_monitor_' + utc.replace('-', '').replace(':', ''))
            out.mkdir(exist_ok=True)
            path = out / f'eval_{sid}_raw.json'
            with path.open('x') as f:
                json.dump(record, f, indent=2)
                f.write('\n')
            ev = evaluation or {}
            result = ev.get('results') or {}
            rows = result.get('per_mlp') or []
            counts = dict(Counter(row.get('state') for row in rows))
            failures = [{k: row.get(k) for k in ('mlp_index', 'state', 'error_code')}
                        for row in rows if row.get('state') == 'failed'
                        or row.get('error_code') not in (None, '', 'SUCCESS')]
            summary = dict(submission_id=sid, observed_utc=utc, receipt=str(path),
                submission_status=submission.get('status'), submission_score=submission.get('score'),
                state=ev.get('state'), phase=ev.get('phase'), updated_at=ev.get('updated_at'),
                completed_at=ev.get('completed_at'), counts=counts, progress=ev.get('progress'),
                setup_codes=[x.get('error_code') for x in result.get('setup_executions', [])],
                error=ev.get('error'), failures=failures)
            print(json.dumps(summary), flush=True)
            if submission.get('status') == 'GRADED':
                done.add(sid)
        except Exception as exc:
            print(json.dumps(dict(submission_id=sid, observed_utc=utc, monitor_error=repr(exc))), flush=True)
    if len(done) == 2:
        break
    if turn + 1 < a.count:
        time.sleep(a.interval)
