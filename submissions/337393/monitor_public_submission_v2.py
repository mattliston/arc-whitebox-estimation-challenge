"""Anonymous read-only public grading monitor; never submits or promotes a reference."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import requests

HERE = Path(__file__).resolve().parent
O = HERE.parents[4]
spec = importlib.util.spec_from_file_location('existing_public_reader', O / 'fire/read_eval_summary_astra_v1.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def summarize(record):
    ev = record['evaluation'] or {}
    result = ev.get('results') or {}
    rows = result.get('per_mlp') or []
    setups = result.get('setup_executions') or []
    counts = dict(Counter(str(row.get('state')) for row in rows))
    failures = [{k: row.get(k) for k in ('mlp_index', 'state', 'error_code')}
                for row in rows if row.get('state') == 'failed'
                or row.get('error_code') not in (None, '', 'SUCCESS')]
    telemetry = {}
    for key in ('participant_predict_wall_s', 'flopscope_server_kernel_wall_s',
                'flopscope_server_overhead_wall_s', 'residual_wall_time_s',
                'participant_setup_wall_s', 'flops_used', 'effective_compute'):
        values = [(row.get('telemetry') or {}).get(key) for row in rows]
        values = [x for x in values if finite(x)]
        telemetry[key] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                              max=max(values) if values else None)
    exhausted = [{k: row.get(k) for k in ('mlp_index', 'state', 'error_code')}
                 for row in rows if any((row.get('telemetry') or {}).get(k) is True for k in
                 ('budget_exhausted', 'time_exhausted', 'residual_wall_time_exhausted', 'combined_budget_exhausted'))]
    score = record['submission_score']
    complete = (record['submission_status'] == 'GRADED' and ev.get('state') == 'completed'
                and len(rows) == 100 and len({r.get('mlp_index') for r in rows}) == 100
                and all(r.get('state') == 'completed' for r in rows) and not failures and not exhausted
                and len(setups) == 5 and all(x.get('error_code') == 'SUCCESS' for x in setups)
                and finite(score) and score > 0)
    return dict(submission_id=record['submission_id'], observed_utc=record['observed_utc'],
                submission_status=record['submission_status'], submission_score=score,
                state=ev.get('state'), phase=ev.get('phase'), updated_at=ev.get('updated_at'),
                completed_at=ev.get('completed_at'), progress=ev.get('progress'),
                counts=counts, failures=failures, exhausted=exhausted,
                setup_codes=[s.get('error_code') for s in setups], runtime=telemetry,
                error=ev.get('error'), verified_100_completed_zero_failure=complete,
                reference_mutated=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--submission-id', type=int, required=True)
    p.add_argument('--release-sha256', required=True)
    p.add_argument('--archive-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=160)
    p.add_argument('--interval', type=float, default=45)
    a = p.parse_args()
    assert a.submission_id > 0 and 1 <= a.count <= 480 and a.interval >= 30
    assert all(len(s) == 64 and all(c in '0123456789abcdef' for c in s)
               for s in (a.release_sha256, a.archive_sha256))
    a.out.mkdir(parents=True, exist_ok=False)
    write(a.out / 'MANIFEST_v1.json', dict(utc=utc(), submission_id=a.submission_id,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        release_sha256=a.release_sha256, archive_sha256=a.archive_sha256,
        anonymous_read_only=True, no_submit_or_pointer_mutation=True))
    for turn in range(a.count):
        stamp = utc()
        try:
            base = f'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{a.submission_id}'
            response = requests.get(base, params={'arc_audit': stamp}, timeout=25,
                headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'})
            response.raise_for_status()
            page = reader.Page()
            page.feed(response.content.decode('utf-8'))
            assert len(page.records) == 1
            submission = page.records[0]
            record = dict(submission_id=a.submission_id, observed_utc=stamp, url=response.url,
                submission_status=submission.get('status'), submission_score=submission.get('score'),
                evaluation=submission.get('evaluation'))
            raw = a.out / ('raw_' + stamp.replace('-', '').replace(':', '') + '.json')
            write(raw, record)
            summary = summarize(record)
            summary['raw_receipt'] = str(raw)
            summary['raw_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
            write(raw.with_name(raw.name.replace('raw_', 'summary_')), summary)
            print(json.dumps(summary, allow_nan=False), flush=True)
            if record['submission_status'] == 'GRADED':
                write(a.out / 'FINAL_v1.json', summary)
                break
        except Exception as exc:
            print(json.dumps(dict(submission_id=a.submission_id, observed_utc=stamp,
                                  monitor_error=repr(exc))), flush=True)
        if turn + 1 < a.count:
            time.sleep(a.interval)

if __name__ == '__main__':
    main()
