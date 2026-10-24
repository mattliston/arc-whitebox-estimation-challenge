#!/usr/bin/env python3
"""Read public submission pages into resumable JSONL, using Python's stdlib.

Usage: scan_submissions.py START END OUT.jsonl
       scan_submissions.py --ids ID,ID,... OUT.jsonl
Only explicit IDs are fetched. No authentication or account actions are used.
"""
import argparse
import concurrent.futures
from html.parser import HTMLParser
import json
from pathlib import Path
import statistics
import urllib.request

URL = 'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{}'

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.submissions = []
    def handle_starttag(self, tag, attrs):
        value = dict(attrs).get('data-react-props')
        if value:
            try:
                s = json.loads(value).get('props', {}).get('data', {}).get('submission')
                if s:
                    self.submissions.append(s)
            except (ValueError, AttributeError):
                pass

def fetch(sid):
    rec = dict(id=sid)
    try:
        req = urllib.request.Request(URL.format(sid), headers={'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=35) as response:
            rec['http'] = response.status
            p = Page()
            p.feed(response.read().decode('utf-8', 'replace'))
        if len(p.submissions) != 1:
            raise ValueError('page did not contain exactly one submission')
        s = p.submissions[0]
        rec.update(who=(s.get('participant') or {}).get('name'), created=s.get('createdAt'), status=s.get('status'), score=s.get('score'), score2=s.get('scoreSecondary'))
        result = (s.get('evaluation') or {}).get('results') or {}
        agg = (result.get('aggregates') or {}).get('public') or {}
        rec['agg'] = {k: v for k, v in agg.items() if not isinstance(v, (dict, list))}
        per = result.get('per_mlp') or []
        tel = [m['telemetry'] for m in per if m.get('telemetry')]
        rec['n'] = len(tel)
        for key, name in [('participant_predict_wall_s', 'wall'), ('residual_wall_time_s', 'resid'), ('participant_roundtrip_wall_s', 'rt')]:
            values = [t[key] for t in tel if t.get(key) is not None]
            if values:
                rec[name] = [statistics.mean(values), max(values)]
        rec['failed'] = [(m.get('mlp_index'), m.get('error_code')) for m in per if m.get('error_code')]
    except Exception as e:
        rec['err'] = repr(e)
    return rec

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids')
    parser.add_argument('args', nargs='+')
    args = parser.parse_args()
    if args.ids:
        ids = sorted(set(int(x) for x in args.ids.split(',')))
        out = Path(args.args[0])
    else:
        first, last, output = args.args
        ids, out = range(int(first), int(last) + 1), Path(output)
    done = set()
    if out.exists():
        done = {r['id'] for r in (json.loads(l) for l in out.read_text().splitlines()) if not r.get('err')}
    pending = [sid for sid in ids if sid not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a') as f, concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for rec in pool.map(fetch, pending):
            f.write(json.dumps(rec) + '\n')
            f.flush()
            print(rec['id'], rec.get('who'), rec.get('status'), rec.get('err', ''), flush=True)
    print('SCAN_DONE', len(pending), flush=True)

if __name__ == '__main__':
    main()
