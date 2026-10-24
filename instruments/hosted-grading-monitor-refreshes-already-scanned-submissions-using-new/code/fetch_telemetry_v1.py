"""Fetch one public submission page (read-only, no credentials) and dump per-MLP telemetry. usage: fetch_telemetry_v1.py SID OUT.json"""
import sys, json, requests
from pathlib import Path
from html.parser import HTMLParser
class Page(HTMLParser):
    def __init__(self): super().__init__(); self.rec = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'data-react-props' in a:
            try: s = json.loads(a['data-react-props']).get('props', {}).get('data', {}).get('submission')
            except Exception: s = None
            if s: self.rec.append(s)
sid = int(sys.argv[1]); url = f'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{sid}'
r = requests.get(url, timeout=40, headers={'Cache-Control': 'no-cache'}, allow_redirects=False)
p = Page(); p.feed(r.content.decode('utf-8', 'replace')); s = p.rec[0]
per = ((s.get('evaluation') or {}).get('results') or {}).get('per_mlp') or []
K = ('participant_predict_wall_s', 'flopscope_server_kernel_wall_s', 'flopscope_server_overhead_wall_s', 'residual_wall_time_s', 'flops_used', 'time_exhausted')
rows = [dict(idx=m['mlp_index'], name=m.get('mlp_name'), state=m.get('state'), err=m.get('error_code'), **{k: (m.get('telemetry') or {}).get(k) for k in K}) for m in per]
Path(sys.argv[2]).write_text(json.dumps(dict(sid=sid, status=s.get('status'), score=s.get('score'), created=s.get('createdAt'), updated=s.get('updatedAt'), rows=rows), indent=1))
print(json.dumps(dict(sid=sid, status=s.get('status'), n=len([x for x in rows if x['participant_predict_wall_s']]))))
