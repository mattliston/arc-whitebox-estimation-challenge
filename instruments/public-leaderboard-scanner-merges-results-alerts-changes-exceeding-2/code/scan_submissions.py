"""TAG opus-reexam board scan: read PUBLIC submission pages of this challenge by id and store compact per-submission records
(participant, created, score, raw, all-layers, util, flops, walls, per-public-MLP scores). Read-only, sequential, polite.
usage: scan_submissions.py START END OUT.jsonl"""
import sys, json, time, requests, statistics as st
from html.parser import HTMLParser
start, end, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
class Page(HTMLParser):
    def __init__(self): super().__init__(); self.rec = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'data-react-props' in a:
            try: s = json.loads(a['data-react-props']).get('props', {}).get('data', {}).get('submission')
            except Exception: s = None
            if s: self.rec.append(s)
done = set()
try:
    for line in open(out): done.add(json.loads(line)['id'])
except FileNotFoundError: pass
sess = requests.Session()
with open(out, 'a') as f:
    for sid in range(start, end + 1):
        if sid in done: continue
        url = f'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{sid}'
        rec = dict(id=sid)
        try:
            r = sess.get(url, timeout=40, headers={'Cache-Control': 'no-cache'}, allow_redirects=False)
            rec['http'] = r.status_code
            if r.status_code == 200:
                p = Page(); p.feed(r.content.decode('utf-8', 'replace'))
                if len(p.rec) == 1:
                    s = p.rec[0]; ev = s.get('evaluation') or {}
                    rec.update(who=(s.get('participant') or {}).get('name'), team=(s.get('team') or {}).get('name') if s.get('team') else None,
                               created=s.get('createdAt'), status=s.get('status'), score=s.get('score'), score2=s.get('scoreSecondary'))
                    res = ev.get('results') or {}
                    agg = (res.get('aggregates') or {}).get('public') or {}
                    rec['agg'] = {k: v for k, v in agg.items() if not isinstance(v, (dict, list))}
                    per = res.get('per_mlp') or []
                    tel = [m['telemetry'] for m in per if m.get('telemetry')]
                    if tel:
                        def col(k): return [t[k] for t in tel if t.get(k) is not None]
                        fl = col('flops_used')
                        rec['n'] = len(tel); rec['flops'] = fl
                        for k, name in (('participant_predict_wall_s', 'wall'), ('flopscope_server_kernel_wall_s', 'kern'), ('flopscope_server_overhead_wall_s', 'ovh'), ('residual_wall_time_s', 'resid'), ('participant_roundtrip_wall_s', 'rt')):
                            v = col(k)
                            if v: rec[name] = [round(st.mean(v), 3), round(max(v), 3)]
                    rec['setup'] = [(e.get('flops'), e.get('wall_s'), e.get('error_code')) for e in (res.get('setup_executions') or [])]
                    rec['pub'] = [(m['mlp_index'], m['scores'].get('final_layer_mse'), m['scores'].get('all_layers_mse')) for m in (res.get('public_scores') or [])]
                    rec['failed'] = [(m.get('mlp_index'), m.get('error_code')) for m in per if m.get('error_code')]
        except Exception as e:
            rec['err'] = repr(e)[:200]
        f.write(json.dumps(rec) + '\n'); f.flush()
        time.sleep(0.15)
print('SCAN_DONE', start, end)
