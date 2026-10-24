"""TAG fable-fire (FF-3): read a submission's public evaluation JSON (per-MLP platform telemetry) from its AIcrowd page; never runs
page code. usage: read_eval_page.py <submission_id>  -> fable_fire_v1/eval_<sid>.json + a wall summary on stdout."""
import sys, json, requests
from html.parser import HTMLParser
sid = int(sys.argv[1])
url = f'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{sid}'
body = requests.get(url, timeout=30, headers={'Cache-Control': 'no-cache'}).content.decode('utf-8')
class Page(HTMLParser):
    def __init__(self): super().__init__(); self.rec = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'data-react-props' in a:
            s = json.loads(a['data-react-props']).get('props', {}).get('data', {}).get('submission')
            if s: self.rec.append(s)
p = Page(); p.feed(body); assert len(p.rec) == 1
ev = p.rec[0].get('evaluation')
json.dump(dict(submission_id=sid, url=url, evaluation=ev), open(f'eval_{sid}.json', 'w'))
per = ev['results']['per_mlp']
def col(k): return [m['telemetry'][k] for m in per if m.get('telemetry')]
for k in ('participant_predict_wall_s', 'flopscope_server_kernel_wall_s', 'flopscope_server_overhead_wall_s', 'residual_wall_time_s'):
    v = sorted(col(k)); print(f'{k:36s} mean {sum(v)/len(v):7.2f}  median {v[len(v)//2]:7.2f}  max {v[-1]:7.2f}  n {len(v)}')
agg = ev['results']['aggregates']['public']; print({k: agg[k] for k in ('score', 'final_layer_mse', 'mean_compute_utilization', 'n_failed_mlps')})
