# [opus-gapclass] load per-network public telemetry for the 4 files (ours 336412 + 3 leaders)
import json, glob, numpy as np
A = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst'
IDS = {'us': 336412, 'marius': 336419, 'j2w': 336188, 'oliver': 336473}
def load(ids=IDS):
    recs = {}
    for f in [A + '/scan_merged.jsonl'] + sorted(glob.glob(A + '/scan_33*_plus*.jsonl')):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get('status') == 'GRADED' and 'pub' in r: recs[r['id']] = r
    out = {}
    for k, i in ids.items():
        r = recs[i]; p = sorted(r['pub'])
        out[k] = dict(idx=np.array([q[0] for q in p]), fin=np.array([q[1] for q in p]), allm=np.array([q[2] for q in p]),
                      flops=np.array(r['flops'], float), score=r['score'], util=r['agg']['mean_compute_utilization'])
    return out, recs
