"""[fable-lead 2026-10-19] paired per-rank comparison of two science32 runs graded against the same 95c control.
usage: compare32_v1.py <run A dir> <run B dir>   (prints B - A per rank in raw/bill/adjusted %, heldout25 and all32 means)"""
import json, math, statistics as st, sys
from pathlib import Path
A, B = Path(sys.argv[1]), Path(sys.argv[2])
def grade(d):
    out = {}
    for name in ('candidate_all32', 'candidate7', 'candidate_heldout25'):
        p = d / name / 'GRADE_STORE_v1.json'
        if p.exists():
            for row in json.load(open(p)):
                out[int(row['rank'])] = row
    return out
ga, gb = grade(A), grade(B)
ranks = sorted(set(ga) & set(gb))
print('%d common ranks' % len(ranks))
seven = [35, 150, 200, 250, 259, 277, 287]
rows = []
for r in ranks:
    a, b = ga[r], gb[r]
    d_adj = b['adjusted_pct'] - a['adjusted_pct']; d_raw = b['raw_pct'] - a['raw_pct']; d_bill = b['bill_pct'] - a['bill_pct']
    rows.append((r, d_raw, d_bill, d_adj))
    print('rank %4d  A adj %+.3f  B adj %+.3f  | B-A raw %+.3f bill %+.3f adj %+.3f' % (r, a['adjusted_pct'], b['adjusted_pct'], d_raw, d_bill, d_adj))
def summ(label, sel):
    if not sel: return
    adj = [x[3] for x in sel]; raw = [x[1] for x in sel]; bill = [x[2] for x in sel]
    se = st.stdev(adj) / math.sqrt(len(adj)) if len(adj) > 1 else float('nan')
    print('%-10s n=%2d  B-A adjusted %+.3f%% (se %.3f, lower %d/%d)  raw %+.3f%%  bill %+.3f%%' % (label, len(sel), st.mean(adj), se, sum(x < 0 for x in adj), len(adj), st.mean(raw), st.mean(bill)))
summ('seven', [x for x in rows if x[0] in seven]); summ('heldout25', [x for x in rows if x[0] not in seven]); summ('all32', rows)
for name in ('RESULT32_v1.json',):
    for d in (A, B):
        p = d / name
        if p.exists():
            j = json.load(open(p)); print(d.name, {k: (round(v['adjusted_pct'], 3), round(v['adjusted_se'], 3), v['lower'], round(v['projected_adjusted_score'] * 1e9, 5)) for k, v in j.items() if isinstance(v, dict) and 'adjusted_pct' in v})
