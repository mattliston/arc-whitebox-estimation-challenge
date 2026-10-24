"""[opus-smtable] build: GOV_v3 0640c0cf + one appended dictionary-only fragment adding the grid shapes whose best Smirnov 3x6x3 plan
has a positive FIRST-call FLOP saving against GOV_v3's own incumbent read (PRICE_grid_v1_s*.json). Existing table entries are kept
as they are (validated plans); only new keys are added. Every plan's family is one of the five existing ones (family_for chose it).
Also writes a diagnostic copy (+ the lead's smcov coverage fragment, observation only) for the coverage screen.
usage: build_smtable_v1.py [min_first_saving_flops]"""
import hashlib, json, re, sys
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = O/'lead/runs/opus_smtable_v1'
base = (O/'lead/runs/opus_governor_v1/candidate_GOV_v3.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest().startswith('0640c0cf')
floor = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rows = []
for f in sorted(D.glob('PRICE_grid_v1_s*.json')):
    rows += json.loads(f.read_text())['rows']
ns = {}
exec(compile(re.search(r'^_ASTRA_SM40_PLANS = (\{.*\})$', base, re.M).group(0), 'plans', 'exec'), ns)
old = ns['_ASTRA_SM40_PLANS']
fam = {f[0] for f in eval(re.search(r'^_ASTRA_SM40_FAMILIES = (\(.*\))$', base, re.M).group(1))}
new, skipped_existing = {}, 0
for r in rows:
    b = r['best']; key = tuple(r['shape'])
    if not b or b['first_saving'] <= floor:
        continue
    if key in old:
        skipped_existing += 1; continue
    assert b['family'][0] in fam and b['P'] == b['family'][2]
    new[key] = (b['name'], b['L'], b['P'])
body = '_ASTRA_SM40_PLANS.update({' + ', '.join(f'{k!r}: {v!r}' for k, v in sorted(new.items())) + '})'
frag = (f"\n\n# BEGIN [opus-smtable] GRID PLANS v1 (append LAST on GOV_v3 0640c0cf; dictionary-only: {len(new)} new read keys, first-call FLOP\n"
        f"# saving > {floor} each vs the incumbent read, priced cost-only by price_grid_v1.py; existing table entries untouched)\n{body}\n# END [opus-smtable] GRID PLANS v1\n")
src = base + frag
out = D/'cand_GOV3_smtable_v1.py'; out.write_text(src)
diag = src + (O/'lead/runs/opus_smcov_v1/fragment_smcov_diag_v1.py.txt').read_text()
(D/'cand_GOV3_smtable_diag_v1.py').write_text(diag)
stats = dict(priced=len(rows), positive_first=sum(1 for r in rows if r['best'] and r['best']['first_saving'] > floor), new=len(new),
             skipped_existing=skipped_existing, sha=hashlib.sha256(src.encode()).hexdigest(), diag_sha=hashlib.sha256(diag.encode()).hexdigest(),
             fragment_bytes=len(frag))
(D/'BUILD_smtable_v1.json').write_text(json.dumps(stats, indent=1) + '\n'); print(stats)
