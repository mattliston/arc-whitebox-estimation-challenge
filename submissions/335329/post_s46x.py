"""TAG opus-release: after s47_pair.sh s46x, one TEAM_LOG line: the twin's bill cut against the safe file, its P(loss) and the pair gain."""
import json, time
from pathlib import Path
R = Path(__file__).resolve().parents[1]; O = R.parent
rows = lambda tag: [json.loads(x) for x in (R/'runs'/f'c_{tag}'/'C0'/'rows.jsonl').read_text().splitlines()][:2]
s, t = rows('s46x_safe'), rows('s46x_twin')
cut = [100*(1 - b['flops']/a['flops']) for a, b in zip(s, t)]
req_s = sum(r['requests'] for r in s)/2; req_t = sum(r['requests'] for r in t)/2
PL = json.loads((O/'analyst'/'a19_page_list.json').read_text())
E = {'local': [v[1] for v in PL['excess'].values()], 'week': [v[0] for v in PL['excess'].values()]}
def p(req, price, usual, rd):
    head = 0.4/(req*price*usual); return sum(1 for x in E[rd] if x > head)/len(E[rd])
cells = []
for name, price, usual in (('median hour', 7.161e-6, 1.137), ('worst hour', 7.267e-6, 1.183)):
    pl, pw = p(req_t, price, usual, 'local'), p(req_t, price, usual, 'week')
    c = sum(cut)/2
    cells.append(f"{name}: P(loss) {100*pl:.1f}-{100*pw:.1f}% per 100 networks, pair gain {c*(1 - pw):.3f}..{c*(1 - pl):.3f}%")
safe_p = p(req_s, 7.161e-6, 1.137, 'local'), p(req_s, 7.161e-6, 1.137, 'week')
gate = json.loads((R/'runs'/'gate_s46x_twin.json').read_text()) if (R/'runs'/'gate_s46x_twin.json').exists() else {'passed': False, 'errors': ['no gate output']}
arch = R/'runs'/'a_s46x_twin'/'submission.tar.gz'
import hashlib
if gate.get('passed'):
    fire = (f"TWIN GATE PASS. FIRE COMMAND (lead): cd {O} && /Users/matt/arc-white-box-estimation/.venv/bin/python -B fire/fire_once.py "
            f"s46x_twin_20261011 {arch} 'S46x twin 8ba4c158: S46 physics + XRQ request diet + HZ7 seven-level strips + zero planner prices' "
            f"(archive sha256 {hashlib.sha256(arch.read_bytes()).hexdigest()}).")
else:
    fire = 'TWIN GATE BLOCKED, DO NOT FIRE: ' + '; '.join(gate.get('errors', []))[:600]
line = (f"[release] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} S46x TWIN PRICED (python -B O/release/runs/post_s46x.py; stage C "
        f"under the platform clock): SAFE 413eaaa3 {s[0]['requests']:,} / {s[1]['requests']:,} requests (first / warm), P(loss) "
        f"{100*safe_p[0]:.1f}-{100*safe_p[1]:.1f}% at the median hour; TWIN 8ba4c158 {t[0]['requests']:,} / {t[1]['requests']:,}; twin bill "
        f"against safe on networks 35 / 100: -{cut[0]:.3f}% / -{cut[1]:.3f}%. " + '; '.join(cells) + '. ' + fire)
with open(O/'TEAM_LOG.md', 'a') as fh:
    fh.write(line + '\n')
print(line)
