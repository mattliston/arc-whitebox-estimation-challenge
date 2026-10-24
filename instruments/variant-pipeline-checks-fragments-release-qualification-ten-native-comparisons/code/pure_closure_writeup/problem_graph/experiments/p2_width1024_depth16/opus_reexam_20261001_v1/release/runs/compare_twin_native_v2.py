"""TAG opus-release v2 (2026-10-12 04:50 UTC: the base session is an argument, so an S50 twin is judged against the S50 safe file, not S48): the twin's native ten networks (quick.py, list a) against the fired S48 file's own native session on the same networks
(engineer logs/quick_keys_s48rc_a.json = m_sd_rc 5fd5a52e): final-layer MSE change, bill change, charged ops, largest prediction difference.
Writes ONE TEAM_LOG line (verdict for the lead's fire: predictions at rounding level, bill and requests as the table said)."""
import json, time, sys, hashlib
from pathlib import Path
TAG = sys.argv[1]
BASE = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'posthoc' else None   # a quick.py json; its preds next to it as <stem>_pred_<rank>.npy
import numpy as np
R = Path(__file__).resolve().parents[1]; O = R.parent; E = O/'engineer'
BJ = ((BASE if BASE.is_absolute() else R/BASE) if BASE else E/'logs'/'quick_keys_s48rc_a.json').resolve()
tw = json.loads((R/'runs'/f'quick_{TAG}_a.json').read_text()); base = json.loads(BJ.read_text())
T = {r['rank']: r for r in tw['rows']}; B = {r['rank']: r for r in base['rows']}
ranks = [r['rank'] for r in base['rows'] if r['rank'] in T]
dm, db, do, dp = [], [], [], []
for k in ranks:
    t, b = T[k], B[k]
    dm.append(100*(t['mse']/b['mse'] - 1)); db.append(100*(t['flops']/b['flops'] - 1)); do.append(t['ops'] - b['ops'])
    pt = np.load(R/'runs'/f'quick_{TAG}_a_pred_{k}.npy'); pb = np.load(BJ.with_name(BJ.stem + f'_pred_{k}.npy'))
    dp.append(float(np.max(np.abs(pt[15] - pb[15]))/np.max(np.abs(pb[15]))))
pool = 100*(sum(T[k]['mse'] for k in ranks)/sum(B[k]['mse'] for k in ranks) - 1)
mean = sum(dm)/len(dm); se = (sum((x - mean)**2 for x in dm)/(len(dm) - 1)/len(dm))**0.5
gate_lines = [l for l in (O/'TEAM_LOG.md').read_text().splitlines() if f'{TAG} SAFE GATE PASS' in l]
fire_cmd = ('FIRE COMMAND' + gate_lines[-1].split('FIRE COMMAND', 1)[1]) if gate_lines and 'FIRE COMMAND' in gate_lines[-1] else 'no gate PASS line found: DO NOT FIRE'
ok_gate = bool(gate_lines)
ok = abs(pool) < 0.5 and max(dp) < 1e-2 and sum(db)/len(db) < 0 and ok_gate
SHA = hashlib.sha256((R/'variants'/f'{TAG}_safe'/'candidate.py').read_bytes()).hexdigest()[:8]
line = (f"[release] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {'(POST-HOC: the pipeline run crashed before writing this line) ' if len(sys.argv) > 2 and sys.argv[2] == 'posthoc' else ''}TWIN {TAG} ({SHA}), NATIVE TEN NETWORKS (list a, quick.py in one process, "
        f"O/release/runs/quick_{TAG}_a.json) against the base file's own native session ({BJ.name}): final-layer MSE "
        f"pooled {pool:+.2f}% (mean of per-network changes {mean:+.2f}%, se {se:.2f}; lower on {sum(1 for x in dm if x < 0)} of {len(dm)}); per network "
        + ' '.join(f'{x:+.2f}' for x in dm) + f" %; largest final-layer prediction difference {max(dp):.1e} of the prediction scale (rounding level); bill "
        f"{sum(db)/len(db):+.3f}% mean ({min(db):+.3f}..{max(db):+.3f}%); charged ops {sum(do)/len(do):+.0f} mean ({min(do):+d}..{max(do):+d}). "
        + (("VERDICT: predictions at rounding level and the bill cut as priced: GO. " + fire_cmd) if ok else
           "VERDICT: OUTSIDE what the table promised (see the numbers): DO NOT FIRE the twin until we look."))
with open(O/'TEAM_LOG.md', 'a') as fh:
    fh.write(line + '\n')
print(line)
