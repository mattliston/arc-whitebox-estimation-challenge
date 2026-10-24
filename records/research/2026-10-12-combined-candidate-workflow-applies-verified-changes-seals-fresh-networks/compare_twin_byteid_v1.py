"""TAG opus-release byte-identical confirmation v1 (2026-10-12 23:05 UTC; lead 22:57Z: s51k2 = s51k1tw + rows2 + npx1 is a BILL-ONLY stack whose
predictions must be BYTE-IDENTICAL to s51k1tw; no MSE seal). Judges the FINAL twin's native ten networks against the native ten of the file it
must equal (quick.py output + _pred_<rank>.npy, all 16 layers): CONFIRMED only when every network has the same pred sha, the same final MSE and
every one of the 16 layers' predictions equal bit for bit, and the gate PASS line exists; ANY difference = DO NOT FIRE. Requests / residual /
bill come from the deterministic count sessions against the LAST FIRED twin, as in compare v5 (native ops / flops are load-dependent and only
reported). Line shape as compare v5: a stack tag reads 'TWIN <tag> CONFIRMED (<sha8>), ...' or 'TWIN <tag> DO NOT FIRE (<sha8>), ...'.
usage: [TL=..] python -B runs/compare_twin_byteid_v1.py <tag> <base quick json> <base tag>"""
import json, time, sys, hashlib, os, re
from pathlib import Path
import numpy as np
TAG, BASE, BTAG = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
R = Path(__file__).resolve().parents[1]; O = R.parent
TL = Path(os.environ.get('TL') or O/'TEAM_LOG.md')
sys.path.insert(0, str(R))
import project   # noqa: E402
BJ = (BASE if BASE.is_absolute() else R/BASE).resolve()
tw = json.loads((R/'runs'/f'quick_{TAG}_a.json').read_text()); base = json.loads(BJ.read_text())
T = {r['rank']: r for r in tw['rows']}; B = {r['rank']: r for r in base['rows']}
ranks = [r['rank'] for r in base['rows'] if r['rank'] in T]
same, diffs, db = [], [], []
for k in ranks:
    pt = np.load(R/'runs'/f'quick_{TAG}_a_pred_{k}.npy'); pb = np.load(BJ.with_name(BJ.stem + f'_pred_{k}.npy'))
    eq = pt.shape == pb.shape and pt.dtype == pb.dtype and np.array_equal(pt, pb) and T[k]['pred_sha'] == B[k]['pred_sha'] and T[k]['mse'] == B[k]['mse']
    same.append(eq)
    if not eq:
        layers = [l for l in range(min(len(pt), len(pb))) if not np.array_equal(pt[l], pb[l])] if pt.shape == pb.shape else ['shape']
        diffs.append(f"{k}: layers {layers[:4]}{'...' if len(layers) > 4 else ''} differ" + (f", max |d| {float(np.max(np.abs(pt - pb))):.1e}" if pt.shape == pb.shape else ''))
    db.append(100*(float(T[k]['flops'])/float(B[k]['flops']) - 1))
gate_lines = [l for l in TL.read_text().splitlines() if f'{TAG} SAFE GATE PASS' in l]
fire_cmd = ('FIRE COMMAND' + gate_lines[-1].split('FIRE COMMAND', 1)[1]) if gate_lines and 'FIRE COMMAND' in gate_lines[-1] else 'no gate PASS line found: DO NOT FIRE'
ok_gate = bool(gate_lines)
ok = ok_gate and len(ranks) == 10 and all(same)


def counts(t):
    for rows in (R/'runs'/f'fg_{t}_n_src'/'C0'/'rows.jsonl', R/'runs'/f'c_{t}'/'C0'/'rows.jsonl'):
        if rows.exists():
            r = [json.loads(x) for x in rows.read_text().splitlines()]
            return r[0]['requests'], r[1]['requests'], r[0]['flops'], r[1]['flops']
    return None


def last_fired_twin():
    best = None
    for a in (O/'fire').glob('*/attempt.json'):
        d = json.loads(a.read_text())
        if 'twin' in d.get('tag', '') and d.get('archive') and (a.parent/'receipt.json').exists() and (best is None or d.get('utc', '') > best[0]):
            best = (d.get('utc', ''), Path(d['archive']).parent, d['tag'])
    return best


me, lf = counts(f'{TAG}_safe'), last_fired_twin()
flag = 'requests / residual / bill against the last fired twin: not known (no counts)'
if me and lf:
    them = counts(lf[1].name[2:])
    graded = project.graded(json.loads((lf[1]/'stage_a.json').read_text())['candidate_sha256'])
    if them:
        up = me[0] > them[0] or me[1] > them[1]
        worst = (f"; its worst platform network residual {graded[0]['residual_max']:.4f} s (#{graded[0]['sid']}) scales to "
                 f"{graded[0]['residual_max']*me[1]/them[1]:.4f} s here (cap 0.4 s)") if graded else '; it is not graded yet'
        b35, b100 = 100*(me[2]/them[2] - 1), 100*(me[3]/them[3] - 1)
        flag = (f"{'RESIDUAL UP' if up else 'residual not up'}: first / warm {me[0]:,} / {me[1]:,} requests against {them[0]:,} / {them[1]:,} of the last "
                f"fired twin {lf[2]} ({me[0] - them[0]:+,} / {me[1] - them[1]:+,}){worst}; {'BILL UP' if b35 + b100 > 0 else 'bill not up'} against it at the "
                f"platform pace (count sessions): network 35 {b35:+.3f}%, 100 {b100:+.3f}%")
SHA = hashlib.sha256((R/'variants'/f'{TAG}_safe'/'candidate.py').read_bytes()).hexdigest()[:8]
ROUND = re.fullmatch(r's\d+tw\d*', TAG) is not None
head = f"TWIN {TAG} ({SHA})," if ROUND else f"TWIN {TAG} {'CONFIRMED' if ok else 'DO NOT FIRE'} ({SHA}),"
line = (f"[release] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {head} BYTE-IDENTICAL MODE (lead 22:57Z), NATIVE TEN NETWORKS of the FINAL "
        f"twin (list a, quick.py, O/release/runs/quick_{TAG}_a.json) against {BTAG}'s own native ten ({BJ.name}): predictions byte-identical on "
        f"{sum(same)} of {len(ranks)} networks (pred sha, final MSE and all 16 layers equal)" + (f"; DIFFERENT: {'; '.join(diffs)[:300]}" if diffs else '')
        + f"; native bill against {BTAG}'s run {sum(db)/len(db):+.3f}% mean (information only: load-dependent); {flag}. "
        + ((f"VERDICT: {'CONFIRMED, ' if ROUND else ''}byte-identical predictions on all ten networks: GO. " + fire_cmd) if ok else
           f"VERDICT: {'DO NOT FIRE, ' if ROUND else ''}" + ('; '.join(x for x in ('' if all(same) and len(ranks) == 10 else 'NOT byte-identical on all ten networks',
                                                                                   '' if ok_gate else 'no gate PASS line') if x)) + ": DO NOT FIRE the twin."))
with open(TL, 'a') as fh:
    fh.write(line + '\n')
print(line)
