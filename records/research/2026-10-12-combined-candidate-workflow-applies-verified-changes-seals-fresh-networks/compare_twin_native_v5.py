"""TAG opus-release v5 (2026-10-12 18:05 UTC; lead's option (ii) for ROUND twins): v4 with the line shape chosen by the tag. A ROUND tag
(s<N>tw, s<N>tw<k>) keeps the shape the trainer's add-ons key on: 'TWIN s<N>tw (<sha8>), NATIVE TEN NETWORKS ... VERDICT: CONFIRMED ...: GO'
(or 'VERDICT: DO NOT FIRE ...'); a STACK tag (s<N>k<j>tw) reads 'TWIN s<N>k<j>tw CONFIRMED (<sha8>), ...' as in v4. Checks unchanged.
v4 (2026-10-12 17:40 UTC): v3 with the BILL taken from the deterministic count sessions, not from the native run. Reason
(measured 17:31Z): the files' prebind binds keys until a 3 s WALL guard (_PB_WALL_S), so on a loaded box quick.py binds fewer keys and its
ops / flops grow (same file f04fa779, network 651: +814 ops, +0.56% flops against the 06:40Z run; predictions bit-identical), so the native
bill check could fail or pass for the load, not the file. The platform bills at its own pace, which the count sessions emulate (--short-setup:
setup clock from requests, load-independent). So: the native ten networks judge the PHYSICS (pooled MSE within +-0.5% of the base, largest
prediction difference < 1e-2, all ten present, gate PASS line); the bill is reported from the count sessions on 35 / 100 against the LAST
FIRED twin ('BILL UP' at any rise), next to the residual flag; the native bill stays in the line as information.
v3 (2026-10-12 17:50 UTC; lead 17:1xZ fastgate): v2 with (1) the lead's fire words: 'TWIN <tag> CONFIRMED (<sha8>), ...'
when every check holds, else 'TWIN <tag> DO NOT FIRE (<sha8>), ...'; (2) the residual flag: this twin's first / warm request counts (its gate's
normal session) against the LAST FIRED twin's, with that twin's worst platform network residual scaled by the warm ratio, 'RESIDUAL UP' when
either count is higher at all; (3) TL=<file> writes the line there instead of TEAM_LOG (rehearsals). The checks are v2's: final-layer MSE pooled
within +-0.5% of the base file's own native session, largest prediction difference < 1e-2 of the scale, bill below the base's, gate PASS line.
usage: [TL=..] python -B runs/compare_twin_native_v3.py <tag> <base quick json>"""
import json, time, sys, hashlib, os
from pathlib import Path
import numpy as np
TAG = sys.argv[1]
BASE = Path(sys.argv[2])
R = Path(__file__).resolve().parents[1]; O = R.parent
TL = Path(os.environ.get('TL') or O/'TEAM_LOG.md')
sys.path.insert(0, str(R))
import project   # noqa: E402
BJ = (BASE if BASE.is_absolute() else R/BASE).resolve()
tw = json.loads((R/'runs'/f'quick_{TAG}_a.json').read_text()); base = json.loads(BJ.read_text())
T = {r['rank']: r for r in tw['rows']}; B = {r['rank']: r for r in base['rows']}
ranks = [r['rank'] for r in base['rows'] if r['rank'] in T]
dm, db, do, dp = [], [], [], []
for k in ranks:
    t, b = T[k], B[k]
    dm.append(100*(float(t['mse'])/float(b['mse']) - 1)); db.append(100*(float(t['flops'])/float(b['flops']) - 1)); do.append(int(t['ops']) - int(b['ops']))
    pt = np.load(R/'runs'/f'quick_{TAG}_a_pred_{k}.npy'); pb = np.load(BJ.with_name(BJ.stem + f'_pred_{k}.npy'))
    dp.append(float(np.max(np.abs(pt[15] - pb[15]))/np.max(np.abs(pb[15]))))
pool = 100*(sum(float(T[k]['mse']) for k in ranks)/sum(float(B[k]['mse']) for k in ranks) - 1)
mean = sum(dm)/len(dm); se = (sum((x - mean)**2 for x in dm)/(len(dm) - 1)/len(dm))**0.5
gate_lines = [l for l in TL.read_text().splitlines() if f'{TAG} SAFE GATE PASS' in l]
fire_cmd = ('FIRE COMMAND' + gate_lines[-1].split('FIRE COMMAND', 1)[1]) if gate_lines and 'FIRE COMMAND' in gate_lines[-1] else 'no gate PASS line found: DO NOT FIRE'
ok_gate = bool(gate_lines)
ok = abs(pool) < 0.5 and max(dp) < 1e-2 and ok_gate and len(ranks) == 10


def counts(t):
    """(first requests, warm requests, flops 35, flops 100) of a gated variant t (= <tag>_safe): its fast-gate copy of the normal session,
    else its own stage-C session (both at the emulated platform setup pace)."""
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


me = counts(f'{TAG}_safe')
lf = last_fired_twin()
flag = 'residual vs the last fired twin: not known (no counts)'
if me and lf:
    st = json.loads((lf[1]/'stage_a.json').read_text())
    them = counts(lf[1].name[2:])
    graded = project.graded(st['candidate_sha256'])
    if them:
        up = me[0] > them[0] or me[1] > them[1]
        worst = (f"; its worst platform network residual {graded[0]['residual_max']:.4f} s (#{graded[0]['sid']}) scales to "
                 f"{graded[0]['residual_max']*me[1]/them[1]:.4f} s here (cap 0.4 s)") if graded else '; it is not graded yet'
        b35, b100 = 100*(me[2]/them[2] - 1), 100*(me[3]/them[3] - 1)
        flag = (f"{'RESIDUAL UP' if up else 'residual not up'}: first / warm {me[0]:,} / {me[1]:,} requests against {them[0]:,} / {them[1]:,} of the last "
                f"fired twin {lf[2]} ({me[0] - them[0]:+,} / {me[1] - them[1]:+,}){worst}; {'BILL UP' if b35 + b100 > 0 else 'bill not up'} against it at the "
                f"platform pace (count sessions): network 35 {b35:+.3f}%, 100 {b100:+.3f}%")
SHA = hashlib.sha256((R/'variants'/f'{TAG}_safe'/'candidate.py').read_bytes()).hexdigest()[:8]
import re
ROUND = re.fullmatch(r's\d+tw\d*', TAG) is not None
head = f"TWIN {TAG} ({SHA})," if ROUND else f"TWIN {TAG} {'CONFIRMED' if ok else 'DO NOT FIRE'} ({SHA}),"
line = (f"[release] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {head} NATIVE TEN NETWORKS "
        f"of the FINAL twin (list a, quick.py, O/release/runs/quick_{TAG}_a.json) against the base file's own native session "
        f"({BJ.name}): final-layer MSE pooled {pool:+.2f}% (mean of per-network changes {mean:+.2f}%, se {se:.2f}; lower on "
        f"{sum(1 for x in dm if x < 0)} of {len(dm)}); per network " + ' '.join(f'{x:+.2f}' for x in dm) + f" %; largest final-layer prediction "
        f"difference {max(dp):.1e} of the prediction scale; native bill {sum(db)/len(db):+.3f}% mean ({min(db):+.3f}..{max(db):+.3f}%) and charged ops "
        f"{sum(do)/len(do):+.0f} mean ({min(do):+d}..{max(do):+d}) against the base (information only: they depend on the box's load through the "
        f"3 s prebind wall guard); {flag}. "
        + ((f"VERDICT: {'CONFIRMED, ' if ROUND else ''}predictions at rounding level on all ten networks: GO. " + fire_cmd) if ok else
           f"VERDICT: {'DO NOT FIRE, ' if ROUND else ''}OUTSIDE what the gate promised (see the numbers" + ('' if ok_gate else '; no gate PASS line') + "): DO NOT FIRE the twin."))
with open(TL, 'a') as fh:
    fh.write(line + '\n')
print(line)
