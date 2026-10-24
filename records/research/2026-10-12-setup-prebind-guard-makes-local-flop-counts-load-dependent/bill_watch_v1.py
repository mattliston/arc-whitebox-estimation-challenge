"""TAG opus-release bill watch v1 (2026-10-12 18:00 UTC; lead 17:4xZ: 'if S51's twin verdict from loop v2 says DO NOT FIRE on the BILL alone,
post the stage C platform-pace flops beside it and I'll judge').
Why: the twin pipeline's verdict (compare_twin_native_v2) needs the NATIVE bill below the base's, but on a loaded box quick.py's flops follow
the load (the files' 3 s prebind wall guard; same file: +0.29..0.56% flops), while the platform bills at its own pace, which the stage-C count
sessions emulate (load-independent).
Polls O/TEAM_LOG.md every 60 s for release's own twin verdict lines ('[release] <stamp> TWIN s<N>tw (<sha8>), NATIVE TEN NETWORKS ...
VERDICT: OUTSIDE ...'). When the physics passed (pooled MSE within +-0.5%, largest prediction difference < 1e-2) and only the native bill
failed (mean >= 0), it posts ONE line: the platform-pace flops of the twin (its gate's stage-C session) against the round's last safe gate
(c_s<N>s<k>_safe, highest k) on networks 35 and 100. It judges nothing: the lead does. State: runs/bill_watch_state.json.
log: runs/bill_watch.log   stop: kill $(cat runs/bill_watch.pid)   usage: nohup python -B bill_watch_v1.py &"""
import json, os, re, time
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parent; O = R.parent
LOG, STATE = R/'runs'/'bill_watch.log', R/'runs'/'bill_watch_state.json'
VERDICT = re.compile(r'^\[release\] (\S+) TWIN (s(\d+)tw\d*) \(([0-9a-f]{8})\), NATIVE TEN NETWORKS .*?final-layer MSE pooled ([+-]?[\d.]+)% .*?'
                     r'largest final-layer prediction difference ([\d.e+-]+) of the prediction scale.*? bill ([+-]?[\d.]+)% mean .*VERDICT: OUTSIDE')
now = lambda: datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(msg):
    with open(LOG, 'a') as fh:
        fh.write(f'{now()} {msg}\n')


def flops(t):
    rows = R/'runs'/f'c_{t}'/'C0'/'rows.jsonl'
    r = [json.loads(x) for x in rows.read_text().splitlines()] if rows.exists() else []
    return {x['rank']: x['flops'] for x in r} if r else None


def main():
    (R/'runs'/'bill_watch.pid').write_text(str(os.getpid()))
    state = json.loads(STATE.read_text()) if STATE.exists() else {'done': []}
    log(f'bill watch v1 started (pid {os.getpid()})')
    while True:
        try:
            for line in (O/'TEAM_LOG.md').read_text().splitlines():
                m = VERDICT.match(line)
                if not m or f'{m.group(1)} {m.group(2)}' in state['done']:
                    continue
                state['done'].append(f'{m.group(1)} {m.group(2)}'); STATE.write_text(json.dumps(state, indent=1))
                tag, n, pool, diff, bill = m.group(2), m.group(3), float(m.group(5)), float(m.group(6)), float(m.group(7))
                log(f'DO NOT FIRE verdict {tag}: pooled {pool:+.2f}%, difference {diff:.1e}, native bill {bill:+.3f}%')
                if not (abs(pool) < 0.5 and diff < 1e-2 and bill >= 0):
                    log(f'{tag}: not a bill-only verdict, no line'); continue
                tw = flops(f'{tag}_safe')
                safes = sorted((int(p.name[len(f'c_s{n}s'):-len('_safe')]), p.name[2:]) for p in (R/'runs').glob(f'c_s{n}s*_safe')
                               if re.fullmatch(rf'c_s{n}s\d+_safe', p.name))
                sf = flops(safes[-1][1]) if safes else None
                if not tw or not sf:
                    msg = f'the stage-C sessions are missing (twin c_{tag}_safe {"ok" if tw else "missing"}, safe {safes[-1][1] if safes else "none"})'
                else:
                    cells = [f'network {k} {100*(tw[k]/sf[k] - 1):+.3f}% ({tw[k]:,} vs {sf[k]:,} flops)' for k in (35, 100) if k in tw and k in sf]
                    up = sum(tw[k] - sf[k] for k in (35, 100) if k in tw and k in sf) >= 0
                    msg = (f"at the platform pace (stage-C count sessions, emulated setup clock, load-independent) the twin bills {'; '.join(cells)} "
                           f"against the round's safe gate {safes[-1][1][:-5]}: {'NO bill cut there either' if up else 'a bill CUT there'}")
                with open(O/'TEAM_LOG.md', 'a') as fh:
                    fh.write(f'[release] {now()} TWIN {tag} BILL-ONLY DO NOT FIRE (lead 17:4xZ, for the lead to judge): the verdict above failed on the '
                             f'native bill alone (pooled MSE {pool:+.2f}%, largest prediction difference {diff:.1e}, native bill {bill:+.3f}%, which '
                             f'follows the box load through the 3 s prebind wall guard); {msg}.\n')
                log(f'{tag}: line posted')
        except Exception as exc:
            log(f'error: {type(exc).__name__}: {exc}')
        time.sleep(60)


if __name__ == '__main__':
    main()
