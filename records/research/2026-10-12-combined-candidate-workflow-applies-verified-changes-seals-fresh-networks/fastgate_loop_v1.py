"""TAG opus-release fastgate loop v1 (2026-10-12 18:00 UTC; lead 17:1xZ: fastgate_v1 approved beside the running release loop).
Polls O/TEAM_LOG.md every 60 s for engineer lines
    [engineer] <stamp> FASTGATE <tag> READY: twin <path> <sha8>, base <safe-physics quick json>, normal <dir>, slow <dir>[, n39 <dir>]
and runs, one at a time and in TEAM_LOG order, BASEQ=<base> bash fastgate_v2.sh (17:40Z, before any launch: v1 -> v2) <twin> <tag> <fire tag> <normal> <slow> [<n39>] (fire tag:
<tag minus 'tw'>twin_<UTC date>, e.g. s51k1tw -> s51k1twin_20261012). fastgate_v1 itself yields to release_loop_v2's rounds (S<N> gates first),
checks the harness pin and falls back to the full twin pipeline. A wrong sha or a used tag posts one line and skips. State in
runs/fastgate_loop_state.json (a line is marked handled before it runs: never twice). Nothing uploads: the lead fires after the CONFIRM line.
rehearsal: LOOP_DRY=1 [LOOP_SRC=<file of test lines>] logs the commands, runs nothing, posts nothing, one pass.
log: runs/fastgate_loop.log   stop: kill $(cat runs/fastgate_loop.pid)   usage: nohup python -B fastgate_loop_v1.py &"""
import hashlib, json, os, re, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parent; O = R.parent
DRY = bool(os.environ.get('LOOP_DRY'))
SRC = Path(os.environ['LOOP_SRC']) if os.environ.get('LOOP_SRC') else None
SUF = '_dry' if DRY else ''
LOG, STATE, PID = R/'runs'/f'fastgate_loop{SUF}.log', R/'runs'/f'fastgate_loop{SUF}_state.json', R/'runs'/f'fastgate_loop{SUF}.pid'
LINE = re.compile(r'^\[engineer\] (\S+) FASTGATE (\S+) READY: twin (\S+) ([0-9a-f]{8}), base (\S+), normal (\S+), slow ([^\s,]+)(?:, n39 (\S+))?')
now = lambda: datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(msg):
    with open(LOG, 'a') as fh:
        fh.write(f'{now()} {msg}\n')


def team(msg):
    if DRY:
        log('DRY team line: ' + msg); return
    with open(O/'TEAM_LOG.md', 'a') as fh:
        fh.write(f'[release] {now()} {msg}\n')


def resolve(p):
    q = Path(p[2:]) if p.startswith('O/') else Path(p)
    return q if q.is_absolute() else (O/q if p.startswith('O/') or not (R/q).exists() else R/q)


def fire_tag(tag):
    t = f"{tag[:-2] if tag.endswith('tw') else tag}twin_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    for suffix in ['', 'b', 'c', 'd', 'e']:
        if not (O/'fire'/(t + suffix)).exists():
            return t + suffix
    return t + 'f'


def run_one(tag, twin, tsha, base, normal, slow, n39):
    if not twin.exists() or hashlib.sha256(twin.read_bytes()).hexdigest()[:8] != tsha:
        team(f'FASTGATE {tag}: twin file {twin} missing or not sha {tsha}: skipped'); return
    if not re.fullmatch(r's\d+[a-z0-9]*', tag) or (R/'runs'/f'fastgate_{tag}.log').exists() or (R/'runs'/f's47safe_{tag}.log').exists():
        team(f'FASTGATE {tag}: tag not of the form s<N>.. or used already: skipped (post the line again with a new tag)'); return
    cmd = ['bash', str(R/'fastgate_v2.sh'), str(twin), tag, fire_tag(tag), str(normal), str(slow)] + ([str(n39)] if n39 else [])
    env = dict(os.environ, BASEQ=str(base), TWIN_DESC=f'{tag} twin: engineer {tsha} (stack, fast-gated)', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    log(f'start fastgate {tag}: BASEQ={base} ' + ' '.join(cmd[1:]))
    if DRY:
        log('DRY run: ' + ' '.join(cmd)); return
    with open(R/'runs'/f'fastgate_loop_{tag}.out', 'w') as fh:
        subprocess.run(cmd, cwd=R, env=env, stdout=fh, stderr=subprocess.STDOUT)
    log(f'fastgate {tag} ended')


def main():
    PID.write_text(str(os.getpid()))
    state = json.loads(STATE.read_text()) if STATE.exists() else {'done': []}
    log(f'fastgate loop v1 started (pid {os.getpid()}); handled before: {len(state["done"])}')
    while True:
        try:
            for line in (SRC or O/'TEAM_LOG.md').read_text().splitlines():
                m = LINE.match(line)
                if not m or f'{m.group(1)} {m.group(2)}' in state['done']:
                    continue
                state['done'].append(f'{m.group(1)} {m.group(2)}'); STATE.write_text(json.dumps(state, indent=1))
                log(f'FASTGATE line: {line[:300]}')
                run_one(m.group(2), resolve(m.group(3)), m.group(4), resolve(m.group(5)), resolve(m.group(6)), resolve(m.group(7)),
                        resolve(m.group(8)) if m.group(8) else None)
        except Exception as exc:
            log(f'error: {type(exc).__name__}: {exc}')
            team(f'FASTGATE LOOP error {type(exc).__name__}: {exc} (line left as handled; see O/release/runs/fastgate_loop.log)')
        if DRY:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
