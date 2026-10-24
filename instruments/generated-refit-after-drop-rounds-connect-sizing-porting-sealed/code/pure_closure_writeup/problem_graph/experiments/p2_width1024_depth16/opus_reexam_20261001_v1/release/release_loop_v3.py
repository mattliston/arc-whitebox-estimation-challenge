"""TAG opus-release generic release loop v3 (2026-10-12 18:00 UTC; lead 17:4xZ + 17:28Z stack protocol: round twins also get FASTGATE lines):
v2 with the TWIN step changed: after the safe gate, when the engineer's line 'FASTGATE s<N>tw READY: twin /...py <the READY line's twin sha8>, ...'
exists (exact shape, line start) and the fast-gate loop is alive, the twin is left to fastgate (one TEAM_LOG line says so); else, after
waiting up to 30 min from the end of the safe gate, the full twin_pipeline_v5.sh runs (native guards 7200 s instead of 3000 s, base
session beside the twin run). Safe gate unchanged. Same state file, so no round runs twice. v2 header follows.
TAG opus-release generic release loop v2 (2026-10-12 08:25 UTC): v1 (release_loop_v1.py, 08:10 UTC) with the twin step on
twin_pipeline_v4.sh (its two native sessions under guard_footprint_v2.py); the stage-C sessions use v2 through common.GUARD.
Same state file, so no round runs twice. v1: lead 08:0xZ, RAD rounds S51-S63 back to back until 10-17.
Polls O/TEAM_LOG.md every 60 s for engineer lines
    [engineer] <stamp> S<N> FILES READY FOR RELEASE: safe <path> <sha8>, twin <path> <sha8>
and runs, for each new one (state in runs/release_loop_state.json), exactly tonight's S50 sequence, one process at a time:
  0. both files exist and match their sha8; the safe file carries each release fragment exactly once (Rl3_bind128, Rlvfs_limit_off,
     Rprices_zero), the twin exactly one Rl3_bind128 and one Rprices_zero and no Rlvfs_limit_off; any mismatch -> one TEAM_LOG line,
     round skipped;
  1. waits for gate.sh (4 GB, label release-loop) and for no open QUIET WINDOW of the lead;
  2. safe gate: FRAGS=none FIRST_PRICED=1 RANKS39=1 PROVEN=<the last FIRED safe file> s47_safe_v7.sh <safe> s<N>s<k> s<N>s_<UTC date>
     (k = 1, or the next unused one when a round is handed over again);
  3. twin: BASESRC=<that safe file> PROVEN=(same) twin_pipeline_v5.sh <twin> s<N>tw s<N>twin_<UTC date> (a repeat round: s<N>tw<k>).
The scripts write the lines the trainer's add-ons key on ('s<N>s1 SAFE GATE PASS' / 's<N>s1 SAFE: DO NOT FIRE', 'TWIN s<N>tw (...) ... VERDICT:').
Last fired safe = the newest O/fire/*/attempt.json whose tag has no 'twin': the estimator.py next to its archive (payload/). Nothing is
uploaded: the lead fires by hand.   log: runs/release_loop.log   stop: kill $(cat runs/release_loop.pid)
usage: nohup python -B release_loop_v3.py &"""
import json, os, re, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import hashlib

R = Path(__file__).resolve().parent; O = R.parent
DRY = bool(os.environ.get('LOOP_DRY'))   # rehearsal: log every command, run none, write no TEAM_LOG line
SRC = Path(os.environ['LOOP_SRC']) if os.environ.get('LOOP_SRC') else None   # rehearsal: read READY lines from this file
SUF = '_dry' if DRY else ''
LOG, STATE, PID = R/'runs'/f'release_loop{SUF}.log', R/'runs'/f'release_loop{SUF}_state.json', R/'runs'/f'release_loop{SUF}.pid'
READY = re.compile(r'^\[engineer\] (\S+) S(\d+) FILES READY FOR RELEASE: safe (\S+) ([0-9a-f]{8}), twin (\S+) ([0-9a-f]{8})')
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


def sha8(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def frag(p, name):
    return sum(1 for l in p.read_text().splitlines() if l.startswith(f'# BEGIN OPUS-RELEASE FRAGMENT {name} '))


def window_open():
    """True when the lead's newest QUIET WINDOW line (first phrase in it) says OPEN."""
    last = [l for l in (O/'TEAM_LOG.md').read_text().splitlines() if re.match(r'^\[lead\] .{0,40}QUIET WINDOW (OPEN|CLOSED)', l)]
    return bool(last) and re.search(r'QUIET WINDOW (OPEN|CLOSED)', last[-1]).group(1) == 'OPEN'


def run(cmd, env, out):
    if DRY:
        log('DRY run: ' + ' '.join(f'{k}={env[k]}' for k in ('FRAGS', 'FIRST_PRICED', 'RANKS39', 'BASESRC', 'PROVEN', 'TWIN_DESC') if k in env)
            + ' ' + ' '.join(cmd) + f' > {out}'); return
    with open(out, 'w') as fh:
        subprocess.run(cmd, cwd=R, env=env, stdout=fh, stderr=subprocess.STDOUT)


def last_fired_safe():
    best = None
    for a in (O/'fire').glob('*/attempt.json'):
        d = json.loads(a.read_text())
        if 'twin' in d.get('tag', '') or not d.get('archive') or not (a.parent/'receipt.json').exists():
            continue
        est = Path(d['archive']).parent/'payload'/'estimator.py'
        if est.exists() and (best is None or d.get('utc', '') > best[0]):
            best = (d.get('utc', ''), est, d.get('tag'))
    return best


DEFER_S = 1800
FG_LINE = r'^\[engineer\] (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) FASTGATE s%dtw READY: twin /\S+\.py %s, '


def fastgate_line(n, tsha):
    """Stamp of the engineer's FASTGATE line for round n's twin with THIS sha8 (exact shape, line start), else None."""
    for line in (SRC or O/'TEAM_LOG.md').read_text().splitlines():
        m = re.match(FG_LINE % (n, tsha), line)
        if m:
            return m.group(1)
    return None


def fastgate_alive():
    try:
        pid = int((R/'runs'/'fastgate_loop.pid').read_text())
        return 'fastgate_loop_v' in subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
    except (OSError, ValueError):
        return False


def free_tag(stem, start_plain):
    if start_plain and not (R/'runs'/f'twinpipe_{stem}.log').exists() and not (R/'runs'/f's47safe_{stem}.log').exists():
        return stem
    k = 1 if not start_plain else 2
    while (R/'runs'/f's47safe_{stem}{k}.log').exists() or (R/'runs'/f'twinpipe_{stem}{k}.log').exists():
        k += 1
    return f'{stem}{k}'


def fire_tag(stem):
    t = f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    for suffix in ['', 'b', 'c', 'd', 'e']:
        if not (O/'fire'/(t + suffix)).exists():
            return t + suffix
    return t + 'f'


def wait_machine(label):
    if DRY:
        log(f'DRY wait gate.sh 4 {label}'); return
    while window_open() or subprocess.run(['pgrep', '-f', 'fastgate_v[0-9].sh'], capture_output=True).returncode == 0:
        log(f'{label}: quiet window open or a fastgate run alive, waiting'); time.sleep(60)
    subprocess.run(['bash', str(O/'gate.sh'), '4', label], stdout=open(R/'runs'/'release_loop_gate.log', 'a'), stderr=subprocess.STDOUT)


def run_round(n, safe, ssha, twin, tsha):
    for p, h, what in ((safe, ssha, 'safe'), (twin, tsha, 'twin')):
        if not p.exists() or sha8(p) != h:
            team(f'S{n} RELEASE LOOP: {what} file {p} missing or not sha {h} (found {sha8(p) if p.exists() else "none"}): round skipped'); return
    fs = {k: frag(safe, k) for k in ('l3_bind128', 'lvfs_limit_off', 'prices_zero')}
    ft = {k: frag(twin, k) for k in ('l3_bind128', 'lvfs_limit_off', 'prices_zero')}
    if fs != {'l3_bind128': 1, 'lvfs_limit_off': 1, 'prices_zero': 1}:
        team(f'S{n} RELEASE LOOP: safe file {ssha} release fragments {fs}, not exactly one each: safe gate skipped (no doubled fragments, lesson of s48s)'); safe_ok = False
    else:
        safe_ok = True
    fired = last_fired_safe()
    proven = fired[1] if fired else O/'engineer'/'a62_modules'/'s48'/'m_sd_rc'/'candidate.py'
    log(f'S{n}: safe {safe} {ssha}, twin {twin} {tsha}; proven = {proven} (fire {fired[2] if fired else "none"})')
    if safe_ok:
        tag = free_tag(f's{n}s', False)
        wait_machine(f'release-loop {tag}')
        env = dict(os.environ, FRAGS='none', FIRST_PRICED='1', RANKS39='1', PROVEN=str(proven), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        log(f'start safe gate {tag}: PROVEN={proven} s47_safe_v7.sh {safe} {tag} {fire_tag(f"s{n}s")}')
        run(['bash', str(R/'s47_safe_v7.sh'), str(safe), tag, fire_tag(f's{n}s'),
             f'S{n} safe: engineer {ssha} (RAD round S{n}; carries Rl3_bind128 + Rlvfs_limit_off + Rprices_zero)'], env, R/'runs'/f'release_loop_{tag}.out')
        log(f'safe gate {tag} ended')
    if ft != {'l3_bind128': 1, 'lvfs_limit_off': 0, 'prices_zero': 1}:
        team(f'S{n} RELEASE LOOP: twin file {tsha} release fragments {ft}, a twin needs one Rl3_bind128, one Rprices_zero, no Rlvfs_limit_off: twin skipped'); return
    deadline = time.time() + DEFER_S
    while True:   # lead 17:2xZ: round twins also get FASTGATE lines: the fast gate takes the twin when its line for THIS file exists
        fg = fastgate_line(n, tsha)
        if fg and fastgate_alive():
            started = DRY
            for _ in range(20):   # the fast-gate run writes runs/fastgate_s<N>tw.log first: give it 10 min to start
                if started or (R/'runs'/f'fastgate_s{n}tw.log').exists():
                    started = True
                    break
                time.sleep(30)
            if started:
                log(f'S{n} twin {tsha}: the engineer posted FASTGATE s{n}tw at {fg} and fastgate took it: no twin pipeline here')
                team(f'S{n} TWIN {tsha} GOES TO FASTGATE (its line of {fg}; the fast-gate run has started): no full twin pipeline from loop v3')
                return
            log(f'S{n} twin {tsha}: FASTGATE s{n}tw line of {fg} but no fast-gate run started within 10 min: full twin pipeline')
            break
        if DRY or time.time() > deadline:
            break
        time.sleep(60)
    log(f'S{n} twin {tsha}: no FASTGATE s{n}tw line for this file within {DEFER_S//60} min (or no fast-gate loop): full twin pipeline')
    ttag = free_tag(f's{n}tw', True)
    wait_machine(f'release-loop {ttag}')
    env = dict(os.environ, BASESRC=str(safe), PROVEN=str(proven), TWIN_DESC=f'S{n} twin: engineer {tsha} (aggressive half of the pair)',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    log(f'start twin pipeline {ttag}: BASESRC={safe} PROVEN={proven} twin_pipeline_v5.sh {twin} {ttag} {fire_tag(f"s{n}twin")}')
    run(['bash', str(R/'twin_pipeline_v5.sh'), str(twin), ttag, fire_tag(f's{n}twin')], env, R/'runs'/f'release_loop_{ttag}.out')
    log(f'twin pipeline {ttag} ended')


def main():
    PID.write_text(str(os.getpid()))
    state = json.loads(STATE.read_text()) if STATE.exists() else {'done': []}
    log(f'loop v3 started (pid {os.getpid()}); handled before: {len(state["done"])}')
    while True:
        try:
            for line in (SRC or O/'TEAM_LOG.md').read_text().splitlines():
                m = READY.match(line)
                if not m:
                    continue
                key = f'{m.group(1)} S{m.group(2)}'
                if key in state['done']:
                    continue
                state['done'].append(key); STATE.write_text(json.dumps(state, indent=1))
                log(f'READY line: {line[:300]}')
                run_round(int(m.group(2)), resolve(m.group(3)), m.group(4), resolve(m.group(5)), m.group(6))
        except Exception as exc:
            log(f'error: {type(exc).__name__}: {exc}')
            team(f'RELEASE LOOP error {type(exc).__name__}: {exc} (round left as handled; see O/release/runs/release_loop.log)')
        if DRY:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
