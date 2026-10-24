"""VM launch queue v3: keeps MAX_ACTIVE launches in flight (launch the next pending QUEUE_v1.txt line whenever a slot is free;
lines are ordered as (control, probe) pairs so the two in flight are normally a probe and its fresh reference). Logs failed runs."""
import hashlib, json, subprocess, time
from pathlib import Path
RUNS = Path('/home/arcworker/arc-runs'); HERE = Path(__file__).resolve().parent
PY = '/home/arcworker/arc-envs/judge/bin/python'
LAUNCHER = str(HERE.parents[0] / 'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/launch_frozen_jobs12_markline_v2.py')
QUEUE = HERE / 'QUEUE_v1.txt'; LOG = HERE / 'queue_runner_v5.log'; MAX_ACTIVE = 3
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def log(msg):
    with LOG.open('a') as f:
        f.write('%s %s\n' % (utc(), msg))

def alive(rec):
    try:
        raw = (Path('/proc') / str(rec['pid']) / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return int(raw[raw.rfind(')') + 2:].split()[19]) == rec['start_ticks']

def active():
    n = 0
    for f in RUNS.glob('f21_*_LAUNCH.json'):   # v5: count only this sweep's launches; the lead's runs are covered by the memory guard
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        if {'pid', 'start_ticks', 'out'} <= set(rec) and alive(rec) and not (Path(rec['out']) / 'COMPLETE.json').exists():
            n += 1
    return n

FAILED_SEEN = set()
MIN_AVAILABLE_GB = 48   # one more 12-job launch needs ~30 GB plus the guard's 19 GB reserve


def available_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1048576.0
    return 0.0


def main():
    log('runner v5 start (own launches only) max_active=%d min_available_gb=%d' % (MAX_ACTIVE, MIN_AVAILABLE_GB))
    while True:
        lines = [l.split('\t')[:3] for l in QUEUE.read_text().splitlines() if l.strip() and not l.startswith('#')]
        pending = [(t, p, c) for t, p, c in lines if not (RUNS / (t + '_LAUNCH.json')).exists()]
        n = active()
        for t, p, c in lines:
            done = RUNS / t / 'COMPLETE.json'
            if done.exists() and t not in FAILED_SEEN:
                FAILED_SEEN.add(t)
                try:
                    ok = json.loads(done.read_text()).get('successful')
                except Exception:
                    ok = None
                if not ok:
                    log('FAILED run %s (%s): COMPLETE.json successful=%r' % (t, c, ok))
        if pending and n < MAX_ACTIVE and available_gb() < MIN_AVAILABLE_GB:
            time.sleep(15)
            continue
        if pending and n < MAX_ACTIVE:
            tag, plan_path, cand = pending[0]
            plan = json.loads(Path(plan_path).read_text())
            cmd = [PY, '-B', LAUNCHER, '--plan', plan_path, '--plan-sha', sha(plan_path), '--candidate', cand,
                   '--source-sha', plan['candidates'][cand]['sha256'], '--tag', tag,
                   '--authorization', 'fable lead VM dial sweep 2026-10-21 (VM teammate queue runner v3)']
            r = subprocess.run(cmd, capture_output=True, text=True)
            log('launch %s %s rc=%d active_before=%d %s' % (tag, cand, r.returncode, n, (r.stdout + r.stderr).strip()[:200]))
            time.sleep(8)
            continue
        time.sleep(15)

if __name__ == '__main__':
    main()
