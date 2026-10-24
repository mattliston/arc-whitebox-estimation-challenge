"""TAG opus-trainer RAD (10-12 13:2xZ; lead 13:3xZ: measure memory during S51's downstream to set the hold rule from the minima): ONE sample,
appended as one JSON line to cert/RAD/memlog_v1.jsonl: UTC, memory_pressure free %, swap used MiB, free + inactive GiB, our RAD workers per run,
and per lane (physicist / engineer / release / analyst / trainer) its python process count, their summed RSS (GiB) and the distinct script
names it runs (the live sessions: stage 2 judge chains, the port, release's safe gate / twin pipeline, truth production).
usage (from O/trainer): python -B cert/RAD/memlog_v1.py   (cert/RAD/memlog_v1.sh calls it once a minute)"""
import json
import re
import subprocess
import time
from pathlib import Path

run = lambda *a: subprocess.run(list(a), capture_output=True, text=True, timeout=20).stdout
mp = int(re.findall(r'(\d+)%', run('memory_pressure').strip().splitlines()[-1])[0])
swap = float(re.search(r'used = ([0-9.]+)M', run('sysctl', '-n', 'vm.swapusage')).group(1))
vm = run('vm_stat')
page = int(re.search(r'page size of (\d+)', vm).group(1))
pages = lambda key: int(re.search(key + r':\s+(\d+)', vm).group(1))
free_inactive = (pages('Pages free') + pages('Pages inactive')) * page / 2 ** 30
LANES = (('trainer', r'worker_generated_v\d+\.py run_RAD_|/trainer/|maintain_workers'),
         ('physicist', r'/physicist/|\bt\d+_[A-Za-z0-9_]+\.(py|sh)\b'),
         ('engineer', r'/engineer/|a62_modules|port_rad|chain_rad_port|parity_verdict'),
         ('release', r'/release/(?!guard_footprint)|s47_safe|twin_pipeline|release_gate|quick\.py'),
         ('analyst', r'/analyst/|mc_truth\.py'))
lanes = {name: dict(py=0, rss_gib=0.0, scripts=set()) for name, _ in LANES}
workers = {}
for line in run('ps', '-axo', 'rss=,command=').splitlines():
    parts = line.split(None, 1)
    if len(parts) < 2:
        continue
    rss, cmd = int(parts[0]), parts[1]
    m = re.search(r'worker_generated_v\d+\.py (run_RAD_\S+) \d+$', cmd)
    if m and 'guard_footprint' not in cmd:
        workers[m.group(1)] = workers.get(m.group(1), 0) + 1
    for name, pat in LANES:
        if re.search(pat, cmd):
            lane = lanes[name]
            if re.search(r'(^|/)python[0-9.]*\s', cmd + ' '):
                lane['py'] += 1
                lane['rss_gib'] += rss / 2 ** 20
            for s in re.findall(r'([A-Za-z0-9_.-]+\.(?:py|sh))\b', cmd):
                if not s.startswith(('guard_footprint', 'gate.sh')):
                    lane['scripts'].add(s)
            break
rec = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), mp_free=mp, swap_mib=swap, free_inactive_gib=round(free_inactive, 2),
           rad_workers=workers, lanes={k: dict(py=v['py'], rss_gib=round(v['rss_gib'], 2), scripts=sorted(v['scripts'])[:12]) for k, v in lanes.items()})
with Path('cert/RAD/memlog_v1.jsonl').open('a') as f:
    f.write(json.dumps(rec) + '\n')
