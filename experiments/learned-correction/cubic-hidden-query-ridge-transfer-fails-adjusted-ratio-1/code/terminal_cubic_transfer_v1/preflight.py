from datetime import datetime,timezone
import os,shutil,subprocess
from common import *
seal_check();rows=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)==4 and int(t[0])!=os.getpid() and any(m in t[3] for m in [str(RUNTIME),str(D/'run_guarded.py'),str(HERE/'run.py')]):rows.append(line)
assert not rows,rows
free=shutil.disk_usage(HERE).free;assert free>100_000_000_000
write(HERE/'preflight_resources.json',dict(utc=datetime.now(timezone.utc).isoformat(),numerical_workers=rows,
    swap=subprocess.check_output(['/usr/sbin/sysctl','-n','vm.swapusage'],text=True).strip(),disk_free_bytes=free))
print(dict(passed=True,disk_free_bytes=free),flush=True)
