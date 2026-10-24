"""One privileged move of a verified, pipe-blocked owned research child."""
import os
from pathlib import Path
import pwd
import sys
assert os.geteuid()==0
cg=Path(sys.argv[1]).resolve();pid=int(sys.argv[2]);ticks=int(sys.argv[3]);guard=int(sys.argv[4])
assert cg.parent==Path('/sys/fs/cgroup/arc-estimator-v5') and cg.name.startswith('arc-'+str(guard)+'-')
page=os.sysconf('SC_PAGE_SIZE')
limit=int((cg/'memory.max').read_text())
assert abs(limit-6500000000)<page, ('kernel page rounding',limit,page)
assert (cg/'memory.swap.max').read_text().strip()=='0'
uid=pwd.getpwnam('arcworker').pw_uid
for target in (pid,guard):
 lines={l.split(':',1)[0]:l.split(':',1)[1].strip() for l in Path('/proc/'+str(target)+'/status').read_text().splitlines()}
 assert int(lines['Uid'].split()[0])==uid
 if target==pid:assert int(lines['PPid'])==guard
raw=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
assert int(raw[19])==ticks and os.getpgid(pid)==pid
assert b'/guard_linux_v7.py' in Path('/proc/'+str(guard)+'/cmdline').read_bytes()
(cg/'cgroup.procs').write_text(str(pid))
assert '/arc-estimator-v5/'+cg.name in Path('/proc/'+str(pid)+'/cgroup').read_text()
