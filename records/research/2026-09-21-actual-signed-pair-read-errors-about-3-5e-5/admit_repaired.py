"""Reverify the additive diagnostic repair and retire-check before execution."""
from datetime import datetime,timezone
import os,shutil,subprocess
from common import *
seal_check()
for p in read(HERE/'repair_sources_sealed.json')['pins']:verify(p)
failed=read(HERE/'unit_watch.json')
assert failed['exit_code']==1 and failed['whole_group_retired'] and failed['max_swap_growth_bytes']==0
active=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,command='],text=True).splitlines():
    t=line.strip().split(None,2)
    if len(t)==3 and (int(t[0])==failed['pid'] or int(t[1])==failed['pid'] or
        (int(t[0])!=os.getpid() and any(x in t[2] for x in ('openblas_runtime_env_v1/bin/python',str(D/'run_guarded.py'),str(HERE/'run.py'),str(HERE/'run_v2.py'))))):active.append(line)
assert not active,active
write(HERE/'repaired_preflight.json',dict(utc=datetime.now(timezone.utc).isoformat(),all_repair_pins_verified=True,
    original_failed_group_retired=True,numerical_workers=[],disk_free_bytes=shutil.disk_usage(HERE).free,
    swap=subprocess.check_output(['/usr/sbin/sysctl','-n','vm.swapusage'],text=True).strip(),
    source=pin(Path(__file__)),old_controller_terminal_session=93145))
raise SystemExit(subprocess.run([sys.executable,'-B',str(HERE/'run_v2.py')]).returncode)
