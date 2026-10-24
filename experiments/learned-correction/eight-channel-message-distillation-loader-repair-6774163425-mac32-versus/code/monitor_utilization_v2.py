"""Read-only VM load sampling; timestamps come from date -u."""
import argparse
import json
from pathlib import Path
import subprocess
import time

def ticks():
    return list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))

def snapshot(window, start):
    end = ticks()
    delta = [b-a for a,b in zip(start,end)]
    memory = {line.split(':')[0]:int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    servers = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                argv=(p/'cmdline').read_bytes().split(b'\0')
                if any(q.rsplit(b'/',1)[-1] in
                       (b'server_linux_v1.py',b'server_linux_v2.py',
                        b'server_linux_v3.py',b'server_linux_v4.py') for q in argv):
                    servers.append(int(p.name))
            except (FileNotFoundError, PermissionError):
                pass
    return dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
        window_s=window,cpu_busy_pct=100*(1-(delta[3]+delta[4])/sum(delta)),
        servers=len(servers),available_GiB=memory['MemAvailable']/2**30,
        swap_used=memory['SwapTotal']-memory['SwapFree'],
        count_method='exact server_linux_v1..4 argv basename')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=1800)
    p.add_argument('--window',type=float,default=20)
    a=p.parse_args()
    assert a.samples>0 and 1<=a.window<=60
    for _ in range(a.samples):
        start=ticks()
        time.sleep(a.window)
        row=snapshot(a.window,start)
        with a.output.open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

if __name__=='__main__':
    main()

