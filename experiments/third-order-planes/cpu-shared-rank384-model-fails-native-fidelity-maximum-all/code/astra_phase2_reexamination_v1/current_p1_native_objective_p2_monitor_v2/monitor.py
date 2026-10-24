"""Stdlib observation transport. No cache/retry; one absolute sample deadline."""
import os
import re
import signal
import subprocess
import time

RESERVE = .10
MAX_INTERVAL = 1.
PS = ('/bin/ps','-axo','pid=,ppid=,pgid=,rss=,command=')
SWAP = ('/usr/sbin/sysctl','vm.swapusage')
PENDING = []


def require(value, message):
    if not value: raise RuntimeError(message)


def parse_rows(raw, watcher_pid):
    rows = []
    for line in raw.splitlines():
        if not line.strip(): continue
        fields = line.split(None,4)
        require(len(fields) == 5,'Complete process command/RSS record')
        pid,parent,group,rss = map(int,fields[:4])
        require(pid > 0 and parent >= 0 and group >= 0 and rss >= 0,'Valid process identity/RSS')
        rows.append(dict(pid=pid,ppid=parent,pgid=group,rss_bytes=rss*1024,command=fields[4]))
    require(len({x['pid'] for x in rows}) == len(rows) and any(x['pid'] == watcher_pid for x in rows),
            'Unique process table containing the watcher')
    return rows


def parse_swap(raw):
    value = raw.strip(); match = re.search(r'used\s*=\s*([\d.]+)([KMGT])',value)
    require(match is not None,'Complete vm.swapusage response')
    return dict(raw=value,used_bytes=float(match[1])*1024**('KMGT'.index(match[2])+1))


def descendants(seeds, rows):
    result = set(seeds)
    while True:
        expanded = result | {x['pid'] for x in rows if x['ppid'] in result}
        if expanded == result: return result
        result = expanded


def scope(rows, watcher_pid, child_pid):
    """Same tree/group/ancestor algebra as frozen P1 observe, on ONE table."""
    owned = set()
    if child_pid is not None:
        owned = descendants({child_pid},rows) | {x['pid'] for x in rows if x['pgid'] == child_pid}
        owned = descendants(owned,rows)
    combined = descendants({watcher_pid},rows) | owned
    allowed = combined | {watcher_pid}
    parents = {x['pid']:x['ppid'] for x in rows}; ancestor = watcher_pid
    while ancestor in parents and parents[ancestor] not in allowed:
        ancestor = parents[ancestor]; allowed.add(ancestor)
    return dict(combined_rss_bytes=sum(x['rss_bytes'] for x in rows if x['pid'] in combined),
        watcher_rss_bytes=sum(x['rss_bytes'] for x in rows if x['pid'] == watcher_pid),
        child_tree_rss_bytes=sum(x['rss_bytes'] for x in rows if x['pid'] in owned),
        child_pids=sorted(x['pid'] for x in rows if x['pid'] in owned),allowed=allowed)


def group_alive(pid):
    try:
        os.killpg(pid,0)
        return True
    except ProcessLookupError:
        return False


def cleanup(commands, deadline, *, clock=time.monotonic, killpg=os.killpg,
            getpgid=os.getpgid, own_pid=os.getpid, own_group=os.getpgrp,
            alive=group_alive, pause=time.sleep):
    """Only the two recorded observational-command groups; no process search."""
    errors = []
    # Failure cleanup may need a final bounded reap AFTER an expired sample.
    # This never changes collect/complete's deadline or permits a late PASS.
    reap_deadline = max(deadline,clock()+.10)
    for child in commands:
        try:
            require(child.pid > 1 and child.pid != own_pid() and child.pid != own_group(), 'Safe owned command group')
            if child.poll() is None:
                try: require(getpgid(child.pid) == child.pid,'Unchanged command process group')
                except ProcessLookupError: pass
            # The recorded group may outlive its already-reaped leader.
            try: killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        except BaseException as exc:
            errors.append(repr(exc))
    for child in commands:
        try:
            remaining = reap_deadline-clock()
            require(remaining > 0 or child.poll() is not None,'Command cleanup deadline')
            child.wait(timeout=max(remaining,0.))
            require(child.poll() is not None,'Owned observational command reaped')
            while alive(child.pid) and clock() < reap_deadline:
                pause(min(.005,max(0.,reap_deadline-clock())))
            require(not alive(child.pid),'Owned observational group retired, not just its leader')
        except BaseException as exc:
            errors.append(repr(exc))
    require(not errors,'Owned command cleanup failed: '+'; '.join(errors))


def collect(deadline, *, clock=time.monotonic, popen=subprocess.Popen,
            terminate=cleanup, watcher_pid=None, alive=group_alive):
    """Two commands are launched before either is awaited; deadline never moves."""
    watcher_pid = os.getpid() if watcher_pid is None else watcher_pid
    started = clock(); acquisition_deadline = deadline-RESERVE
    require(started < acquisition_deadline,'No acquisition time left before reserved completion margin')
    commands, outputs, spans = [],[],[]
    try:
        for argv in (PS,SWAP):
            require(clock() < acquisition_deadline,'Observation launch exceeded absolute acquisition deadline')
            commands.append(popen(list(argv),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                  text=True,start_new_session=True))
            PENDING.append(commands[-1])
        for child,argv in zip(commands,(PS,SWAP)):
            remaining = acquisition_deadline-clock()
            require(remaining > 0,'Observation acquisition deadline expired')
            out,err = child.communicate(timeout=remaining)
            finished = clock()
            require(finished < acquisition_deadline and child.returncode == 0 and len(out) <= 8*1024**2,
                    'Complete bounded observational command before absolute deadline')
            outputs.append(out)
            spans.append(dict(argv=list(argv),pid=child.pid,finished_since_start_s=finished-started))
        rows,swap = parse_rows(outputs[0],watcher_pid),parse_swap(outputs[1])
        require(all(x.poll() is not None and not alive(x.pid) for x in commands),
                'Every observational group retired before accepting the sample')
        require(clock() < deadline,'Parsing consumed the remaining sample deadline')
        return dict(rows=rows,swap=swap,command_records=spans,started_monotonic_s=started,
            acquired_monotonic_s=clock(),absolute_deadline_monotonic_s=deadline,
            reserved_completion_margin_s=RESERVE,commands_reaped=all(x.poll() is not None for x in commands),
            one_shared_process_table=True,retry_count=0,cached_sample_used=False)
    except BaseException:
        terminate(commands,deadline,clock=clock)
        raise
    finally:
        # A finished leader is NOT evidence that its recorded group retired.
        retained = []
        for child in PENDING:
            try: retired = child.poll() is not None and not alive(child.pid)
            except BaseException: retired = False
            if not retired: retained.append(child)
        PENDING[:] = retained


def retire_pending(deadline):
    """Failure-only bounded cleanup; never a sample retry or deadline reset."""
    cleanup(list(PENDING),deadline)
    PENDING[:] = [x for x in PENDING if x.poll() is None or group_alive(x.pid)]
    require(not PENDING,'All recorded observational commands retired')


def complete(previous, deadline, *, clock=time.monotonic):
    """Call AFTER progress, checks and log flush; only success advances time."""
    now = clock(); interval = now-previous
    require(now < deadline and 0 <= interval <= MAX_INTERVAL,'Actual completed-sample interval<=1s')
    return now,interval
