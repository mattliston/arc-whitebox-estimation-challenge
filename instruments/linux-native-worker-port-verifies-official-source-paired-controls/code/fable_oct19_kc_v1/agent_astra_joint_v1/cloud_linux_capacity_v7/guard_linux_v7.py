"""Linux-only6.5GB group guard with atomic admission and full owned-group cleanup.

Optional delegated cgroup-v2 is preferred. Without it, summed process-tree RSS
and per-process swap are sampled; this fallback is explicitly not a kernel hard
memory limit or macOS physical footprint. No source/budget/timing rules change.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import resource
import subprocess
import sys
import time
from common_linux_v7 import THREADS, PORT_V1, thread_provenance, proc_stat, sha, utc, write_new


def processes():
    result = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                row = proc_stat(int(p.name))
                if row['state'] != 'Z':
                    result[row['pid']] = row
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    return result


def members(table, pgid, known):
    group_identity_ok = pgid not in table or known.get(pgid, table[pgid]['start_ticks']) == table[pgid]['start_ticks']
    found = {pid for pid, row in table.items() if (group_identity_ok and row['pgid'] == pgid)
             or known.get(pid) == row['start_ticks']}
    while True:
        extra = {pid for pid, row in table.items() if row['ppid'] in found} - found
        if not extra:
            break
        found |= extra
    for pid in found:
        known[pid] = table[pid]['start_ticks']
    return {pid: table[pid] for pid in found}


def memory():
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, v = line.split(':', 1); fields[k] = int(v.strip().split()[0]) * 1024
    total = fields['MemTotal']; available = fields['MemAvailable']
    # Respect enclosing cgroup limits as well as the physical VM.
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            current = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
            while current.is_relative_to('/sys/fs/cgroup'):
                try:
                    cap = (current / 'memory.max').read_text().strip()
                    used = int((current / 'memory.current').read_text())
                    if cap != 'max':
                        total = min(total, int(cap)); available = min(available, max(0, int(cap) - used))
                except (FileNotFoundError, PermissionError):
                    pass
                if current == Path('/sys/fs/cgroup'):
                    break
                current = current.parent
    return dict(total=total, available=available)


def swapped(owned):
    amount = 0
    for pid in owned:
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmSwap:'):
                    amount += int(line.split()[1]) * 1024
        except FileNotFoundError:
            pass
    return amount


def cgroup_events(path):
    if path is None:
        return None
    return {k: int(v) for k, v in (line.split() for line in (path / 'memory.events').read_text().splitlines())}


def signal_owned(pgid, known, sig):
    table = processes(); owned = members(table, pgid, known)
    if any(row['pgid'] == pgid for row in owned.values()):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
    # Also cover an owned descendant which changed its process group.
    for pid, row in owned.items():
        if row['pgid'] != pgid:
            try:
                now = proc_stat(pid)
                if now['start_ticks'] == known[pid]:
                    os.kill(pid, sig)
            except (ProcessLookupError, FileNotFoundError):
                pass


class Admission:
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.lock = root / 'admission.lock'; self.state = root / 'admission.json'

    def update(self, own, remove=False, cap=None, max_active=None, reserve=None, reservation=None):
        with self.lock.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = json.loads(self.state.read_text()) if self.state.exists() else {}
            table = processes(); clean = {}
            for key, rec in records.items():
                pid = int(key)
                if pid in table and table[pid]['start_ticks'] == rec['start_ticks']:
                    clean[key] = rec
            key = str(os.getpid()); admitted = False; info = memory()
            if remove:
                clean.pop(key, None)
            elif key in clean:
                clean[key].update(own); admitted = True
            else:
                outstanding = 0
                for rec in clean.values():
                    active_rss = sum(r['rss_bytes'] for r in table.values() if r['pgid'] == rec.get('pgid'))
                    outstanding += max(0, rec.get('admission_reservation_bytes', rec['cap']) - active_rss)
                reserved_budget = cap if reservation is None else reservation
                assert 0 < reserved_budget <= cap
                if len(clean) < max_active and info['available'] - outstanding >= reserved_budget + reserve:
                    clean[key] = dict(own, cap=cap, admission_reservation_bytes=reserved_budget); admitted = True
                info.update(active=len(clean), outstanding_reservation=outstanding)
            tmp = self.state.with_name('admission.tmp')
            tmp.write_text(json.dumps(clean, indent=2) + '\n'); os.replace(tmp, self.state)
            return admitted, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receipt', type=Path, required=True); p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--cgroup-root', type=Path, help='existing delegated cgroup-v2 root; fail if unusable')
    p.add_argument('--max-active', type=int, required=True, choices=range(1, 37))
    p.add_argument('--cap-bytes', type=int, default=6500000000)
    p.add_argument('--reserve-bytes', type=int, default=19000000000)
    p.add_argument('--admission-reservation-bytes', type=int, default=4800000000, choices=(4800000000,), help='empirical current32-family reservation; retirement cap remains6.5GB')
    p.add_argument('--reserve-fraction', type=float, default=.10)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--admission-timeout', type=float, default=7200)
    p.add_argument('--poll', type=float, default=.20)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    threads = thread_provenance()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    assert sys.platform == 'linux', 'Linux /proc required'
    assert a.cgroup_root is not None, 'v5 requires kernel per-group and parent memory limits'
    assert int((a.cgroup_root / 'memory.max').read_text()) <= 160 * 2**30, 'parent aggregate cap required'
    assert (a.cgroup_root / 'memory.swap.max').read_text().strip() == '0'
    assert a.cap_bytes == 6500000000, 'frozen empirical group limit is6.5GB'
    assert a.reserve_bytes >= 19000000000 and a.reserve_fraction >= .10 and .05 <= a.poll <= .5
    cmd = a.command[1:] if a.command and a.command[0] == '--' else a.command
    assert cmd and not a.receipt.exists()
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    admission = Admission(a.state_dir); me = proc_stat(os.getpid())
    own = dict(start_ticks=me['start_ticks'], pgid=None, blas_threads=threads['blas_threads'])
    child = None; cg = None; known = {}; reason = None; start = time.monotonic()
    peak_rss = peak_cg = max_swap = 0; lowest_available = None; samples = 0
    initial_events = final_events = None; reserved = False; info = None; cleanup_errors = []
    def interrupted(signum, _):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            mem = memory(); reserve = max(a.reserve_bytes, int(mem['total'] * a.reserve_fraction))
            reserved, info = admission.update(own, cap=a.cap_bytes, max_active=a.max_active, reserve=reserve, reservation=a.admission_reservation_bytes)
            if reserved:
                break
            if time.monotonic() - start > a.admission_timeout:
                raise TimeoutError('admission timeout')
            time.sleep(2)
        if a.cgroup_root is not None:
            assert a.cgroup_root.is_dir(), 'delegated cgroup root absent'
            cg = a.cgroup_root / f'arc-{os.getpid()}-{me["start_ticks"]}'
            cg.mkdir()
            (cg / 'memory.max').write_text(str(a.cap_bytes))
            (cg / 'memory.swap.max').write_text('0')
            (cg / 'memory.oom.group').write_text('1')
            initial_events = cgroup_events(cg)
        read_fd, write_fd = os.pipe()
        entry = PORT_V1 / 'guard_entry_linux_v1.py'
        try:
            child = subprocess.Popen([sys.executable, '-B', str(entry), str(read_fd), *cmd],
                                     env=dict(os.environ, **THREADS), start_new_session=True, pass_fds=(read_fd,))
        finally:
            os.close(read_fd)
        own['pgid'] = child.pid; admission.update(own)
        known[child.pid] = proc_stat(child.pid)['start_ticks']
        try:
            if cg is not None:
                mover = Path(__file__).resolve().parent / 'move_owned_child_v7.py'
                assert sha(mover) == 'eb7078ae9f473026dcf32a37b5b64fc51bf67d8b7360bfcfdc0ca66a8329594a'
                subprocess.run(['sudo', '-n', 'python3', str(mover), str(cg), str(child.pid), str(known[child.pid]), str(os.getpid())], check=True, timeout=10)
            os.write(write_fd, b'1')
        finally:
            os.close(write_fd)
        running = time.monotonic()
        while child.poll() is None:
            owned = members(processes(), child.pid, known)
            rss = sum(row['rss_bytes'] for row in owned.values()); swap = swapped(owned)
            mem = memory(); current = int((cg / 'memory.current').read_text()) if cg else 0
            peak_rss = max(peak_rss, rss); peak_cg = max(peak_cg, current); max_swap = max(max_swap, swap)
            lowest_available = mem['available'] if lowest_available is None else min(lowest_available, mem['available'])
            samples += 1
            if rss > a.cap_bytes or current > a.cap_bytes:
                reason = 'group_memory_limit'; break
            if swap:
                reason = 'owned_process_swap'; break
            if mem['available'] < reserve:
                reason = 'global_available_memory_reserve'; break
            if time.monotonic() - running > a.timeout:
                reason = 'group_timeout'; break
            if cg:
                ev = cgroup_events(cg)
                if ev.get('oom', 0) > initial_events.get('oom', 0) or ev.get('oom_kill', 0) > initial_events.get('oom_kill', 0):
                    reason = 'cgroup_oom'; break
            time.sleep(a.poll)
        if reason is None and child.returncode != 0:
            reason = 'child_nonzero_exit'
    except BaseException as exc:
        reason = reason or repr(exc)
    finally:
        if child is not None:
            try:
                signal_owned(child.pid, known, signal.SIGTERM)
                for _ in range(40):
                    if not members(processes(), child.pid, known):
                        break
                    time.sleep(.1)
                if members(processes(), child.pid, known):
                    if cg is not None and (cg / 'cgroup.kill').exists():
                        (cg / 'cgroup.kill').write_text('1')
                    signal_owned(child.pid, known, signal.SIGKILL)
            except BaseException as exc:
                cleanup_errors.append('signal cleanup: ' + repr(exc))
                # Retry through the same PID/start-time ownership check.
                try:
                    signal_owned(child.pid, known, signal.SIGKILL)
                except BaseException as retry:
                    cleanup_errors.append('kill retry: ' + repr(retry))
            try:
                child.wait(timeout=10)
            except BaseException as exc:
                cleanup_errors.append('child wait: ' + repr(exc))
        try:
            retired = child is None or not members(processes(), child.pid, known)
        except BaseException as exc:
            retired = False; cleanup_errors.append('retirement read: ' + repr(exc))
        if not retired:
            reason = reason or 'owned_group_not_retired'
        if cg is not None:
            try:
                final_events = cgroup_events(cg)
                if initial_events is not None and any(final_events.get(k, 0) > initial_events.get(k, 0) for k in ('oom', 'oom_kill')):
                    reason = reason or 'cgroup_oom'
                cg.rmdir()
            except BaseException as exc:
                cleanup_errors.append('cgroup cleanup: ' + repr(exc))
        if reserved:
            try:
                admission.update(own, remove=True)
            except BaseException as exc:
                cleanup_errors.append('admission release: ' + repr(exc))
        if cleanup_errors:
            reason = reason or 'cleanup_error'
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = dict(utc=utc(), guard_sha256=sha(__file__), command=cmd, successful=reason is None,
                      thread_provenance=threads,
                      child_user_s=usage_after.ru_utime-usage_before.ru_utime,
                      child_system_s=usage_after.ru_stime-usage_before.ru_stime,
                      reason=reason, cleanup_errors=cleanup_errors, child_exit=child.returncode if child else None, whole_group_retired=retired,
                      group=child.pid if child else None, peak_group_rss_bytes=peak_rss,
                      peak_cgroup_memory_bytes=peak_cg if cg else None, max_owned_swap_bytes=max_swap,
                      min_available_bytes=lowest_available, admission=info, samples=samples,
                      wall_s=time.monotonic() - start, cap_bytes=a.cap_bytes,
                      reserve_bytes=max(a.reserve_bytes, int(memory()['total'] * a.reserve_fraction)),
                      max_active=a.max_active, admission_reservation_bytes=a.admission_reservation_bytes,
                      reservation_basis='empirical4.8GB admission; kernel6.5GB pergroup and160GiB parent caps plus19GB physical reserve; simultaneous peaks may retire jobs' ,
                      cgroup=bool(cg), cgroup_events_initial=initial_events,
                      cgroup_events_final=final_events,
                      metric='cgroup-v2 memory.current + process-tree RSS' if cg else 'sampled process-tree RSS; not a kernel hard cap; no macOS footprint equivalence')
        write_new(a.receipt, result)
        print(json.dumps(result), flush=True)
    return 0 if reason is None else 1


if __name__ == '__main__':
    sys.exit(main())
