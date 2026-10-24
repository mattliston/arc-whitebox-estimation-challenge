"""Inert mocked monitor certificate: no real process, signal, model or array."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROTOCOL_SHA = '968910dc63ba1c090665322c945b4e7db3b2978b1a059baa10bab13ea2dc3af7'
MONITOR_SHA = 'd4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873'
ORIGINAL = HERE.parent/'current_d50_native_objective_p1_v1/watch.py'
ORIGINAL_SHA = '4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value: raise RuntimeError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now


class Process:
    def __init__(self, pid, argv, clock, finished, output, events, code=0):
        self.pid,self.argv,self.clock,self.finished,self.output,self.events,self.code = pid,argv,clock,finished,output,events,code
        self.returncode = None
    def communicate(self, timeout):
        self.events.append(('communicate',self.pid))
        if self.finished > self.clock.now+timeout:
            self.clock.now += timeout
            raise subprocess.TimeoutExpired(self.argv,timeout)
        self.clock.now = max(self.clock.now,self.finished); self.returncode = self.code
        return self.output,''
    def poll(self): return self.returncode
    def wait(self, timeout):
        self.events.append(('wait',self.pid)); require(self.returncode is not None,'Mock command was terminated/reaped')
        return self.returncode


def fixture(ps_finish=.65, swap_finish=.08, code=0):
    clock,events,children = Clock(),[],[]
    def popen(argv, **kwargs):
        require(kwargs == dict(stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               text=True,start_new_session=True),'Exact isolated observation launch')
        i = len(children); require(i < 2,'No observational retry')
        value = Process(2101+i,argv,clock,(ps_finish,swap_finish)[i],
            ('111 10 111 1024 /watcher\n222 111 222 2048 /worker\n',
             'vm.swapusage: total = 1.00G used = 64.00M free = 960.00M')[i],events,code if i == 0 else 0)
        events.append(('launch',value.pid)); children.append(value); return value
    def terminate(commands, deadline, **kwargs):
        events.append(('cleanup',len(commands)))
        for child in commands: child.returncode = -9; child.wait(timeout=.1)
    return clock,events,children,popen,terminate


def checks(m, old):
    passed = []
    clock,events,children,popen,terminate = fixture()
    result = m.collect(1.,clock=clock,popen=popen,terminate=terminate,watcher_pid=111,alive=lambda _:False)
    require(events[:2] == [('launch',2101),('launch',2102)] and len(children) == 2
            and result['commands_reaped'] and result['one_shared_process_table']
            and result['retry_count'] == 0 and not result['cached_sample_used'],'Both commands overlap without retry/cache')
    clock.now = .8
    require(m.complete(0.,1.,clock=clock) == (.8,.8),'Delayed>.4s observation still completes inside1s')
    passed.append('concurrent_delayed_within_actual_1s')

    clock,events,children,popen,terminate = fixture(ps_finish=1.01)
    try: m.collect(1.,clock=clock,popen=popen,terminate=terminate,watcher_pid=111,alive=lambda _:False)
    except subprocess.TimeoutExpired: pass
    else: raise RuntimeError('Timeout was not fatal')
    require(len(children) == 2 and ('cleanup',2) in events and all(x.poll() is not None for x in children)
            and clock.now == .9,'Reserved deadline triggers owned cleanup; no late retry')
    passed.append('timeout_no_retry_and_owned_reap')

    clock.now = 1.001
    try: m.complete(0.,1.,clock=clock)
    except RuntimeError: pass
    else: raise RuntimeError('Late logging/completion reset its deadline')
    passed.append('late_finalization_is_failure')

    clock,events,children,popen,terminate = fixture(code=2)
    try: m.collect(1.,clock=clock,popen=popen,terminate=terminate,watcher_pid=111,alive=lambda _:False)
    except RuntimeError: pass
    else: raise RuntimeError('Nonzero command accepted')
    require(('cleanup',2) in events,'Both recorded commands cleaned after command failure')
    passed.append('nonzero_command_is_failure')

    rows = m.parse_rows('10 1 10 1 /parent\n111 10 111 2 /watcher\n'
        '222 111 222 3 /worker\n333 222 222 4 /nested\n444 1 222 5 /reparented\n'
        '999 1 999 6 /Users/matt/anaconda3/bin/python '+str(old.NEW/'foreign.py')+'\n',111)
    scoped = m.scope(rows,111,222)
    require(scoped['child_pids'] == [222,333,444] and scoped['combined_rss_bytes'] == 14*1024
            and scoped['child_tree_rss_bytes'] == 12*1024 and {10,1,111} <= scoped['allowed'],
            'Same child/group/descendant/ancestor resource scope')
    require([x['pid'] for x in old.other_workers(rows,scoped['allowed'])] == [999],
            'Frozen foreign-worker classifier sees only the foreign process')
    passed.append('single_snapshot_tree_and_foreign_inventory')

    try: m.parse_rows('111 1 111 1 /a\n111 1 111 2 /b\n',111)
    except RuntimeError: pass
    else: raise RuntimeError('Ambiguous process table accepted')
    passed.append('malformed_duplicate_process_table_rejected')

    clock,events,children,popen,_ = fixture()
    popen(list(m.PS),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    popen(list(m.SWAP),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    killed,live_groups = [],{2101,2102}
    def killpg(pid, sig):
        require(pid in (2101,2102),'Never signal another process/group')
        killed.append((pid,int(sig))); children[pid-2101].returncode = -9
        live_groups.discard(pid)
    def getpgid(pid):
        if pid == 2102: raise ProcessLookupError('Mock leader-exit race')
        return pid
    m.cleanup(children,1.,clock=clock,killpg=killpg,getpgid=getpgid,own_pid=lambda:111,own_group=lambda:111,
              alive=lambda pid:pid in live_groups,pause=lambda dt:setattr(clock,'now',clock.now+dt))
    require([x[0] for x in killed] == [2101,2102] and all(x.poll() is not None for x in children),
            'Recorded-only group cleanup tolerates leader-exit race and reaps both')
    passed.append('owned_group_cleanup_and_exit_race')
    killed.clear(); live_groups.update((2101,2102))
    for child in children: child.returncode = 0  # Leaders exited, groups still live.
    m.cleanup(children,1.,clock=clock,killpg=killpg,getpgid=getpgid,own_pid=lambda:111,own_group=lambda:111,
              alive=lambda pid:pid in live_groups,pause=lambda dt:setattr(clock,'now',clock.now+dt))
    require([x[0] for x in killed] == [2101,2102] and not live_groups,'Exited leaders do not release live group ownership')
    passed.append('exited_leader_live_group_is_retired')
    require(not m.PENDING,'No pending observational command after mocked success/failures')
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for key in ('source','monitor','protocol'): parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run: parser.print_help(); return 0
    began = time.monotonic(); out = HERE/'monitor_tests_v1'
    sources = {str(Path(__file__).resolve()):args.source_sha256,str(HERE/'monitor.py'):MONITOR_SHA,
               str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(ORIGINAL):ORIGINAL_SHA}
    require(args.monitor_sha256 == MONITOR_SHA and args.protocol_sha256 == PROTOCOL_SHA
            and all(sha(p) == v for p,v in sources.items()),'Every reviewed source pin')
    require(sys.flags.optimize == 0 and not out.exists(),'Assertions and exclusive mock output')
    out.mkdir(exist_ok=False)
    result = dict(schema='current_p1_native_objective_p2_monitor_tests_v2',passed=False,completed=False,
        source_bindings=sources,real_subprocesses_started=0,real_signals_sent=0,numerical_imports=False,
        model_accessed=False,arrays_accessed=False,targets_accessed=False,checks=[])
    try:
        m,old = load(HERE/'monitor.py','p2_repair_mock_monitor'),load(ORIGINAL,'p2_original_inventory_classifier')
        result['checks'] = checks(m,old)
        require(len(result['checks']) == 8 and not any(x in sys.modules for x in ('numpy','torch','flopscope')),
                'Exactly eight mocked checks; no numerical import')
        require(all(sha(p) == v for p,v in sources.items()),'Sources unchanged after mocks')
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if result['elapsed_s'] >= 5 or result['peak_rss_bytes'] >= 256_000_000:
        result.update(passed=False,completed=False)
    for name,value in (('result.json',result),('completion.json',None)):
        if value is None: value = dict(passed=result['passed'],completed=result['completed'],result_sha256=sha(out/'result.json'))
        with (out/name).open('x') as stream: json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
