"""Root-only fixed all13 singleton-cold pairs; same source and guards, no warm qualification or submission.

Uses only the original pinned helper functions; never calls its historical main.
"""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'ISOLATED_PLAN_v1.json'
MANIFEST_SHA='392ccd079e2f1c281bab03e44e2a339e88e344e46b6901f6bce9bd19f84ad76c'
ROOT_PID=77740
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()

def main():
 ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true');args=ap.parse_args()
 assert sha(MANIFEST)==MANIFEST_SHA
 plan=json.loads(MANIFEST.read_text())
 def verify():
  for path,digest in plan['source_pins'].items():assert sha(path)==digest,('changed pin',path)
 verify()
 hp=Path(plan['isolation_helper']);assert sha(hp)==plan['isolation_helper_sha256']
 spec=importlib.util.spec_from_file_location('frozen_isolation_helpers',hp);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 out=Path(plan['quiet_output']);record=HERE/'ISOLATION_v1.json'
 assert not record.exists()
 for name in [key for key in plan['commands'] if key != 'analyze']:
  dest=out/name;assert dest.is_dir() and not any(dest.iterdir()),dest
 assert not (out/'analyze_coverage.log').exists() and not (HERE/'COVERAGE_RESULT_v1.json').exists()
 initial=h.processes();assert ROOT_PID in initial and 'run_queue_astra_v6.sh' in initial[ROOT_PID]['command']
 h.deny_external_heavy(initial,h.descendants(initial,ROOT_PID))
 if args.check:
  print(json.dumps(dict(status='READ_ONLY_READY_NO_NATIVE',manifest_sha256=sha(MANIFEST),runner_sha256=sha(__file__),commands=plan['commands'],no_release_or_fire=True)));return 0
 rec=dict(start_utc=stamp(),status='ISOLATING',manifest_sha256=sha(MANIFEST),runner_sha256=sha(__file__),helper_sha256=sha(hp),source_pins=plan['source_pins'],runs=[],no_release_or_fire=True)
 paused={};active=None;began=time.monotonic();exitcode=1
 def save():record.write_text(json.dumps(rec,indent=2)+'\n')
 def interrupt(signum,frame):raise KeyboardInterrupt('signal %s'%signum)
 signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
 def pause(pid,before):
  # Register before STOP so asynchronous interruption cannot strand this queue.
  paused[pid]=before
  os.kill(pid,signal.SIGSTOP)
 def run(cmd,log,timeout):
  nonlocal active
  with log.open('x') as stream:
   active=subprocess.Popen(cmd,cwd=h.O,stdout=stream,stderr=subprocess.STDOUT)
   try:return active.wait(timeout=timeout)
   finally:
    if active.poll() is None:active.send_signal(signal.SIGINT);active.wait(timeout=30)
    active=None
 try:
  pause(ROOT_PID,initial[ROOT_PID]);table=h.processes()
  judges=[pid for pid,r in table.items() if r['ppid']==ROOT_PID and 'run_current32_' in r['command'] and '_fable_v1.py ' in r['command']]
  for pid in judges:pause(pid,table[pid])
  rec['paused']=paused;save();print(stamp(),'DRAINING',len(judges),flush=True)
  deadline=time.monotonic()+210
  while True:
   table=h.processes();tree=h.descendants(table,ROOT_PID);h.deny_external_heavy(table,tree)
   remaining={pid for pid in tree-set(paused) if not table[pid]['state'].startswith('Z')}
   if not remaining:break
   assert time.monotonic()<deadline,remaining
   time.sleep(2)
  rec.update(drained_utc=stamp(),free_pct_before=h.pressure());save();assert rec['free_pct_before']>=60
  for label,log,timeout in [(key,out/key/'run.log',140) for key in plan['commands'] if key != 'analyze'] + [('analyze',out/'analyze_coverage.log',30)]:
   verify();table=h.processes();h.deny_external_heavy(table,h.descendants(table,ROOT_PID))
   cmd=plan['commands'][label];print(stamp(),'START',label,flush=True);rc=run(cmd,log,timeout)
   rec['runs'].append(dict(label=label,command=cmd,returncode=rc,log=str(log),end_utc=stamp()));save()
   assert rc==0,(label,rc,str(log));print(stamp(),'DONE',label,flush=True)
  rec['status']='FIXED_THIRTEEN_PARITY_RUNTIME_REVIEW_REQUIRED';exitcode=0
 except BaseException as e:
  rec.update(status='FAILED',error=repr(e));raise
 finally:
  # Cleanup errors must not prevent attempting every same-identity queue restore.
  cleanup=[];resumed=[];already_stopped=[]
  for pid,before in reversed(list(paused.items())):
   try:
    now=h.processes().get(pid)
    if now and now['started']==before['started'] and now['command']==before['command']:
     if before['state'].startswith('T'):already_stopped.append(pid)
     else:os.kill(pid,signal.SIGCONT);resumed.append(pid)
   except ProcessLookupError:pass
   except BaseException as e:cleanup.append(dict(pid=pid,error=repr(e)))
  rec.update(end_utc=stamp(),pause_elapsed_s=time.monotonic()-began,resumed=resumed,previously_stopped_preserved=already_stopped,cleanup_errors=cleanup,exitcode=exitcode)
  try:rec['free_pct_after']=h.pressure()
  except Exception as e:rec['free_pct_after_error']=repr(e)
  try:save()
  finally:print(rec['end_utc'],'RESEARCH_RESTORED',resumed,'prior_stops',already_stopped,'errors',cleanup,'exitcode',exitcode,flush=True)
 return exitcode

if __name__=='__main__':raise SystemExit(main())
