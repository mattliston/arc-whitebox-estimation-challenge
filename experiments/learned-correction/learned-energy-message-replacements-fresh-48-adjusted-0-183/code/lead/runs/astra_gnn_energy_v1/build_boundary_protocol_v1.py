"""Held scheduling-only source; preserves active worker and original protocol."""
from pathlib import Path
import json,hashlib,ast
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=D/'FOLLOWUP_PREREG_v2.json';c=json.loads(base.read_text());assert sha(base)=='258eb40f5e4c3fca3b447d6dc73518782397585f926d349067f9d9815ff8912b'
run=D/c['runs']['fresh'];owner=run/'OWNER_v1.json';started=run/'STARTED_v1.json'
assert json.loads(started.read_text())['guard_pids']==[85281]
pins=dict(c['source_pins'])
for p in [base,D/'finite_energy_resume_UNVALIDATED_v2.py',D/'FRESH_ACTIVE_SESSION_RECEIPT_v1.json',run/'MANIFEST_v1.json',owner,started,D/'ROOT_FRESH_ONE_SLOT_GRANT_v1.json']:
 pins[str(p)]=sha(p)
for p,h in pins.items():assert sha(p)==h
ast.parse((D/'finite_energy_resume_UNVALIDATED_v2.py').read_text())
p=D/'FRESH_BOUNDARY_PROTOCOL_DRAFT_v1.json'
r=dict(status='HELD_SCHEDULING_ONLY_NO_STOP_RESIZE_OR_NEW_NUMERICAL_GRANT',base_protocol=dict(path=str(base),sha256=sha(base)),pins=pins,canonical_run=str(run),segment_name='segment2_v1',original_controller_pid=85259,original_session_id=85735,original_guard_pids=[85281],hold_path=str(D/'ROOT_FRESH_BOUNDARY_HOLD_v1.json'),drain_path=str(D/'ROOT_FRESH_BOUNDARY_DRAINED_v1.json'),
  timing='Root source review and explicit hold receipt BEFORE STOP, recording original controller+expected guards live, current completed hashes and noerrors. Holdmtime precedes original retirement. No stop or resize from this source package.',
  original_partial='Original helper may assert exit1 only AFTER clean guard0 retirement writes incomplete record. Require original completedfalse, exceptionNone,errorsempty,allownedgroupsretired,allguardexit0,explicit root drained controllerexit1/reaped/emptyremaining. Never reinterpret unplanned failures.',
  preserved='Same canonical fixed48 manifest, source, requests, claims, results, seeds1112-1127, C/E3/E23, Torch2 profile and original full core. Prior completed bytes frozen and rehashed. Only queued unclaimed remainder consumed; no retry/newseed/tuning.',
  owners='Original owner/STARTED/retirement/guards immutable. New segment2 gets separate owner/STARTED/guard/retirement/lock files; exact expected set checked. One additional resume only, later failure needs new explicit protocol.',
  grading='Private imported original SCI.clean replaced only forfresh by aggregate verified original intentional boundary and fullcomplete segment2. Original SCI.grade/read_result/price/plan unchanged. Both root-frozen actualMAX prices before anymean; full48 and every ownerretired. OOS interfaces reuse aggregatefresh gate then originalOOS gate; separate OOS grant remains required.',
  future_consumers='After boundary use this reviewed wrapper forfreshgrade and OOS prepare/launch/grade; matched-C parity v3 will import same aggregategate under explicit root resume-review evidence. Original direct freshgrader intentionally fails on incomplete firstsegment.',
  resources='One separately root-granted new segment pool1/2/3 only after original guardedworker/controller retired. Controllernice0, all7outerBLAS1, child6BLAS2+TRAINER2, guardnice10once, free>=50/count+reserve<=10 admission/live10s, ownSIGINTbounded30s, retainreservation/lock ifunverified.',
  science_unchanged=True,numerical_work_performed=False,stop_written=False)
with p.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(protocol_sha256=sha(p),helper_sha256=sha(D/'finite_energy_resume_UNVALIDATED_v2.py'),pins=len(pins))))
