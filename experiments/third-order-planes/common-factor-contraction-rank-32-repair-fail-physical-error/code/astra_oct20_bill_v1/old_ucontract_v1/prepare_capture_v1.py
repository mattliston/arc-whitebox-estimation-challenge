"""Freeze two weights-only singleton observations; old transport/guard unchanged."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'breakthrough_v1';ROOT=next(p for p in HERE.parents if(p/'HANDOFF_CURRENT.md').exists())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
price=json.loads((HERE/'PRICE_CURRENT32_v1.json').read_text());assert price['mean_bill_delta_pct']<-.3
test=json.loads((HERE/'OBSERVER_CHECK_v1.json').read_text());assert test['successful']and test['hook_sha256']==sha(HERE/'old_su_hook_v1.py')
plan=json.loads((OLD/'PLAN_v1.json').read_text());assert plan['groups']==[[35],[150]]and all(set(r)=={'weights'}for r in plan['rows'].values())
plan.update(utc=now,kind='Current777a common-u contracted-pair diagnostic; all256 outputaxes retained, target-freeTRAIN35/150')
def record(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)
recipe=dict(utc=now,kind=plan['kind'],teacher=plan['candidates']['teacher'],capture_ranks=[35,150],layers=list(range(10,15)),target_free=True,reference_submission=337422,hook=record(HERE/'old_su_hook_v1.py'),
    harness=[record(OLD/n)for n in('client_capture_v1.py','run_group_capture_v1.py','fragment_mark_complete_line_v1.py')]+[record(HERE/'run_capture_v1.py')],byte_limit=1500000000,guard_unchanged=True,no_native_handles_retained=True,price_sha256=sha(HERE/'PRICE_CURRENT32_v1.json'),
    predeclared='Retain75% complete contracted common-u pairs roundedup64 selectedatL10 by centeredSu energy timesFu frameenergy; reuseindices throughL14; preservefullu rowmean/all256 outputaxes. Assess physicalread error beforecandidatebuild; no sweep/no truth/noheld25.')
for name,data in(('PLAN_CAPTURE_v1.json',plan),('RECIPE_CAPTURE_v1.json',recipe)):
    with(HERE/name).open('x')as f:json.dump(data,f,indent=2);f.write('\n')
print(json.dumps(dict(plan_sha256=sha(HERE/'PLAN_CAPTURE_v1.json'),recipe_sha256=sha(HERE/'RECIPE_CAPTURE_v1.json'),hook_sha256=recipe['hook']['sha256'],net_bill_delta_pct=price['mean_bill_delta_pct'],conservative_bill_delta_pct=price['mean_conservative_bill_delta_pct'])))
