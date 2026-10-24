"""No arrays/SDK runtime/model: verify observer against exact span generator AST."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
source=HERE.parent/'sync_exact4685_v1/sdk_snapshot_v1/_dispatch.py'
observer=HERE.parents[1]/'astra_oct20_bill_v1/profile_outside_dispatch_v1.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('outside_profile_metadata',observer)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sdk=dict(__name__='flopscope._dispatch',contextmanager=contextlib.contextmanager,
         _now_ns=time.perf_counter_ns,_caller_is_internal=lambda **_:True,
         _count_participant_span=lambda:None,_total_dispatch_ns=0)
node=next(n for n in ast.parse(source.read_text()).body
          if isinstance(n,ast.FunctionDef)and n.name=='_counted_span')
exec(compile(ast.Module(body=[node],type_ignores=[]),'isolated_official_span_AST','exec'),sdk)
filename='candidate_metadata_fixture_only';env={}
def sdk_step():
    with sdk['_counted_span']():
        env['inside_sdk']()
        with sdk['_counted_span']():
            env['inside_sdk']()
code='''def child(x):
    return abs(x)+1
def inside_sdk():
    return sum(range(20))
def parent():
    values=[]
    for i in range(8):
        values.append(child(i))
    sdk_step()
    return sum(values)
'''
env['sdk_step']=sdk_step
exec(compile(code,filename,'exec'),env)
profile=module.OutsideDispatchProfile(filename,lambda:sdk['_total_dispatch_ns'])
profile.start();answer=env['parent']();profile.stop()
assert answer==36 and not profile.stack and not profile.active_sdk_spans
assert sys.getprofile()is None
assert sum(r['outside_exclusive_ns']for r in profile.stats.values())==profile.root_outside_ns
assert not any(k[2]=='inside_sdk'for k in profile.stats)
assert profile.excluded_candidate_events_in_sdk>=8
assert profile.root_dispatch_ns==sdk['_total_dispatch_ns']>0
report=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
            python=sys.version,successful=True,metadata_only=True,model_count=0,
            sdk_runtime_imported=False,official_span_ast_sha256=sha(source),observer_sha256=sha(observer),
            answer=answer,rows=len(profile.stats),nested_sdk_candidate_events_excluded=profile.excluded_candidate_events_in_sdk,
            all_dispatch_subtracted_once=True,exclusive_sum_equals_root=True,profiler_restored=True)
output=HERE/('METADATA_CHECK_py%d%d_v1.json'%sys.version_info[:2])
with output.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report))
