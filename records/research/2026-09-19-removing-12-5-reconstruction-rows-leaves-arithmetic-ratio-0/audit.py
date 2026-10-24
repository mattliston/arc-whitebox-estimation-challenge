"""Stdlib-only source/shape arithmetic screen; no arrays or model calls."""
import ast
import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import time

HERE=Path(__file__).resolve().parent
D=HERE.parent
A=D.parent/'astra_phase2_reexamination_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2)
started=time.monotonic()
parent=D/'lazy_rotation_objective_v1/candidate.py'
assert sha(parent)=='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
tree=ast.parse(parent.read_text())
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
tile=None
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_PACKED_TILE' for t in node.targets):
        tile=ast.literal_eval(node.value)
assert tile==8192
namespace={'_COST_RULE_CACHE':{}}
exec(compile(ast.Module(body=[functions['_rule_depth']],type_ignores=[]),'<parent scalar rule>','exec'),namespace)

def price(m,k,n):
    # This records the rule's arithmetic formula, NOT full flopscope cost.
    if n>tile:
        parts=[price(m,k,min(tile,n-start)) for start in range(0,n,tile)]
        return dict(arithmetic=sum(p['arithmetic'] for p in parts),
                    depths=sum((p['depths'] for p in parts),[]))
    depth=namespace['_rule_depth']('push',m,k,n)
    if depth==0:
        bill=Fraction(2*m*k*n-m*n)
    else:
        packets=(4096,n-4096) if depth==6 and n>4096 else (n,)
        growth=Fraction(7,4)**depth
        bill=sum(2*m*k*nc*Fraction(7,8)**depth-m*nc*growth
                 +Fraction(7,3)*(m*k+k*nc+m*nc)*(growth-1) for nc in packets)
    assert bill.denominator==1
    return dict(arithmetic=int(bill),depths=[depth])

rows=[]
for width in (2048,6144,8128,16384):
    base=price(1024,384,width)
    trimmed=[]
    for m in (832,864,896,928,960,992):
        value=price(m,384,width)
        trimmed.append(dict(rows=m,row_fraction=m/1024,**value,
                            arithmetic_ratio=value['arithmetic']/base['arithmetic']))
    rows.append(dict(old_columns=width,full=base,trimmed=trimmed))

pins=[parent,D/'orthogonal_cubic_merge_information_v1/closeout.json',
      A/'COLD_CP_TRACE_ROUND_V1.md',A/'COLD_CP_TRACE_ROUND_V1_CLOSED.md',
      A/'CP_ONLY_ONE_HIGH_V1_CLOSED.md',A/'ONE_HIGH_THIRD_V1_CLOSED.md',
      A/'THIRD_COVARIANCE_EDGE_V1_CLOSED.md',A/'RESIDUAL_TRACE_REPRESENTATION_V1_CLOSED.md',
      A/'JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_V1.md',
      A/'JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_ADDENDUM_V1.md']
result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            parent_sha256=sha(parent),scalar_selector_ast_sha256=hashlib.sha256(ast.dump(functions['_rule_depth'],include_attributes=False).encode()).hexdigest(),
            packed_tile=tile,mode='exact integer evaluation of current selector arithmetic formula',
            illustrative_shape_grid=True,current_network_shape_census=False,
            complete_candidate_bill=False,complete_kernel_bill=False,
            measured_runtime=False,accuracy_or_score_measured=False,
            rows=rows,external_pins=[dict(path=str(p),sha256=sha(p)) for p in pins],
            arrays_decoded=0,model_calls=0,new_numerical_workers=0,
            new_samples=0,truth_files_accessed=0,new_training=False,
            new_candidate=False,new_archive=False,new_upload=False,
            elapsed_s=time.monotonic()-started,remaining_disk_GiB=shutil.disk_usage(HERE).free/2**30)
write(HERE/'result.json',result)
files=sorted(p for p in HERE.iterdir() if p.is_file())
closeout=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              status='source and scalar kernel preflight complete; no inference candidate selected',
              artifacts=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],
              external_pins=result['external_pins'],goal_status='active',goal_achieved=False)
write(HERE/'closeout.json',closeout)
print(json.dumps(dict(closeout_sha256=sha(HERE/'closeout.json'),
                     artifact_count=len(files),external_pin_count=len(pins),rows=rows)),flush=True)
