"""Qualify exact frozen edge8 + local90 source compositions; no native dispatch."""
import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;ROOT=Path('/home/arcworker/arc-workspace')
LOCAL='/Users/matt/arc-white-box-estimation';V=D.parent/'astra_oct20_vm_frontier_v1'
resolve=lambda p:Path(str(p).replace(LOCAL,str(ROOT)))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
prepared=read(HERE/'edge8_stacks_v1/PREPARED_v1.json');base=V/'PLAN_BANK8_ACTUAL_SHARED_v1.json'
assert sha(base)=='cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'
rows=[]
for arm in prepared['candidates']:
 folder=HERE/'edge8_stacks_v1'/arm['model'];bp=folder/'BUILD_v1.json'
 assert sha(bp)==arm['build_sha256']
 b=read(bp)
 subprocess.run([sys.executable,'-B',str(HERE/'check_edge_stack_v1.py'),'--folder',str(folder)],check=True)
 imp=folder/'NUMPY_FREE_IMPORT_v1.json';assert read(imp)['successful']
 src=resolve(b['outputs']['science']['path']);assert sha(src)==b['outputs']['science']['sha256']
 plan=read(base);name='edge8_local90_'+arm['model']
 plan.update(utc=utc(),kind='FIXED_EDGE8_LOCAL90_STACK',reference_submission=337422,head_fit_sha256=b['head_fit_sha256'],build_sha256=sha(bp),no_additive_gain_claim=True)
 plan['candidates']={'bank337393':plan['candidates']['bank337393'],name:dict(path=str(src.relative_to(ROOT)),bytes=src.stat().st_size,sha256=sha(src),effective_sha256=sha(src),role='frozen_candidate')}
 pp=folder/'PLAN_v1.json';write(pp,plan)
 row=dict(utc=utc(),successful=True,candidate=name,source_sha256=sha(src),outputs={k:{**v,'path':str(resolve(v['path']))} for k,v in b['outputs'].items()},plan=str(pp),plan_sha256=sha(pp),import_sha256=sha(imp),build_sha256=sha(bp),requested_jobs=12,shared_max_active=36,blas_threads=4,no_native_launch=True,requires_standalone_full32_pass=True,reference_submission=337422)
 write(folder/'LINUX_READY_v1.json',row);rows.append(row)
write(HERE/'edge8_stacks_v1/QUALIFIED_v1.json',dict(utc=utc(),successful=True,qualified=rows,no_native_launch=True,no_gain_claim=True))
print(json.dumps(dict(qualified=[r['candidate'] for r in rows],no_native_launch=True)))

