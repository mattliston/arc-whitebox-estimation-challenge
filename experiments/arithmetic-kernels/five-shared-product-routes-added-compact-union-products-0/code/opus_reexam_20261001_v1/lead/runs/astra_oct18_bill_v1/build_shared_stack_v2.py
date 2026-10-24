"""Preserve reconstruction and read-leaf reuse contexts in the shared router."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
parent=D/'candidate_v61j_union_shared_HELD_v1.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='fb3802cd4e7d946f6ca85856e5d135d946acc6baee125008e05ccc75ebdb809a'
source=parent.read_text()
old='            or _UBX_CTX[0] is not None or _ASTRA_CLUSTER_CTX[0] is not None\n'
new=old+'            or _UBX_RC[0] is not None or _ASTRA_RLF_CONTEXT[0] is not None\n'
assert source.count(old)==1
source=source.replace(old,new,1)
out=D/'candidate_v61j_union_shared_HELD_v2.py'
compile(source,str(out),'exec')
fragment=(D/'fragment_shared_stack_v1.py').read_text()
assert fragment.count(old)==1
fragment=fragment.replace(old,new,1)
receipt=json.loads((D/'SHARED_STACK_BUILD_v1.json').read_text())
receipt.update(candidate=str(out),sha256=hashlib.sha256(source.encode()).hexdigest(),
               revision_parent=str(parent),revision_parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
               guard_additions=['_UBX_RC[0] is not None','_ASTRA_RLF_CONTEXT[0] is not None'],
               forecast_warning='Rank35 standalone shape total is an upper bound: reconstruction reuse may exclude two routes.')
for path,body in ((out,source),(D/'fragment_shared_stack_v2.py',fragment),(D/'SHARED_STACK_BUILD_v2.json',json.dumps(receipt,indent=2)+'\n')):
 with path.open('x')as f:f.write(body)
print(json.dumps(dict(candidate=str(out),sha256=receipt['sha256'],guard_additions=receipt['guard_additions'])))
