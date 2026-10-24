"""Fixed full384 Smirnov dispatch transport; no score-selected plans or models."""
from pathlib import Path
import ast,hashlib,json,difflib
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=D/'candidate_full384_smirnov_RANK_BASE_HELD_v1.py'
assert sha(base)=='eafb9d1dc57864269b7b1bba273ed3bebc40407f02792956b757e8d4f66267f9'
s=base.read_text();lines=s.splitlines(keepends=True);nodes={}
for n in ast.parse(s).body:
 if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('_ASTRA_SM40_PLANS','_ASTRA_SM40_FAMILIES','_ASTRA_SM40_PREBIND'):nodes[n.targets[0].id]=n
assert len(nodes)==3
old={k:ast.literal_eval(n.value)for k,n in nodes.items()};new=dict(old)
plans={};moved=[]
for key,plan in old['_ASTRA_SM40_PLANS'].items():
 target=key[:3]+(384,)if key[3]==320 else key
 if target in plans:assert plans[target]==plan,('conflicting inherited plans',target)
 plans[target]=plan
 if target!=key:moved.append(dict(before=key,after=target,plan=plan))
assert moved
new['_ASTRA_SM40_PLANS']=plans
families=[]
for family in old['_ASTRA_SM40_FAMILIES']:
 if family[0]=='L4_all':
  assert family==('L4_all',4,40,13,80,16);family=('L4_all',4,40,13,81,16)
 families.append(family)
new['_ASTRA_SM40_FAMILIES']=tuple(families)
prebind=[]
for family,args in old['_ASTRA_SM40_PREBIND']:
 if family[0]=='L4_all':family=('L4_all',4,40,13,81,16)
 prebind.append((family,args))
new['_ASTRA_SM40_PREBIND']=tuple(prebind)
edits=[(n.lineno,n.end_lineno,k+' = '+repr(new[k])+'\n')for k,n in nodes.items()]
for start,end,replacement in sorted(edits,reverse=True):lines[start-1:end]=[replacement]
result=''.join(lines);target=D/'candidate_full384_smirnov_HELD_v2.py';compile(result,str(target),'exec')
target.open('x').write(result)
(D/'fixed_dispatch_source_diff_v2.patch').open('x').write(''.join(difflib.unified_diff(s.splitlines(True),result.splitlines(True),fromfile=str(base),tofile=str(target))))
out=dict(status='HELD_FIXED_SOURCE_PENDING_TINY_AND_ROOT_GRANT',candidate=dict(path=str(target),sha256=sha(target)),rank_base=dict(path=str(base),sha256=sha(base)),submitted_parent_sha256='2b10843a76fc033325ceee9dd30703eff29a1fa535cdc094c5feef72023a0c3c',rank_admission_literals=11,common_rank=256,rank_rule={7:384,10:384,13:384},plans_before=len(old['_ASTRA_SM40_PLANS']),plans_after=len(plans),moved_plans=moved,plan_values_unchanged=True,family_before=('L4_all',4,40,13,80,16),family_after=('L4_all',4,40,13,81,16),prebind_shapes_unchanged=True,prebind_count=len(prebind),models=0,captures=0,authorization='Root directed one fixed full384 candidate; separate model grant required.')
p=D/'FIXED_SOURCE_READY_v2.json';p.open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='moved_plans'},indent=2));print('moved',len(moved),'manifest',sha(p))
