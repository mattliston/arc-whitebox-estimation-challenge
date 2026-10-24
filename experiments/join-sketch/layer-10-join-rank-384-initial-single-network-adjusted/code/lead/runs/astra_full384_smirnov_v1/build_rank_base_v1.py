"""Transport the exact eleven rank/admission literals onto submitted Smirnov.

Source-only held base. New Smirnov shape keys must be separately priced/bound
before this source is eligible for a root-authorized numerical pilot.
"""
from pathlib import Path
import ast,difflib,hashlib,io,json,tokenize
D=Path(__file__).resolve().parent;R=D.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=R/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'
rank=R/'astra_full384_current_feasibility_v1/candidate_full384_current_UNVALIDATED_v1.py'
current=R/'astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py'
assert sha(old)=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
assert sha(rank)=='6cf3973085575eb4917160549765f6dedd1c42578c7825aff98e495480dd0068'
assert sha(current)=='2b10843a76fc033325ceee9dd30703eff29a1fa535cdc094c5feef72023a0c3c'
def spans(s):
 lines=s.splitlines(keepends=True);counts={};out={}
 for n in ast.parse(s).body:
  if isinstance(n,ast.FunctionDef):
   i=counts.get(n.name,0);counts[n.name]=i+1;out[(n.name,i)]=(n.lineno,n.end_lineno,''.join(lines[n.lineno-1:n.end_lineno]))
 return out
a,b,c=old.read_text(),rank.read_text(),current.read_text();aa,bb,cc=map(spans,(a,b,c))
changes=[]
for key,(_,_,before)in aa.items():
 after=bb[key][2]
 if before!=after:
  assert key[0]in ('_uvp_state','_ipt_forward_b_ab')
  start,end,now=cc[key];assert now==before,(key,'current admission body drifted')
  changes.append((start,end,after,key))
assert len(changes)==4
lines=c.splitlines(keepends=True)
for start,end,replacement,key in sorted(changes,reverse=True):lines[start-1:end]=[replacement]
result=''.join(lines)
before='_R320_RULE = {7: 384, 10: 320, 13: 384}'
after='_R320_RULE = {7: 384, 10: 384, 13: 384}'
assert result.count(before)==1;result=result.replace(before,after,1)
tokens=lambda s:[(t.type,t.string)for t in tokenize.generate_tokens(io.StringIO(s).readline)]
ct,rt=tokens(c),tokens(result);assert len(ct)==len(rt)
diff=[(x,y)for x,y in zip(ct,rt)if x!=y]
assert len(diff)==11 and all(x==(tokenize.NUMBER,'320')and y==(tokenize.NUMBER,'384')for x,y in diff)
inverse=result.replace(after,before,1);invlines=inverse.splitlines(keepends=True)
for start,end,replacement,key in sorted(changes,reverse=True):invlines[start-1:end]=[cc[key][2]]
assert ''.join(invlines)==c
target=D/'candidate_full384_smirnov_RANK_BASE_HELD_v1.py';compile(result,str(target),'exec')
target.open('x').write(result)
(D/'rank_base_source_diff_v1.patch').open('x').write(''.join(difflib.unified_diff(c.splitlines(True),result.splitlines(True),fromfile=str(current),tofile=str(target))))
out=dict(status='HELD_SOURCE_ONLY_NEW_SMIRNOV_KEYS_PENDING',candidate=dict(path=str(target),sha256=sha(target)),parent=dict(path=str(current),sha256=sha(current)),changed_numeric_literals=11,exact_inverse_parent=True,changed_functions=[dict(name=k[0],ordinal=k[1],line=start)for start,end,replacement,k in changes],rank_rule=dict(L7=384,L10=384,L13=384),common_rank=256,unchanged='RPE feature22, UVP split views, covariance cache, all learned weights and numerical dials, submitted Smirnov helper and current dispatch table.',required_before_pilot='Cost-only fixed Smirnov plans for rank384 L10–12; actual fixed pilot8 grant and source-specific contracts.',models=0,captures=0)
(D/'RANK_BASE_SOURCE_READY_v1.json').open('x').write(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
