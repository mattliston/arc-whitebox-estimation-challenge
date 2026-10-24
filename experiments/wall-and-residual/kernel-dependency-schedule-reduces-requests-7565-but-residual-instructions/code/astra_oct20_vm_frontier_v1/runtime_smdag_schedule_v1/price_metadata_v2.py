"""Target-free fixed-beam Smirnov DAG scheduling price, no official/native execution."""
import ast, collections, hashlib, heapq, itertools, json, subprocess, time
from pathlib import Path

HERE=Path(__file__).resolve().parent;RUNS=HERE.parent.parent
K=RUNS/'fable_oct19_kc_v1';D=RUNS/'astra_oct20_bill_v1'
source=K/'candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4'
tree=ast.parse(source.read_text());env={}
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ASTRA_SM40_RECORDS' for t in n.targets):
  env['_SMX_RECORDS']=dict(ast.literal_eval(n.value))
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='_SMX_RECORDS' for t in n.targets):
  exec(compile(ast.Module(body=[n],type_ignores=[]),'<literal-record>','exec'),env)
 elif isinstance(n,ast.FunctionDef) and n.name=='_o20_smdirect_runs':
  exec(compile(ast.Module(body=[n],type_ignores=[]),'<fixed-greedy>','exec'),env)
oldruns=env['_o20_smdirect_runs']
fixed={(7,8,10,12,13):((7,10,13),(8,12)),(2,18,31,34,38):((2,18,34),(31,38))}
def runs(pos):
 pos=tuple(sorted(pos));return fixed.get(pos,tuple(oldruns(pos)))

def solve(plan):
 input_refs,groups,outputs,uses=plan
 nodes=[(ref,args,(tag[1],tag[2])) for tag,items in groups for ref,args in items]
 at={ref:i for i,(ref,args,key) in enumerate(nodes)}
 assert len(at)==len(nodes)
 output_at={ref:i for i,ref in enumerate(outputs)}
 bykey=collections.defaultdict(list)
 deps=[]
 for i,(ref,args,key) in enumerate(nodes):
  assert all(a in at or a in input_refs for a in args)
  deps.append(sum(1<<at[a] for a in set(args) if a in at))
  bykey[key].append(i)
 def cost(indices):
  return int(any(nodes[i][0] not in output_at for i in indices))+len(runs(output_at[nodes[i][0]] for i in indices if nodes[i][0] in output_at))
 oldcost=sum(cost([at[ref] for ref,args in items]) for tag,items in groups)
 allmask=(1<<len(nodes))-1
 # Include the original legal schedule as a guaranteed upper bound.
 upper=oldcost;winner=[(tag[1],tag[2],tuple(at[r] for r,a in items)) for tag,items in groups]
 started=time.monotonic();seen=0;exact=False
 beam={0:(0,())};visited={0:0}
 for depth in range(len(nodes)+1):
  nxt={}
  for mask,(spent,hist) in beam.items():
   seen+=1
   if mask==allmask:
    if spent<upper:upper=spent;winner=list(hist)
    continue
   for key,ii in bykey.items():
    ready=tuple(i for i in ii if not mask>>i&1 and deps[i]&mask==deps[i])
    if not ready:continue
    choices={ready,tuple(i for i in ready if nodes[i][0] not in output_at),tuple(i for i in ready if nodes[i][0] in output_at)}
    choices.discard(())
    for choice in sorted(choices):
     newmask=mask|sum(1<<i for i in choice);newcost=spent+cost(choice)
     if newcost>=upper or newcost>=visited.get(newmask,10**9):continue
     visited[newmask]=newcost;nxt[newmask]=(newcost,hist+((key[0],key[1],choice),))
  if not nxt:break
  def priority(item):
   mask,(spent,hist)=item
   remaining=sum(cost([at[r] for r,a in items if not mask>>at[r]&1]) for tag,items in groups)
   return (spent+.7*remaining,spent,-mask.bit_count(),mask)
  beam=dict(sorted(nxt.items(),key=priority)[:512])
  if time.monotonic()-started>15:break
 q=beam
 # Replay every dependency and every ordered operand; no algebraic identity.
 have=set(input_refs);newgroups=[]
 for gi,(op,lit,indices) in enumerate(winner):
  items=[(nodes[i][0],nodes[i][1]) for i in indices]
  assert all(a in have for r,args in items for a in args)
  have.update(r for r,args in items)
  newgroups.append(((gi,op,lit),items))
 assert set(output_at)<=have and sum(len(x[1]) for x in newgroups)==len(nodes)
 assert {r:(args,(tag[1],tag[2])) for tag,items in newgroups for r,args in items}=={r:(args,key) for r,args,key in nodes}
 return dict(original_compact_calls=oldcost,proposed_calls=upper,saved_calls=oldcost-upper,
             nodes=len(nodes),original_groups=len(groups),proposed_groups=len(newgroups),
             search_states=seen,search_seconds=time.monotonic()-started,
             search_exhausted=not q,found_best_in_restricted_ready_set_search=exact,
             original_scalar_ops_and_ordered_operands_identical=True,
             private_owner_planes_unchanged=len(nodes),groups=newgroups)

cache={};rows=[]
for record in env['_SMX_RECORDS'].values():
 for kind in ['L','R']:
  plan=record['prepared'][kind];key=hashlib.sha256(repr(plan).encode()).hexdigest()
  if key not in cache:cache[key]=solve(plan)
  rows.append(dict(record=record['name'],kind=kind,plan_sha256=key,**cache[key]))
prior=json.loads((D/'SMDIRECT_COMPACT_AP_PREPARED_v1.json').read_text())['rank35_census_projection']
projection=[]
for n,(name,kind) in zip(prior['signature_counts'],prior['signature_names']):
 row=next(r for r in rows if r['record']==name and r['kind']==kind)
 projection.append(dict(record=name,kind=kind,count=n,old_compact_calls=n*row['original_compact_calls'],new_calls=n*row['proposed_calls']))
out=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 mechanism='Rebatch exact independent nodes across historical depth groups; no changed scalar arithmetic or operand order.',
 plans=rows,historical_rank35_projection=projection,
 projected_saved_calls_beyond_compact=sum(x['old_compact_calls']-x['new_calls'] for x in projection),
 projected_flop_delta=0,official_test=False,native_test=False,
 limitations=['Price assumes static plan groups are replaced consistently before setup on every shared record/kernel alias.','No claim of fullsource binding, exact native output, current call mix or residual improvement.','Fixed beam512 prioritizes spent+0.7 times original remaining-group cost. All-ready/internal-only/output-only actions only; no global optimality claim.'])
with (HERE/'PRICE_METADATA_v2.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(dict(plans=[{k:r[k] for k in ['record','kind','original_compact_calls','proposed_calls','saved_calls','search_states','search_seconds']} for r in rows],additional_saved_calls=out['projected_saved_calls_beyond_compact'])))
