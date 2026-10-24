"""Source-only fixed output-projection derivative; no model execution."""
from pathlib import Path
import ast, hashlib, json
D=Path(__file__).resolve().parent;G=D.parent/'astra_gnn_nonnegative_rank2_v1';O=D.parents[2]
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,s):
 p=D/name
 if p.suffix=='.py':compile(s,str(p),'exec')
 with p.open('x')as f:f.write(s)
 return p
def js(name,value):return put(name,json.dumps(value,indent=2)+'\n')
old=(G/'fragment_gnr2_UNVALIDATED_v1.py').read_text()
assert H(G/'fragment_gnr2_UNVALIDATED_v1.py')=='1d4e680737d40a780f47668e59931a3f675883e5f79da9fe6558125edcac7eb5'
node=next(x for x in ast.parse(old).body if isinstance(x,ast.FunctionDef)and x.name=='_gnr2_contract')
segment=ast.get_source_segment(old,node)
kernel=(D/'kernel_HELD_v1.py').read_text().strip().replace('def output_projection_contract(T, X):','def _gnr2_contract(T, X):')
assert H(D/'kernel_HELD_v1.py')=='8255effd5c213c96544a08d4365a6da332573acd395de94529b91d3155fb8eb8'
assert old.count(segment)==1
changed=old.replace(segment,kernel)
fragment=changed.replace('GNR2','GOP2').replace('gnr2','gop2')
put('fragment_gop2_UNVALIDATED_v1.py',fragment)
P=O/'scout_native/k4liab_s51k2tw/candidate.py'
assert H(P)=='6a145fadb3224501e683e95424bd9ada0ff571041850c7d482969b742d8f8f41'
prefix=P.read_text().rstrip()+'\n\n'
ON=put('matchedC_ON_UNVALIDATED_v1.py',prefix+fragment)
OFF=put('matchedC_OFF_UNVALIDATED_v1.py',prefix+fragment.replace('_GOP2_ON = True','_GOP2_ON = False',1))
# Reverse the one numerical function substitution and namespace renaming.
reverse=fragment.replace('GOP2','GNR2').replace('gop2','gnr2').replace(kernel,segment)
assert reverse==old
manifest=js('SOURCE_MANIFEST_v1.json',dict(status='HELD_FIXED_SOURCE_NO_MODELS',parent_path=str(P),parent_sha256=H(P),donor_fragment_sha256=H(G/'fragment_gnr2_UNVALIDATED_v1.py'),kernel_sha256=H(D/'kernel_HELD_v1.py'),only_numerical_delta='_gnr2_contract body replaced by frozen output_projection_contract; namespace only otherwise',reverse_fragment_byte_exact=True,OFF_boolean_only=True,rank=2,layers=list(range(1,16)),unchanged_odd_message_products_and_shared_cache=True,no_new_value_cache=True,files={str(p):H(p)for p in [ON,OFF,D/'fragment_gop2_UNVALIDATED_v1.py']}))
judge=put('native_gop2_probe_judge_v1.py',(G/'native_gnr2_probe_judge_v1.py').read_text().replace('GNR2','GOP2').replace('gnr2','gop2'))
runner_text=(G/'finite_gnr2_smoke_HELD_v1.py').read_text().replace('GNR2','GOP2').replace('gnr2','gop2').replace('gnn_nonnegative_rank2_native_smoke2','gnn_output_projection_rank2_native_smoke2').replace('8719377a85aceaaee89081021dcc839d842d1e8784250448e32040a17047be20',H(ON))
runner=put('finite_gop2_smoke_HELD_v1.py',runner_text)
reverse_runner=runner_text.replace('GOP2','GNR2').replace('gop2','gnr2').replace('gnn_output_projection_rank2_native_smoke2','gnn_nonnegative_rank2_native_smoke2').replace(H(ON),'8719377a85aceaaee89081021dcc839d842d1e8784250448e32040a17047be20')
assert reverse_runner==(G/'finite_gnr2_smoke_HELD_v1.py').read_text()
c=json.loads((G/'NATIVE_SMOKE_PROTOCOL_HELD_v1.json').read_text())
c.update(purpose='gnn_output_projection_rank2_native_smoke2',run_name='run_gop2_smoke_v1')
c['candidates']['ON']=dict(path=str(ON),sha256=H(ON),mode=1)
c['source_pins']={p:h for p,h in c['source_pins'].items()if not p.startswith(str(G)+'/')}
names=['PROPOSAL_HELD_v1.json','kernel_HELD_v1.py','check_kernel_cost_v1.py','KERNEL_COST_RESULT_v1.json','build_source_smoke_v1.py','fragment_gop2_UNVALIDATED_v1.py','matchedC_ON_UNVALIDATED_v1.py','matchedC_OFF_UNVALIDATED_v1.py','SOURCE_MANIFEST_v1.json','native_gop2_probe_judge_v1.py','finite_gop2_smoke_HELD_v1.py']
c['source_pins'].update({str(D/n):H(D/n)for n in names})
c['source_pins'].update({str(G/n):H(G/n)for n in ['fragment_gnr2_UNVALIDATED_v1.py','finite_gnr2_smoke_HELD_v1.py','native_gnr2_probe_judge_v1.py','SOURCE_TINY_RESULT_v1.json']})
c['source_pins'][str(P)]=H(P)
c['OFF_evidence']='Boolean-only OFF and exact frozen GNR2 parent delegation; only rank2 contract body changes, all other donor fragment bytes reverse exactly. Source-only full import check required before grant; no OFF model call.'
c['mechanism']='Original fixed source-PCA seed range followed by exact real least-squares output projection; same rank2/full-mean/maps/layers/cache/odd-message paths.'
c['implementation_gates']=[s.replace('GNR2','GOP2')for s in c['implementation_gates']]
protocol=js('NATIVE_SMOKE_PROTOCOL_HELD_v1.json',c)
grant=dict(approved=False,reviewed_by='root',purpose=c['purpose'],protocol_sha256=H(protocol),runner_sha256=H(runner),run_name=c['run_name'],numerical_slots=1,fixed_order=c['fixed_order'],resource_policy=c['resource_policy'],admission_wait_policy=c['admission_wait_policy'],accepted_TCH_control_reuse=c['native_reuse'],selection=c['selection'])
js('ROOT_NATIVE_SMOKE_GRANT_DRAFT_v1.json',grant)
print(json.dumps(dict(candidate_sha256=H(ON),runner_sha256=H(runner),judge_sha256=H(judge),protocol_sha256=H(protocol),only_numerical_delta='frozen contract kernel',reverse_donor_runner_exact=True)))
