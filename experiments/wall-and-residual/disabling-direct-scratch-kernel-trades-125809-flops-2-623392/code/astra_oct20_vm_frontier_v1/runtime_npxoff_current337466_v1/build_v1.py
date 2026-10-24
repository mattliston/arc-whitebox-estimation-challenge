"""Static exact NP partner-copy runtime reversal, current337466 only."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parent.parent;K=R/'fable_oct19_kc_v1';D=R/'astra_oct20_bill_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parents={'science':(K/'candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py','434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4'),'release':(K/'candidate_rel_astra_o21_edge8_local90_per_layer960_v1.py','660bc2d6daa5950c8614ca505775d305d3e19031a7a57eb0ac815b90ab5b8200')}
tail='\n# Fixed exact shared partner copy. No runtime trigger or changed NP weights.\nassert _O20_NPX_COPY_EXACT_ON is True\nassert _K3LB_PARENT_LEFT is _o20_npx_copy_exact_left_v1\nassert _O20_NPX_COPY_EXACT_PARENT is _astra_compute_left_v1\nassert _ASTRA_COMPUTE_ON is False\nassert _ASTRA_COMPUTE_PARENT_V1 is _npx_left\n_O20_NPX_COPY_EXACT_ON = False\n'
nogov=K/'agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py'
out={}
for role,(parent,pin) in parents.items():
 assert sha(parent)==pin
 text=parent.read_text();target=HERE/f'candidate_npxoff_{role}_v1.py';assert not target.exists()
 candidate=text+tail+'\n'+nogov.read_text();ast.parse(candidate);target.write_text(candidate)
 assert candidate.startswith(text) and ast.unparse(ast.parse(candidate).body[-1])=='_FAST_RESCUE_FRAC = {}'
 out[role]=dict(path=str(target),sha256=sha(target),parent_path=str(parent),parent_sha256=pin)
source=parents['science'][0];historic=K/'candidate_rel_v4a_sci_v1.py'
def funcs(p,name):return [ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name]
checks={}
for name in ['_npx_left','_astra_compute_left_v1','_np_ages','_np_tables','_np_const']:
 checks[name]=funcs(source,name)==funcs(historic,name);assert checks[name],name
checks['_o20_npx_copy_exact_left_v1']=funcs(source,'_o20_npx_copy_exact_left_v1')==funcs(D/'tail_npx_copy_exact_v1.py','_o20_npx_copy_exact_left_v1');assert checks['_o20_npx_copy_exact_left_v1']
old=json.loads((D/'NPX_COPY_EXACT_CHECK_v1.json').read_text());assert old['numerical_and_auxiliary_exact'] and all(x['runtime_off_identical'] for x in old['records'])
r=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),outputs=out,tail=tail,affected_ast_equal=checks,historical_tiny=dict(path=str(D/'NPX_COPY_EXACT_CHECK_v1.json'),sha256=sha(D/'NPX_COPY_EXACT_CHECK_v1.json'),cases=len(old['records'])),no_submission=True,static_choice_only=True,projected_historical_rank35=dict(flops_added=132558848,requests=-316,approx_bill_pct=.056),limitation='Current whole-network tariff/runtime and prediction parity remain unmeasured.')
with (HERE/'BUILD_v1.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(outputs=out,affected_ast_equal=checks)))
