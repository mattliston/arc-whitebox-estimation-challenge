"""Line-preserving selection-only patch on exact graded95c parent."""
import ast,hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
parent=D.parent/'astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
out=D/'candidate_95c_tensor_membership_l13_HELD_v1.py'
assert not out.exists()
raw=parent.read_bytes();assert hashlib.sha256(raw).hexdigest()=='95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf'
s=raw.decode()
old="            _jn_cols1 = fnp.argsort(-(_jn_c.astype(fnp.float64)), kind='stable')[:_JSK_KC[layer]]"
new="            _jn_cols1 = _o18_jst_columns(layer, J, weight, _jn_c, state, start) if layer == 13 else fnp.argsort(-(_jn_c.astype(fnp.float64)), kind='stable')[:_JSK_KC[layer]]"
assert s.count(old)==2,s.count(old)
patched=s.replace(old,new)
assert len(s.splitlines())==len(patched.splitlines())
changed_lines=[i+1 for i,(a,b) in enumerate(zip(s.splitlines(),patched.splitlines())) if a!=b]
assert len(changed_lines)==2
pa,ca=ast.parse(s),ast.parse(patched)
changed=[dict(line=x.lineno,name=getattr(x,'name',None))for x,y in zip(pa.body,ca.body)if ast.dump(x)!=ast.dump(y)]
assert [x['name']for x in changed]==['_shared_old384_push','_astra_bcc_push_impl'],changed
assert [(x.lineno,x.end_lineno)for x in pa.body]==[(x.lineno,x.end_lineno)for x in ca.body]
fragment=(D/'fragment_tensor_membership_l13_v1.py').read_text()
text=patched+'\n\n'+fragment
compile(text,str(out),'exec')
with out.open('x')as f:f.write(text)
meta=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),status='HELD_SOURCE_NEEDS_TINY_PEER_AND_NATIVE',
  parent=str(parent),parent_sha256=hashlib.sha256(raw).hexdigest(),candidate=str(out),candidate_sha256=hashlib.sha256(text.encode()).hexdigest(),
  helper_sha256=hashlib.sha256(fragment.encode()).hexdigest(),changed_functions=changed,changed_parent_line_numbers=changed_lines,
  all_parent_ast_locations_preserved=True,parent_E2_lifetime_code_unchanged=True,non13_selector_expression_identical=True,
  no_retained_float64_matrix=True,L13_kc=1536,rank384_unchanged=True,Y_weights_full_Fj_old_term_and_core_rule_unchanged=True,
  native_diagnostic='Root owns fixed cold ranks35/200; useful negative adjusted mean and no catastrophic raw change before any extension. No sweep or submission authorized by this source artifact.',
  expected_adjusted_gain='Unknown. Historical L13 CP proxy improved9.33/6.21%; K3 proxy worsened13.24/3.09%. These are not endpoint predictions.')
with(D/'TENSOR_MEMBERSHIP_BUILD_v1.json').open('x')as f:json.dump(meta,f,indent=2)
print(json.dumps(meta))
