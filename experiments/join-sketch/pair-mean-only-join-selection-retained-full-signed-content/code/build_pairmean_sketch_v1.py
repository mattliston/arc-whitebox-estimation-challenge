import ast,hashlib,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent
parent,out=map(Path,sys.argv[1:]);assert not out.exists()
s=parent.read_text()
old="""            _jn_cols1 = fnp.argsort(-(_jn_c.astype(fnp.float64)), kind='stable')[:_JSK_KC[layer]]
            J1 = J[:, _jn_cols1]
            Y = _shared_old384_mm(J1, weight[_jn_cols1][:, None] * J1[_jn_rows, :].T)
            del J1, _jn_cols1, _jn_c"""
new="""            J1, _jsm_weight = _astra_jsm_columns(layer, J, weight, _jn_c, state, start)
            Y = _shared_old384_mm(J1, _jsm_weight[:, None] * J1[_jn_rows, :].T)
            del J1, _jsm_weight, _jn_c"""
assert s.count(old)==2,s.count(old)
patched=s.replace(old,new)
pa,ca=ast.parse(s),ast.parse(patched)
changed=[(x.lineno,x.name)for x,y in zip(pa.body,ca.body)if ast.dump(x)!=ast.dump(y)]
assert [x[1]for x in changed]==['_shared_old384_push','_astra_bcc_push_impl']
fragment=(D/'fragment_pairmean_sketch_v1.py').read_text();text=patched+'\n\n'+fragment
compile(text,str(out),'exec');out.write_text(text)
meta=dict(parent=str(parent.resolve()),parent_sha256=hashlib.sha256(s.encode()).hexdigest(),candidate=str(out.resolve()),sha256=hashlib.sha256(text.encode()).hexdigest(),changed_functions=changed,helper_sha256=hashlib.sha256(fragment.encode()).hexdigest(),rank_and_full_Fj_unchanged=True,row_selection_expression_unchanged=True,pair_mean_columns={7:1024,10:768,13:768},status='SOURCE_READY_PENDING_TINY_AND_NATIVE')
with out.with_suffix('.source.json').open('x')as f:json.dump(meta,f,indent=2);f.write('\n')
print(json.dumps(meta))
