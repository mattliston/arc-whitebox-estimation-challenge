from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
parent=HERE/'candidate_old_both_tail_trace_bounded_v6.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent)=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
source=parent.read_text()
old="                lam = fnp.concatenate([lam * _S19_LAM[0], _s19_new, -_s19_new])"
assert source.count(old)==1
records=[]
for name,gain in [('plus',1.001),('minus',.999)]:
    new=f'''                _ocal_count = int(shared_old['columns'])
                if _ocal_count:
                    _ocal_carry = fnp.concatenate([lam[:_ocal_count] * {gain!r}, lam[_ocal_count:]])
                else:
                    _ocal_carry = lam
                lam = fnp.concatenate([_ocal_carry * _S19_LAM[0], _s19_new, -_s19_new])'''
    body=source.replace(old,new)
    assert body.replace(new,old)==source
    ast.parse(body)
    out=HERE/f'candidate_old_cohort_carry_{name}_v1.py'
    with out.open('x') as f:f.write(body)
    records.append(dict(arm=name,old_prefix_multiplier=gain,path=str(out),sha256=sha(out)))
record=dict(parent_sha256=sha(parent),arms=records,whole_source_inverse_exact=True,
            original_uniform_carry=1.0032287195131746,validation_accessed=False)
with (HERE/'old_cohort_carry_information_manifest_v1.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
