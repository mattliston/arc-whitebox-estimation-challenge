"""One order edit, including exact whole-source inverses to all prior variants."""
import ast
from common import HERE,PARENT,PARENT_SHA,OLD,read,write,pin,verify,sha,weights_path

assert sha(PARENT)==PARENT_SHA
old_build=read(OLD/'build.json')
verify(old_build['parent'])
assert old_build['parent']==pin(PARENT)
sources=[dict(coordinate=None,label='base',source=pin(PARENT))]
sources += [r for r in old_build['variants'] if r['coordinate'] in [0,8]]
assert len(sources)==7
(HERE/'variants').mkdir(exist_ok=False)
rows=[]
for row in sources:
    verify(row['source'])
    source=open(row['source']['path']).read()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='select')
    old=ast.get_source_segment(source,node)
    marker="    indices = fnp.argsort(-score, kind='stable')[:count]\n"
    assert old.count(marker)==1
    new=old.replace(marker,marker+'    indices = fnp.sort(indices)\n',1)
    assert source.count(old)==1
    changed=source.replace(old,new,1)
    assert changed.replace(new,old,1)==source
    ast.parse(changed)
    path=HERE/'candidate.py' if row['label']=='base' else HERE/'variants'/f"head{row['coordinate']:02d}_{row['label']}.py"
    with path.open('x') as f: f.write(changed)
    rows.append(dict(coordinate=row['coordinate'],label=row['label'],original=row['source'],
        canonical=pin(path),whole_source_inverse_exact=True))
controls=[]
base_path=OLD/'baseline_replay_audit_v1/base36/result.json'
base=read(base_path)['records'][0]
controls.append(dict(label='base',receipt=pin(base_path),call=base,output=pin(base['output'])))
for label in ['plus','minus','double']:
    path=OLD/f'r0036_h00_{label}/result.json'
    call=read(path)['records'][0]
    controls.append(dict(label=label,receipt=pin(path),call=call,output=pin(call['output'])))
write(HERE/'build.json',dict(parent=pin(PARENT),variants=rows,original36_controls=controls,
    old_response=pin(OLD/'response_qualified.json'),old_closeout=pin(OLD/'closeout_after_replay_v1.json'),
    weights=[dict(rank=r,file=pin(weights_path(r))) for r in [36,38]],
    protocol=pin(HERE/'PROTOCOL.md'),builder=pin(__file__),targets_accessed=False,numeric_calls=0))
print(dict(variants=len(rows),inverse_all_exact=True,controls=len(controls)),flush=True)
