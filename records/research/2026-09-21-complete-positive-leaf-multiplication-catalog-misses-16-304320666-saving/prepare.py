import ast
from common import *
OLD=D/'complete_catalog_depth_v1'
assert sha(PARENT)==PARENT_SHA
assert sha(D/'continuation_checkpoint_v79.json')=='2b8380ce3c52fc9bae8f9452752cc8700a22a23b0777953ca986236052fa4b5e'
previous=read(D/'continuation_checkpoint_v79.json')
check(previous['study']);check(previous['official_bank'])
for p in read(OLD/'closeout.json')['artifacts']+read(OLD/'closeout.json')['external_pins']:check(p)
edits={
 'reprice.py':[
    ('DEPTH = 7','DEPTH = 14',1),('MIN_LEAF = 8','MIN_LEAF = 1',1),
    ('    return tuple(result)','    kept = [item for item in result if item[1] <= ARENA]\n    if result[-1] not in kept:\n        kept.append(result[-1])\n    return tuple(kept)',1),
    ('def frontier(m, k, n, stages):','def frontier(m, k, n):',1),
    ('        if stages:','        if max(m,k,n) > 1:',1),
    ('frontier(a,b,c,stages-1)','frontier(a,b,c)',2),
    ('frontier(m,k,n,DEPTH)','frontier(m,k,n)',1)],
 'verify_paths.py':[
    ('assert min(a,b,c)>=8','assert min(a,b,c)>=1 and max(a,b,c)<max(m,k,n)',1),
    ("assert path.count(' / ')<=7","assert path.count(' / ')<=(max(x['shape'])-1).bit_length()",1)]}
inverses=[]
for name,changes in edits.items():
    old=(OLD/name).read_text();new=old
    for a,b,count in changes:
        assert new.count(a)==count,(name,a);new=new.replace(a,b)
    restored=new
    for a,b,count in reversed(changes):
        assert restored.count(b)==count,(name,b);restored=restored.replace(b,a)
    assert restored==old;ast.parse(new)
    with (HERE/name).open('x') as f:f.write(new)
    inverses.append(dict(old=pin(OLD/name),new=pin(HERE/name),edits=changes,whole_source_inverse=True))
with (HERE/'reprice_inputs.json').open('xb') as f:f.write((OLD/'reprice_inputs.json').read_bytes())
inputs=read(HERE/'reprice_inputs.json')
external=[OLD/'closeout.json',OLD/'reprice.py',OLD/'verify_paths.py',OLD/'reprice_result.json',OLD/'PROTOCOL.md',
    OLD/'RESULT.md',OLD/'reprice_inputs.json',D/'continuation_checkpoint_v79.json',D/'official_best_331684.json',
    D/'run_guarded.py',PARENT,D/'read6_packet_v1/NEGATIVE_RESULT.md',
    A/'SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md',A/'SMIRNOV_GLOBAL_BATCH_REPRESENTATIVE_V1_CLOSED.md']
external += [Path(r['path']) for r in inputs['censuses']]
external += [Path(inputs[k]['path']) for k in ['smirnov_certificate','catalog_certificate']]
for p in read(OLD/'reprice_inputs_sealed.json')['pins']:check(p)
write(HERE/'build.json',dict(source_inverses=inverses,previous_checkpoint=pin(D/'continuation_checkpoint_v79.json'),
    prior_study=pin(OLD/'closeout.json'),minimum_leaf=[8,1],maximum_structural_depth=[7,14],
    shape_only_cache=True,arena_frontier_pruning=True,arrays_not_decoded=True))
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in dict.fromkeys(external)]
write(HERE/'reprice_inputs_sealed.json',dict(pins=pins,before_new_calculation=True,targets_not_decoded=True))
write(HERE/'sources_sealed.json',dict(pins=pins+[pin(HERE/'reprice_inputs_sealed.json')]))
print(dict(sealed=True,whole_source_inverses=2,leaf_minimum=1,structural_depth=14),flush=True)
