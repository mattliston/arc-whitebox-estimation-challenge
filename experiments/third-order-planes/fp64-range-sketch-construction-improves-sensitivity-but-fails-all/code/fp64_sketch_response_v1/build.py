import ast
from datetime import datetime,timezone
from pathlib import Path
from common import HERE,D,OLD,DIAG,PARENT,NATIVE,read,write,pin,verify

assert read(DIAG/'completed.json')['completed']
helper='''

# BEGIN CURRENT-SKETCH FP64 PRODUCER V1
def _stable_shared_sketch_qr(J, weight, Omega, Qp, core):
    j64 = J.astype(fnp.float64)
    omega64 = Omega.astype(fnp.float64)
    weight64 = weight.astype(fnp.float64)
    Y64 = j64 @ (weight64[:, None] * (j64.T @ omega64))
    del j64, weight64
    if Qp is not None:
        q64 = Qp.astype(fnp.float64)
        core64 = core.astype(fnp.float64)
        Y64 = Y64 + q64 @ (core64 @ (q64.T @ omega64))
        del q64, core64
    del omega64
    Q64, unused_R64 = fnp.linalg.qr(Y64, mode='reduced')
    Q32 = Q64.astype(fnp.float32)
    del Q64, unused_R64, Y64
    return Q32
# END CURRENT-SKETCH FP64 PRODUCER V1
'''
before='''        Y = _shared_old384_mm(J, weight[:, None] * _shared_old384_mm(J.T, Omega))
        if old_columns:
            Y = Y + _shared_old384_mm(
                Qp, _shared_old384_mm(state['core'], _shared_old384_mm(Qp.T, Omega)))
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        del unused_R, Y, Omega'''
after='''        Qn = _stable_shared_sketch_qr(J, weight, Omega, Qp,
                                     state['core'] if old_columns else None)
        del Omega'''
(HERE/'variants').mkdir(exist_ok=False)
old_build=read(OLD/'build.json')
entries=[dict(coordinate=None,label='base',source=pin(PARENT))]
entries += [r for r in old_build['variants'] if r['coordinate'] in [0,8] and r['label'] in ['plus','minus','double']]
assert len(entries)==7
rows=[]
for e in entries:
    verify(e['source']);source=Path(e['source']['path']).read_text()
    node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push'][-1]
    lines=source.splitlines(keepends=True)
    start=sum(map(len,lines[:node.lineno-1]));end=sum(map(len,lines[:node.end_lineno]))
    block=source[start:end];assert block.count(before)==1
    changed=block.replace(before,after)
    candidate=source[:start]+changed+source[end:]+helper
    assert candidate[:-len(helper)][:start]+changed.replace(after,before)+candidate[:-len(helper)][start+len(changed):]==source
    ast.parse(candidate)
    name='candidate.py' if e['label']=='base' else f"variants/head{e['coordinate']:02d}_{e['label']}.py"
    dest=HERE/name
    with dest.open('x') as f:f.write(candidate)
    rows.append(dict(coordinate=e['coordinate'],label=e['label'],original=e['source'],candidate=pin(dest),inverse_exact=True))
controls=[]
old_inputs=read(OLD/'inputs.json')
old_response=read(OLD/'response_qualified.json')
for rank in [36,38]:
    base=next(x for x in old_inputs['qualification_parent'] if x['rank']==rank)
    controls.append(dict(rank=rank,coordinate=None,label='base',output=base['output'],flops=base['flops']))
    for c in [0,8]:
        for label in ['plus','minus','double']:
            row=next(x for x in old_response['rows'] if x['rank']==rank and x['coordinate']==c and x['label']==label and not x['repeat'])
            controls.append(dict(rank=rank,coordinate=c,label=label,output=row['output'],flops=row['flops']))
for x in controls:verify(x['output'])
write(HERE/'build.json',dict(variants=rows,controls=controls,parent=pin(PARENT),diagnostic=pin(DIAG/'closeout.json'),
     old_response=pin(OLD/'response_qualified.json'),targets_opened=False))
paths=[*HERE.glob('*.py'),*HERE.glob('variants/*.py'),HERE/'PROTOCOL.md',HERE/'build.json',
       D/'run_guarded.py',NATIVE,DIAG/'analysis.json',DIAG/'closeout.json',DIAG/'snapshot.py']
paths.extend(r['original']['path'] for r in rows)
paths.extend(r['output']['path'] for r in controls)
write(HERE/'sources_sealed.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
     pins=[pin(p) for p in sorted({str(Path(p).resolve()) for p in paths})],targets_opened=False))
print(dict(variants=len(rows),inverse_exact=True,controls=len(controls)),flush=True)
