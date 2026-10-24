"""Tighten the old-block price for the bank's shared youngest-pair push."""
from pathlib import Path

HERE=Path(__file__).resolve().parent;path=HERE/'price_cp_only_one_high_v1.py';source=path.read_text()
edits=(('cp_only_one_high_cost_v1.json','cp_only_one_high_cost_v2.json'),
       ('removed=mm(n,prev,R)+8*n*R',
        'young=0 if l==0 else int(births[l-1])\n        removed=mm(n,prev,R-2*young)+mm(n,prev,young)+4*n*young+8*n*R'),
       ('Existing star/trace/cov/readout costs remain in measured baseline.',
        'V2 explicitly prices the existing shared-youngest-pair push; V1 overestimated the removed push. Existing star/trace/cov/readout costs remain in measured baseline.'))
for old,new in edits:
    assert source.count(old)==1;source=source.replace(old,new)
exec(compile(source,str(path),'exec'),dict(__file__=__file__,__name__='__main__'))
