"""Source-only exact-Gram sibling; the sketch preparation remains unexecuted."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SKETCH=HERE.parent/'teammate_global_pair_sketch_20260927_v1'

def edit(text,before,after,count=1):
    assert text.count(before)==count,(before,text.count(before),count)
    return text.replace(before,after)

def put(name,text):
    ast.parse(text)
    (HERE/name).write_text(text)

for name in ('common.py','pair_base.py','projection.py','gradient.py','evaluate.py','launch_fit.py'):
    put(name,(SKETCH/name).read_text())

fit=(SKETCH/'fit.py').read_text()
fit=edit(fit,"omega=torch.tensor(read(HERE/'omega.json')['values'],dtype=torch.float32)\nassert omega.shape==(1024,64)\n",'')
fit=edit(fit,"        Z=R.float()@omega\n        a['G']=Z@Z.T","        Rphysical=R.float()\n        a['G']=Rphysical@Rphysical.T")
fit=edit(fit,'del fs,rs,ks,gs,omega,Z,G64;','del fs,rs,ks,gs,Rphysical,G64;')
fit=edit(fit,"fresh_initialization=True,omega=pin(HERE/'omega.json'),","fresh_initialization=True,")
fit=edit(fit,'Gram of fixed width64 sketch of current standardized post covariance','exact shared-neighbor Gram of current standardized post covariance')
put('fit.py',fit)

fragment=(SKETCH/'fragment.py').read_text()
fragment=edit(fragment,'_CF_OMEGA = None\n','')
fragment=edit(fragment,'global _CF_PARAMS, _CF_OMEGA','global _CF_PARAMS')
fragment=edit(fragment,'        _CF_OMEGA = fnp.asarray(_CF_OMEGA_LITERAL,dtype=fnp.float32)\n','')
fragment=edit(fragment,"    Z=Rphysical@_CF_OMEGA[:int(F.shape[0]),:]\n    G=(Z@Z.T)/p['global_scale']","    G=(Rphysical@Rphysical.T)/p['global_scale']")
put('fragment.py',fragment)

unit=(SKETCH/'unit.py').read_text()
unit=edit(unit,"OMEGA=np.asarray(read(HERE/'omega.json')['values'],dtype='float32')\n",'')
unit=edit(unit,"    Z=Rphysical@OMEGA[:len(F),:]\n    G=(Z@Z.T)/P['global_scale']","    G=(Rphysical@Rphysical.T)/P['global_scale']")
put('unit.py',unit)

build=(SKETCH/'build.py').read_text()
build=edit(build,"omega_entry=frozen['omega'];assert pin(omega_entry['path'])==omega_entry\nomega=read(omega_entry['path'])['values']\nassert len(omega)==1024 and all(len(row)==64 and all(x in (-0.125,0.125) for x in row) for row in omega)\n",'')
build=edit(build,"+'\\n_CF_OMEGA_LITERAL = '+repr(omega)",'')
build=edit(build,",'omega.json'",'')
build=edit(build,'global_pair_sketch64','global_pair_exact')
put('build.py',build)

assess=(SKETCH/'assess.py').read_text()
assess=edit(assess,'global_pair_sketch64','global_pair_exact',count=2)
assess=assess[:assess.index("raise RuntimeError('Use team_score_screen_v1.py")]+"raise RuntimeError('Use team_score_screen_v1.py for enabled bank comparison')\n"
put('assess.py',assess)
launch=(SKETCH/'launch_native.py').read_text().replace('global_pair_sketch64','global_pair_exact')
put('launch_native.py',launch)

sources=[]
for p in sorted(SKETCH.iterdir()):
    if p.suffix in ('.py','.md','.json'):
        sources.append(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(HERE/'sketch_preparation_pins.json').write_text(json.dumps(dict(sources=sources,
    sketch_numerical_jobs=0,sketch_fits=0,sketch_endpoint_scores=0,
    selection='Exact RR^T chosen before fitting because a width64 sketch can dominate weak offdiagonal Gram signal.'),indent=2)+'\n')
print('EXACT_GRAM_SOURCE_ADAPTERS_PREPARED_NO_NUMERICAL_JOBS')
