"""Use the standard process-order-matched evaluator, adding one scalar call count."""
from pathlib import Path
import hashlib,json
STUDY=Path(__file__).resolve().parent
H2=STUDY.parent
for e in json.loads((STUDY/'build.json').read_text())['sources']:
    p=Path(e['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
u=json.loads((STUDY/'unit.json').read_text());g=json.loads((STUDY/'unit_guard.json').read_text())
assert u['passed'] and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
source=(H2/'native_eval.py').read_text()
a='HERE=Path(__file__).resolve().parent';b='HERE=H2'
c="uvp=dict(getattr(mod,'_UVP_STATS',{})),ks="
d="cf=dict(getattr(mod,'_CF_STATS',{})),uvp=dict(getattr(mod,'_UVP_STATS',{})),ks="
assert source.count(a)==1 and source.count(c)==1
source=source.replace(a,b).replace(c,d)
assert source.replace(b,a).replace(d,c)==(H2/'native_eval.py').read_text()
exec(compile(source,str(H2/'native_eval.py')+':covfeedback','exec'),globals())
