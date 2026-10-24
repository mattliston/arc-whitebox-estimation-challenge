"""Unchanged finite FMM DP with all36 certified catalog orientations."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE/'price_rectangular_catalog_v1.py').read_text()
source=source.replace('rectangular_catalog_envelope_v1.json','rectangular_catalog_cyclic_envelope_v1.json')
start=source.index("extra='''")
end=source.index("'''",start+len("extra='''"))+3
extra="""
catalog=json.loads((HERE/'rectangular_catalog_cyclic_certificate_v1.json').read_text())
assert catalog['passed']
for rec in catalog['records']:
    programs.append(dict(name=rec['name'],shape=tuple(rec['shape']),
                         rank=rec['rank'],arithmetic=tuple(rec['arithmetic'])))
"""
source=source[:start]+'extra='+repr(extra)+source[end:]
exec(compile(source,str(HERE/'price_rectangular_catalog_v1.py'),'exec'),
     dict(__file__=str(Path(__file__)),__name__='__main__'))
