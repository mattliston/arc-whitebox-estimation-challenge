"""Add only the six certified programs and their exact transpose duals."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'price_smirnov_linear_circuit_v1.py').read_text()
source=source.replace('smirnov_linear_circuit_envelope_v1.json','rectangular_catalog_envelope_v1.json')
needle="exec(compile(source, str(HERE / 'price_mixed_fmm_envelope_v1.py'), 'exec'),"
assert source.count(needle)==1
extra='''
catalog = json.loads((HERE/'rectangular_catalog_certificate_v1.json').read_text())
assert catalog['passed']
for rec in catalog['records']:
    m,k,n = rec['shape']
    a,b,c = rec['arithmetic']
    name = 'catalog_'+'x'.join(map(str,rec['shape']))
    programs.append(dict(name=name,shape=(m,k,n),rank=rec['rank'],arithmetic=(a,b,c)))
    programs.append(dict(name=name+'_transpose',shape=(n,k,m),rank=rec['rank'],arithmetic=(b,a,c)))
'''
injection="source = source.replace(\"programs.append(dict(name='bank_winograd'\", " + repr(extra) + " + \"programs.append(dict(name='bank_winograd'\")\n"
source=source.replace(needle,injection+needle)
exec(compile(source,str(HERE/'price_smirnov_linear_circuit_v1.py'),'exec'),
     dict(__file__=str(HERE/'price_rectangular_catalog_v1.py'),__name__='__main__'))
