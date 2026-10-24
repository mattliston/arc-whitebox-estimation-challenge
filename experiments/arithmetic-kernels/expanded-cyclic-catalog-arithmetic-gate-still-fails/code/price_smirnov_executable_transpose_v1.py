"""Same finite census, executable Smirnov costs and labelled catalog bound."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE/'price_mixed_fmm_envelope_v1.py').read_text()
source = source.replace('mixed_fmm_envelope_v1.json', 'smirnov_executable_transpose_envelope_v1.json')
before = "programs=[dict(name=p['name'],shape=p['shape'],rank=40,arithmetic=p['scalar_linear_arithmetic']) for p in certificate['programs']]"
after = """actual=json.loads((HERE/'smirnov_executable_transpose_certificate_v1.json').read_text())
assert actual['passed']
programs=[dict(name=p['name'],shape=tuple(p['shape']),rank=40,
               arithmetic=tuple(p['packed_linear_operations'])) for p in actual['records']]
catalog=json.loads((HERE/'rectangular_catalog_cyclic_certificate_v1.json').read_text())
assert catalog['passed']
for p in catalog['records']:
    programs.append(dict(name=p['name'],shape=tuple(p['shape']),rank=p['rank'],
                         arithmetic=tuple(p['arithmetic'])))
"""
assert source.count(before) == 1
source = source.replace(before, after)
peeler = (HERE/'price_mixed_fmm_peel_v2.py').read_text()
fringe = peeler.split("source=source.replace(needle,needle+'''", 1)[1].split("''')", 1)[0]
needle = "                options.append((p['rank']*cost+linear,p['rank']*memory,p['name']+' / '+child))"
assert source.count(needle) == 1
source = source.replace(needle, needle+fringe)
source = source.replace('result=dict(records=records,programs=programs,',
                        'result=dict(stage1_twenty_percent_gate=all(r["optimistic_fraction"]>=.20 for r in records),records=records,programs=programs,')
source = source.replace("qualification='Bounded optimistic arithmetic envelope:",
                        "qualification='Smirnov executable regauged DAG costs including direct/duplicate packed copies, not transposition arithmetic formulas. Other catalog/DPS entries remain optimistic algebra envelopes and need executable circuits before a kernel. Exact classical fringes included. Bounded optimistic arithmetic envelope:")
source = source.replace("{**result,'records':", "{**{k:v for k,v in result.items() if k!='programs'},'records':")
exec(compile(source, str(HERE/'price_mixed_fmm_envelope_v1.py'), 'exec'),
     dict(__file__=str(HERE/'price_smirnov_executable_transpose_v1.py'), __name__='__main__'))
