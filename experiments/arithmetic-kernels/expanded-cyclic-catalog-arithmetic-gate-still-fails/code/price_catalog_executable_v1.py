"""Reprice only executable packed-count programs; omit uncompiled DPS."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE/'price_smirnov_executable_transpose_v1.py').read_text()
source = source.replace('smirnov_executable_transpose_envelope_v1.json', 'catalog_executable_envelope_v1.json')
source = source.replace('rectangular_catalog_cyclic_certificate_v1.json', 'catalog_executable_cse_certificate_v1.json')
source = source.replace("arithmetic=tuple(p['arithmetic'])", "arithmetic=tuple(p['packed_linear_operations'])")
needle = "exec(compile(source, str(HERE/'price_mixed_fmm_envelope_v1.py'), 'exec'),"
assert source.count(needle) == 1
injection = """
source = source.replace("programs.append(dict(name='DPS2026_accurate48',shape=(4,4,4),rank=48,arithmetic=(84,76,124)))", '')
source = source.replace("programs.append(dict(name='DPS2026_accurate48_transpose',shape=(4,4,4),rank=48,arithmetic=(76,84,124)))", '')
source = source.replace('Other catalog/DPS entries remain optimistic algebra envelopes and need executable circuits before a kernel.', 'All36 catalog entries now use certified executable DAGs and packed copies; uncompiled DPS omitted.')
source = source.replace('Smirnov/DPS literal copies, allocations and remote overhead excluded.', 'Explicit required packed direct/duplicate copies included; allocations and remote overhead excluded.')
"""
source = source.replace(needle, injection+'\n'+needle)
exec(compile(source, str(HERE/'price_smirnov_executable_transpose_v1.py'), 'exec'),
     dict(__file__=str(HERE/'price_catalog_executable_v1.py'), __name__='__main__'))
