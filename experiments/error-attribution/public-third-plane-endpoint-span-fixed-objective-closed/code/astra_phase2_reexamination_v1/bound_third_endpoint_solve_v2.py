"""Serialization-only repair: convert NumPy scalar flags before JSON output."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
original=HERE/'bound_third_endpoint_solve_v1.py'
source=original.read_text()
for before,after in (
    ("OUT=HERE/'third_endpoint_solve_bound_v1.json'","OUT=HERE/'third_endpoint_solve_bound_v2.json'"),
    ('    assert small_pass','    small_pass=bool(small_pass)\n    assert small_pass'),
    ('    record=dict(small_certificate_passed=small_pass',
     '    lower=float(lower)\n    record=dict(small_certificate_passed=small_pass')):
    assert source.count(before)==1,before
    source=source.replace(before,after)
exec(compile(source,str(original),'exec'),dict(__file__=str(Path(__file__).resolve()),__name__='__main__'))
