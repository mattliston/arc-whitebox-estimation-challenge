"""Reuse the fixed two-worker input-only orchestration, new row namespace."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE/'generate_final_variance_source_rows_v1.py').read_text()
assert 'variance_source' in source
source = source.replace('variance_source', 'cohort_source')
exec(compile(source, str(HERE/'generate_final_variance_source_rows_v1.py'), 'exec'),
     dict(__file__=str(HERE/'generate_final_cohort_source_rows_v1.py'), __name__='__main__'))
