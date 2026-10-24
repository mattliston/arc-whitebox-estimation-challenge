"""Unchanged shadow equations, repaired report-contract tape directory."""
from pathlib import Path
here = Path(__file__).resolve().parent
source = (here/'current_transport_shadow_v1.py').read_text()
source = source.replace('rows_current_transport_shadow_v1','rows_current_transport_shadow_v2')
exec(compile(source,__file__,'exec'))
