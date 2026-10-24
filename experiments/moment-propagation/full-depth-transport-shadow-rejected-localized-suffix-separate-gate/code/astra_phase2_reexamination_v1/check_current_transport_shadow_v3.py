"""Original fidelity gates; erf avoids unsupported MPS erfc operator."""
from pathlib import Path
here = Path(__file__).resolve().parent
source = (here/'check_current_transport_shadow_v1.py').read_text()
source = source.replace('from current_transport_shadow_v1','from current_transport_shadow_v3')
source = source.replace('rows_current_transport_shadow_v1','rows_current_transport_shadow_v2')
source = source.replace('current_transport_shadow_fidelity_v1','current_transport_shadow_fidelity_v3')
exec(compile(source,__file__,'exec'))
