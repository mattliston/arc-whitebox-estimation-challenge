"""Versioned repair of a primary-package helper import; v1 is preserved."""
from pathlib import Path

source = Path(__file__).with_name('certify_augmented_fourth_trace_v1.py').read_text()
old = 'from mlp_kprop.tensor_utils import symmetrize, zero_repeated'
assert source.count(old) == 1
source = source.replace(old, 'from mlp_kprop.tensor_utils import symmetrize\nfrom mlp_kprop.diagslice import zero_repeated')
source = source.replace('augmented_fourth_trace_certificate_v1.json', 'augmented_fourth_trace_certificate_v2.json')
exec(compile(source, str(Path(__file__)), 'exec'), globals())
