"""Unchanged small-map gates with the separately certified native CDF."""
from pathlib import Path
import json
import hashlib
import sys

here = Path(__file__).resolve().parent
device = sys.argv[sys.argv.index('--device')+1] if '--device' in sys.argv else 'cpu'
cdf = json.loads((here/f'gaussian_cdf_quadrature_{device}_v1.json').read_text())
assert cdf['passed']
assert cdf['source_sha256']==hashlib.sha256((here/'gaussian_cdf_quadrature_v1.py').read_bytes()).hexdigest()
source = (here/'certify_centered_torch_numerics_v1.py').read_text()
source = source.replace('from centered_torch_numerics_v1 import','from centered_torch_numerics_v2 import')
source = source.replace('centered_torch_numerics_{args.device}_small_v1.json','centered_torch_numerics_{args.device}_small_v2.json')
source = source.replace("HERE/'centered_torch_numerics_v1.py'","HERE/'centered_torch_numerics_v2.py'")
source = source.replace('erfc_probe','cdf_probe')
source = source.replace('# This is intentionally a direct backend call: unsupported erfc fails.',
                        '# Native quadrature only; no unsupported operation or CPU fallback.')
exec(compile(source,__file__+':certificate','exec'),globals())
