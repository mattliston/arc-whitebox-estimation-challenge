"""Original fixed fit, requiring eight complete coherent-8M rows."""
from pathlib import Path
import json

here = Path(__file__).resolve().parent
receipt = json.loads((here / 'covariance_innovation_probe_rows_v2.json').read_text())
assert receipt['n'] == 8 and len(receipt['records']) == 8
assert all(r['protocol'] == 'coherent_8m_v2' for r in receipt['records'])
source = (here / 'train_covariance_innovation_probe_v1.py').read_text()
assert source.count('_v1') == 4
source = source.replace('_v1', '_v2')
source = source.replace('stored MC covariance labels.', 'coherent stored 8M mean/covariance labels.')
exec(compile(source, __file__, 'exec'))
