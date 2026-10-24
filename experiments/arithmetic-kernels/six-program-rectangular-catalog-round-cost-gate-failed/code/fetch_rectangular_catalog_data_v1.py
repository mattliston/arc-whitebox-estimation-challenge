"""Download six whitelisted public algebra datasets; never execute content."""
import bz2
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
DEST = HERE/'public_rectangular_catalog_data_v1'
DEST.mkdir(exist_ok=True)
OUT = HERE/'rectangular_catalog_download_v1.json'
assert not OUT.exists()
records = []
for shape in ('2x3x3','2x4x5','2x4x8','3x4x4','3x4x5','3x5x5'):
    url = 'https://fmm.univ-lille.fr/'+shape+'_raw.mpl.bz2'
    target = DEST/(shape+'_raw.mpl.bz2')
    assert not target.exists()
    with urlopen(url, timeout=20) as response:
        data = response.read(2_000_001)
        assert len(data) <= 2_000_000
    decoded = bz2.decompress(data).decode('ascii')
    with target.open('xb') as f:
        f.write(data)
    records.append(dict(shape=shape,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    print(json.dumps(records[-1]), flush=True)
    print(decoded[:300], flush=True)
OUT.write_text(json.dumps(dict(records=records),indent=2)+'\n')
