"""Fixed20/4 adaptation using the inherited, unchanged GNN trainer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;G22=HERE.parent/'public_g22_operator_features_v1'
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=('control','positive'),required=True);a=ap.parse_args()
assert json.loads((HERE/'positive_path_pilot_rows_v1.json').read_text())['honest_profile_gate']
rows=HERE/('rows_compact_feed_fullpair_v1' if a.branch=='control' else 'rows_positive_path_pilot_v1')
tag=f'ASTRA_POSITIVE_PATH_PILOT_{a.branch.upper()}_V1'
for suffix in ('MODEL.pt','RESULT.json'):assert not (G22/f'GNN_{tag}_{suffix}').exists()
for key in list(os.environ):
    if key.startswith('GNN_'):del os.environ[key]
os.environ.update(GNN_FIRST='1',GNN_HIDDEN='128',GNN_CM='16',GNN_LR='5e-4',GNN_LR_MIN='5e-5',
                  GNN_INIT=str(G22/'GNN_ASTRA_COMPACT_FEED_FULLPAIR_V1_C64_E40_MODEL.pt'))
sys.argv=[str(G22/'train_gnn_oracle.py'),str(rows),tag,'40','64']
path=G22/'train_gnn_oracle.py';source=path.read_text()
needle='        rank = int(d.name.split("_")[1])'
assert source.count(needle)==1
source=source.replace(needle,needle+'\n        if rank not in range(35,59): continue')
print(json.dumps(dict(branch=a.branch,trainer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      train=20,valid=4,epochs=40,change='only the predeclared rank filter; inherited trainer, final checkpoint')),flush=True)
exec(compile(source,str(path),'exec'),dict(__file__=str(path),__name__='__main__'))
