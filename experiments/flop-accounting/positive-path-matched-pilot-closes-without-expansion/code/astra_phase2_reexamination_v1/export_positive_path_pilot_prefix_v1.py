"""Mean-only export to a unique guarded pilot payload."""
import argparse
import os
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent;G22=HERE.parent/'public_g22_operator_features_v1'
ap=argparse.ArgumentParser();ap.add_argument('--branch',required=True,choices=('control','positive'));a=ap.parse_args()
out=HERE/f'payload_positive_path_pilot_prefix_{a.branch}_v1.npz';assert not out.exists()
model=G22/f'GNN_ASTRA_POSITIVE_PATH_PILOT_{a.branch.upper()}_V1_MODEL.pt';assert model.exists()
os.environ.update(EXPORT_FIRST='1',EXPORT_VHEAD='0')
sys.argv=[str(G22/'export_gnn_payload.py'),str(model),str(out)]
runpy.run_path(str(G22/'export_gnn_payload.py'),run_name='__main__')
