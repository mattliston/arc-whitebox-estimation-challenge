"""Matched full-bank screen of the physical activation-gate GNN variant."""
from pathlib import Path
import runpy
import k3_radial_numpy
import radial_gnn_numpy
from radial_binary_cubic import predict
from radial_gnn_physicalgate import gnn_step

k3_radial_numpy.predict = predict
radial_gnn_numpy.gnn_step = gnn_step
runpy.run_path(str(Path(__file__).resolve().parent / "screen_radial_closedloop.py"), run_name="__main__")
