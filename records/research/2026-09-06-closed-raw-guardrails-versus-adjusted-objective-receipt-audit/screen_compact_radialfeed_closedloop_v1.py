"""Strict matched-head closed-loop screen for the compact radial source."""
from pathlib import Path
import runpy
import k3_radial_numpy
from radial_compact_radialfeed_v1 import predict as base_predict


def predict(W,**kwargs):
    return base_predict(W,gate_floor=.01,**kwargs)


k3_radial_numpy.predict=predict
runpy.run_path(str(Path(__file__).resolve().parent/"screen_radial_closedloop.py"),run_name="__main__")
