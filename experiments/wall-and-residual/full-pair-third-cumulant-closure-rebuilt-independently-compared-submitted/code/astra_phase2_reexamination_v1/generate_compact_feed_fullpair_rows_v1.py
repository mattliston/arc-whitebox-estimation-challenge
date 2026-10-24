"""Honest matched source/consumer rows; truth is written only after features."""
from pathlib import Path
import runpy
import k3_radial_numpy
from radial_compact_feed_fullpair_v1 import predict as base_predict


def predict(W,**kwargs):
    return base_predict(W,gate_floor=.01,**kwargs)


k3_radial_numpy.predict=predict
runpy.run_path(str(Path(__file__).resolve().parent/"generate_radial_rows.py"),run_name="__main__")
