"""Strict39-feature matched-model full-loop development screen."""
from pathlib import Path
import runpy
import k3_radial_numpy
import radial_features
from radial_compact_tensorfeat_v1 import predict as base_predict,features,FEATURE_NAMES


def predict(W,**kwargs):
    return base_predict(W,gate_floor=.01,**kwargs)


k3_radial_numpy.predict=predict
radial_features.features=features
radial_features.FEATURE_NAMES=FEATURE_NAMES
runpy.run_path(str(Path(__file__).resolve().parent/"screen_radial_closedloop.py"),run_name="__main__")
