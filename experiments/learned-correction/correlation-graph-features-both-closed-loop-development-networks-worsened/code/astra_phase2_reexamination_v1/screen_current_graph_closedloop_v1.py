"""Both the graph features and all applied heads match this new payload."""
from pathlib import Path
import runpy
import k3_radial_numpy
import radial_features
from radial_graph_features_v1 import features
from radial_compact_feed_fullpair_v1 import predict as base_predict

def predict(W, **kwargs):
    return base_predict(W, gate_floor=.01, **kwargs)

k3_radial_numpy.predict = predict
radial_features.features = features
runpy.run_path(str(Path(__file__).resolve().parent / "screen_radial_closedloop.py"), run_name="__main__")
