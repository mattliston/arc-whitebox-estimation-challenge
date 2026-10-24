"""Mechanical FlopScope port of the independently checked full pair map."""
import argparse
import ast
from pathlib import Path

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument("--out",required=True)
args=ap.parse_args()
target=HERE/args.out
if target.exists():
    raise FileExistsError(target)
source=(HERE/"candidate_compact_radialfeed_unlearned_d6_v1.py").read_text()
reference=(HERE/"radial_mehler.py").read_text()
for node in ast.parse(reference).body:
    if isinstance(node,ast.FunctionDef) and node.name in ("derivatives","make_nonlinear"):
        part=ast.get_source_segment(reference,node).replace("np.","fnp.").replace("base.gaussian_raw","gaussian_raw")
        source+="\n\n"+part
source+="\n\nnonlinear=make_nonlinear(6,2,True,True)\n"
ast.parse(source)
target.write_text(source)
print(target)
