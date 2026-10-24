"""Port the frozen signed-pair cap to a new FlopScope-only candidate."""
import argparse
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--cap", required=True, type=int)
ap.add_argument("--out", required=True)
args = ap.parse_args()
target = HERE / args.out
if target.exists():
    raise FileExistsError(target)
source = (HERE / "candidate_binary_radial_v1.py").read_text()
tree = ast.parse((HERE / "radial_binary_cap.py").read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_predict")
blocks = {}
for node in fn.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ("old", "new"):
            blocks[node.targets[0].id] = ast.literal_eval(node.value).replace("np.", "fnp.")
assert source.count(blocks["old"]) == 1
source = source.replace(blocks["old"], blocks["new"].replace("pair_cap", str(args.cap)))
old = "a.dtype!=fnp.float32 or b.dtype!=fnp.float32"
assert source.count(old) == 1
source = source.replace(old, 'str(a.dtype)!="float32" or str(b.dtype)!="float32"')
ast.parse(source)
target.write_text(source)
print(target)
