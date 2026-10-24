"""Share the youngest signed pair's identical off-diagonal push exactly.

Each new pair is B[:,j] + t_j e_j and B[:,j] + s_j e_j, followed by the
same live-row mask. At its first linear transport only B needs a GEMM;
the two diagonal additions use columns of the known next-layer weight.
Every older factor and both newborn factors are retained. Arithmetic
rounding changes, but there is no factor-rank or physics approximation.
"""
import argparse
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
target = HERE / args.out
if target.exists():
    raise FileExistsError(target)
source = (HERE / args.source).read_text()
tree = ast.parse(source)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_predict_binary"][-1]
predict = ast.get_source_segment(source, node)
edits = {
    "    means, states = ([], [])": "    means, states = ([], [])\n    newborn = None",
    "            pp = _mm(w32[previous,:].T,p[previous,:])":
        "            pp = _push_bank_shared_newborn(w32,p,previous,newborn)",
    "        pp = _mm(w32[previous, :].T, p[previous, :])":
        "        pp = _push_bank_shared_newborn(w32,p,previous,newborn)",
    "        p = fnp.concatenate([g.astype(fnp.float32)[:, None] * pp, p1[:, selected].astype(fnp.float32), p2[:, selected].astype(fnp.float32)], axis=1)":
        """        newborn = (int(pp.shape[1]), selected,
                   (bp[:,selected]*alive[:,None]).astype(fnp.float32),
                   (t[selected]*alive[selected]).astype(fnp.float32),
                   (s[selected]*alive[selected]).astype(fnp.float32))
        p = fnp.concatenate([g.astype(fnp.float32)[:, None] * pp, p1[:, selected].astype(fnp.float32), p2[:, selected].astype(fnp.float32)], axis=1)""",
}
for old, new in edits.items():
    assert predict.count(old) == 1, old
    predict = predict.replace(old, new)
helper = '''
def _push_bank_shared_newborn(w,p,previous,newborn):
    wp = w[previous,:].T
    if newborn is None:
        return _mm(wp,p[previous,:])
    old_columns,selected,base,t,s = newborn
    inherited = _mm(wp,p[previous,:old_columns])
    common = _mm(wp,base[previous,:])
    directions = w[selected,:].T
    return fnp.concatenate([inherited,common+directions*t[None,:],
                            common+directions*s[None,:]],axis=1)
'''
source += "\n\n" + helper + "\n\n" + predict + "\n"
ast.parse(source)
target.write_text(source)
print(target)
