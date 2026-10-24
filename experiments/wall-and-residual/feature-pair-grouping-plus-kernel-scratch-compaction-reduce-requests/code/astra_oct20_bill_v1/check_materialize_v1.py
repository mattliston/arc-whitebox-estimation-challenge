"""Tiny official-API identity/bill checks; no estimator import or prediction."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

D = Path(__file__).resolve().parent
K = D.parent / "fable_oct19_kc_v1"
OFFICIAL = D.parent / "astra_crunch_runtime_v1/failure336631_v1/flopscope_0.12.3"
assert not any(name.startswith("flopscope") for name in sys.modules)
sys.path.insert(0, str(OFFICIAL))
import flopscope as fs
from flopscope import numpy as fnp

assert fs.__version__.startswith("0.12.3+")
fs.configure(symmetry_warnings=False)
source = (K / "candidate_rel_v4a_sci_v1.py").read_text()
tree = ast.parse(source)
env = dict(fnp=fnp, _rc_math=math, _BH_HOLD=[], _L3_STATS=dict(quads=0),
           _ASTRA_QL_ON=False)


def install(name, minimum=0, maximum=1000000):
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name == name and minimum <= n.lineno <= maximum]
    assert len(nodes) == 1, (name, [n.lineno for n in nodes])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<extracted-current>", "exec"), env)


for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_ASTRA_SM40_RECORDS":
                env[target.id] = ast.literal_eval(node.value)
env["_SMX_RECORDS"] = dict(env["_ASTRA_SM40_RECORDS"])
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
            and t.value.id == "_SMX_RECORDS" for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<literal-record>", "exec"), env)
install("_Sm40Prepared")
install("_astra_sm40_linear_grouped", 20000, 20100)
env["_LINEAR_PARENT_GROUPED"] = lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected fallback"))
env["_sm40_program_key"] = lambda p, i, o, r: (id(p), id(i), id(o))
sm_parent = env["_astra_sm40_linear_grouped"]
exec((D / "tail_sm_direct_v1.py").read_text(), env)
sm_new = env["_astra_sm40_linear_grouped"]
rng = np.random.default_rng(20261020)
sm_rows = []
with fs.BudgetContext(flop_budget=2**39, quiet=True) as ctx:
    for record in env["_SMX_RECORDS"].values():
        for kind in ("L", "R"):
            plan = record["prepared"][kind]
            inputs_count = max(plan[0].values()) + 1
            for shape in ((2, 3), (3, 2), (4, 5)):
                for strided in (False, True):
                    inputs = tuple(fnp.asarray(rng.normal(size=shape).astype("float32"))
                                   for _ in range(inputs_count))
                    total = sum(len(nodes) for _, nodes in plan[1])
                    storage = fnp.empty((total, *shape), dtype=fnp.float32)
                    cuts = np.cumsum([len(nodes) for _, nodes in plan[1]])[:-1].tolist()
                    prepared = env["_Sm40Prepared"](
                        (tuple(fnp.split(storage, cuts, axis=0)),
                         tuple(fnp.unstack(storage, axis=0))))
                    count = len(plan[2])
                    if strided:
                        owner = fnp.empty((count, shape[0] + 2, shape[1] + 3), dtype=fnp.float32)
                        out = owner[:, :shape[0], :shape[1]]
                    else:
                        out = fnp.empty((count, *shape), dtype=fnp.float32)
                    stats = dict(temp_peak=0, temp_live=0)
                    before = ctx.flops_used
                    old = np.asarray(sm_parent(fnp, plan, inputs, shape, stats,
                                               out=out, prepared=prepared)).copy()
                    old_flops = ctx.flops_used - before
                    before = ctx.flops_used
                    new = np.asarray(sm_new(fnp, plan, inputs, shape, stats,
                                           out=out, prepared=prepared)).copy()
                    new_flops = ctx.flops_used - before
                    assert np.array_equal(old, new), (record["name"], kind, shape, strided)
                    assert old_flops - new_flops == count * math.prod(shape)
                    # A repeat uses cached argument routing; off uses the live
                    # parent rather than accidentally leaving a direct route.
                    assert np.array_equal(new, np.asarray(
                        sm_new(fnp, plan, inputs, shape, stats, out=out, prepared=prepared)))
                    env["_O20_SMDIRECT_ON"] = False
                    assert np.array_equal(old, np.asarray(
                        sm_new(fnp, plan, inputs, shape, stats, out=out, prepared=prepared)))
                    env["_O20_SMDIRECT_ON"] = True
                    key = env["_sm40_program_key"](plan, inputs, out, prepared)
                    sm_rows.append(dict(record=record["name"], kind=kind, shape=shape,
                                        strided_out=strided, exact=True,
                                        old_flops=old_flops,
                                        new_flops=new_flops,
                                        old_calls=len(plan[1]) + 1,
                                        new_calls=len(prepared.programs[key][2])))

install("_rs_nat_perm")
install("_l3_quad", 11830, 11860)
install("_rs_level1", 11900, 11940)
install("_rs_load_a", 11870, 11890)
install("_rs_load_b", 11870, 11900)
load_a, load_b = env["_rs_load_a"], env["_rs_load_b"]
exec((D / "tail_flatload_v1.py").read_text(), env)
load_rows = []
with fs.BudgetContext(flop_budget=2**39, quiet=True) as ctx:
    for kind in (0, 1):
        for depth in (1, 2, 3, 4):
            for strided in (False, True):
                r, c = 2, 3
                rows, cols = (1 << depth) * r, (1 << depth) * c
                raw = rng.normal(size=(rows, cols)).astype("float32")
                if strided:
                    owner = fnp.asarray(np.pad(raw, ((0, 2), (0, 3))))
                    operand = owner[:rows, :cols]
                else:
                    operand = fnp.asarray(raw)
                tree_shape = (7,) * depth + (r, c)
                initial = rng.normal(size=tree_shape).astype("float32")
                old_tree, new_tree = fnp.asarray(initial), fnp.asarray(initial)
                old_x, old_slot = env["_rs_level1"](kind, depth, old_tree, (r, c))[0]
                new_x, new_slot = env["_rs_level1"](kind, depth, new_tree, (r, c))[0]
                # Bind both exact quads outside measured loading cost.
                env["_l3_quad"](old_slot, kind, depth)
                env["_l3_quad"](new_slot, kind, depth)
                env["_o20_flatload_indices"](rows * cols)
                old_fn = load_a if kind == 0 else load_b
                new_fn = env["_rs_load_a" if kind == 0 else "_rs_load_b"]
                before = ctx.flops_used
                old_fn(operand, old_x, old_slot, depth, rows, cols)
                old_flops = ctx.flops_used - before
                before = ctx.flops_used
                new_fn(operand, new_x, new_slot, depth, rows, cols)
                new_flops = ctx.flops_used - before
                assert np.array_equal(np.asarray(old_tree), np.asarray(new_tree)), (kind, depth, strided)
                assert old_flops - new_flops == rows * cols
                load_rows.append(dict(kind=kind, depth=depth, input_strided=strided,
                                      shape=[rows, cols], exact=True,
                                      old_flops=old_flops, new_flops=new_flops))

report = dict(
    utc=subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    flopscope_version=fs.__version__, sm_direct=sm_rows, flatload=load_rows,
    sm_stats=env["_O20_SMDIRECT_STATS"], flat_stats=env["_O20_FLATLOAD_STATS"],
    limitation="Tiny extracted-function checks only. No estimator imported, set up or predicted. Full native ownership/wall and current32 adjusted score remain unmeasured.",
)
target = D / "MATERIALIZE_CHECK_v1.json"
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(dict(utc=report["utc"], sm_cases=len(sm_rows), flat_cases=len(load_rows),
                     exact=True, official_bill_savings=True,
                     sm_call_frontier=[{k: row[k] for k in ("record", "kind", "old_calls", "new_calls")}
                                       for row in sm_rows[::6]], receipt=str(target)), indent=2))
