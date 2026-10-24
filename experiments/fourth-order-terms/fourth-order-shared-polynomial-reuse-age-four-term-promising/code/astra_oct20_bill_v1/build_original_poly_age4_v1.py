"""Build age4 with the original polynomial, optionally exact copies and micro.

Source and small Python-table checks only; no estimator import or prediction.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

D = Path(__file__).resolve().parent
K = D.parent / "fable_oct19_kc_v1"
BASE_SCI_SHA = "e2f0c4cc4552c4770b383fa96ab2dc30b3a54dc76a27f2bcf614c386119975f9"
BASE_REL_SHA = "b41627a6cc4d3c67280ab85fd7400ff07a0a167621326fcdb07690abe84ccac7"
SCREEN_SHA = "dc654294e53bf3a3be61cb4d23b2df2926b6a4b2f1a24a1efbe4ee3762327366"
COPY_SHA = "099cebb660c23d3b01b52dde359853421939a18584241971d646b062d708de4f"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def freeze(path, text):
    if path.exists():
        assert path.read_text() == text, "Frozen file differs: " + str(path)
    else:
        path.write_text(text)


base_sci = (K / "candidate_rel_v4a_sci_v1.py").read_text()
base_rel = (K / "candidate_rel_v4a.py").read_text()
screen = (D / "candidate_k31_npswap_screen_sci_v1.py").read_text()
assert sha(base_sci) == BASE_SCI_SHA and sha(base_rel) == BASE_REL_SHA
assert sha(screen) == SCREEN_SHA
assert sha((D / "tail_copy_all_exact_complete_v1.py").read_text()) == COPY_SHA

# Preserve the screened age wrapper exactly, with explicit names that do not
# imply the rejected K31 reassociation or its weighted-copy implementation.
start = screen.index("# Runtime screen control: append the historical fourth-age coefficient .6.")
old_tail = screen[start:]
tail = old_tail.replace("_O20_NPSWAP_ADD_AGE4", "_O20_ORIGPOLY_ADD_AGE4")
tail = tail.replace("_O20_NPSWAP_PARENT_AGES", "_O20_ORIGPOLY_PARENT_AGES")
tail = tail.replace("_O20_ORIGPOLY_ADD_AGE4 = False", "_O20_ORIGPOLY_ADD_AGE4 = True")
tail_path = D / "tail_origpoly_age4_v1.py"
freeze(tail_path, tail)
assert "K31PAIR" not in tail and "_K3LB_PARENT_LEFT" not in tail

# Extract only Python constants and generic polynomial-table definitions.
# Neither the estimator module nor any array backend is imported/executed.
tree = ast.parse(base_sci)
env = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and (
                    target.id.startswith("_NP_") or target.id == "_S19_LAM"):
                env[target.id] = value
functions = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        functions.setdefault(node.name, []).append(node)


def define(node, target):
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<source-table-only>", "exec"), target)


define(functions["_np_ages"][0], env)
env["_K3LB_PARENT_AGES"] = env["_np_ages"]
env["_K3LB_LAYER"] = [None]
define(functions["_np_ages"][-1], env)
define(functions["_np_key"][-1], env)
define(functions["_np_tables"][-1], env)
old_age_function = env["_np_ages"]
old_table = env["_np_tables"](1024)
exec(compile(tail, "<age-wrapper-only>", "exec"), env)
new_table = env["_np_tables"](1024)
assert old_table == new_table, "The generic table must not change with age count"
age_rows = []
for layer in [None, *range(16)]:
    env["_K3LB_LAYER"][0] = layer
    old = old_age_function()
    new = env["_np_ages"]()
    assert len(old) == 3 and new == old + (0.6,)
    env["_O20_ORIGPOLY_ADD_AGE4"] = False
    assert env["_np_ages"]() == old
    env["_O20_ORIGPOLY_ADD_AGE4"] = True
    age_rows.append(dict(layer=layer, original=old, age4=new))

# The live update evicts by the live age tuple's length, and both the original
# and exact-copy fold read the same global age function on every invocation.
update = ast.get_source_segment(base_sci, functions["_np_update"][-1])
assert "b <= l - len(_np_ages())" in update
original_fold = ast.get_source_segment(base_sci, functions["_npx_left"][-1])
assert "ages = _np_ages()" in original_fold
copy_fold = (D / "tail_npx_copy_exact_v1.py").read_text()
assert "ages = _np_ages()" in copy_fold
copy_checks = json.loads((D / "NPX_COPY_EXACT_CHECK_v1.json").read_text())
age4_checks = [x for x in copy_checks["records"] if x["ages"] == 4]
assert len(age4_checks) == 4 and all(
    x["fold_identical"] and x["auxiliary_state_identical"] for x in age4_checks)

specifications = [
    ("origpoly_age4_astra", [tail_path]),
    ("copyall_origpoly_age4_astra", [D / "tail_copy_all_exact_complete_v1.py", tail_path]),
    ("micro_copyall_ecut_origpoly_age4_astra", [
        D / "tail_micro_v1.py", D / "tail_copy_all_exact_complete_v1.py",
        D / "tail_ecut_radical_v1.py", tail_path]),
]
outputs = []
for name, pieces in specifications:
    sci = K / f"candidate_{name}_sci_v1.py"
    rel = K / f"candidate_rel_{name}.py"
    assert not sci.exists() and not rel.exists(), "Append-only build name exists"
    subprocess.run(["bash", str(K / "make_stack_v1.sh"), name,
                    *(str(p) for p in pieces)], cwd=K, check=True)
    for kind, path, prefix in [("science", sci, base_sci), ("release", rel, base_rel)]:
        source = path.read_text()
        assert source.startswith(prefix)
        appended = source[len(prefix):]
        assert appended.endswith("".join(p.read_text() for p in pieces))
        assert not re.search(r"^_GN_KC\s*=", appended, re.M)
        assert not re.search(r"^_ASTRA_K31PAIR_ON\s*=", appended, re.M)
        assert "_o20_k31_npswap_left_v1" not in appended
        compiled = ast.parse(source)
        bindings = [n.value.id for n in compiled.body if isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Name)
                    and any(isinstance(t, ast.Name) and t.id == "_K3LB_PARENT_LEFT"
                            for t in n.targets)]
        expected = "_astra_compute_left_v1" if len(pieces) == 1 else "_o20_npx_copy_exact_left_v1"
        assert bindings[-1] == expected
        final_ages = [n for n in compiled.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_np_ages"][-1]
        assert ast.dump(final_ages, include_attributes=False) == ast.dump(
            ast.parse(tail).body[-1], include_attributes=False)
        if name.startswith("micro_"):
            assert "_ASTRA_FHM_ACTIVE_CORE = _tr_clearing(_astra_fhm_core)" in appended
        outputs.append(dict(name=name, kind=kind, path=str(path), sha256=sha(source),
                            prefix_sha256=sha(prefix), prefix_byte_identical=True,
                            final_np_binding=bindings[-1], late_kc_override=False,
                            k31_reassociation=False, tails=[
                                dict(path=str(p), sha256=sha(p.read_text())) for p in pieces]))

# The interrupted v1 test was a no-op for the optimization flag, but its age4
# switch was live. Preserve this adverse current-lineage observation.
old_log = (K / "eval_k31pair_fresh_v1.log").read_text()
observed = []
for m in re.finditer(r"^  seed (\d+) (\{.*\}) U ([0-9.e+-]+) bill ([0-9.e+-]+)", old_log, re.M):
    observed.append(dict(seed=int(m[1]), setting=json.loads(m[2]), U=float(m[3]), bill=float(m[4])))
assert len(observed) == 4 and all(x["seed"] == 7100021 for x in observed)
b, a = observed[0], observed[2]
known = dict(seed=7100021, U=a["U"], bill=a["bill"],
             raw_pct=100 * (a["U"] / b["U"] - 1),
             bill_pct=100 * (a["bill"] / b["bill"] - 1),
             adjusted_pct=100 * (a["U"] * a["bill"] / (b["U"] * b["bill"]) - 1),
             note="One current native seed only; interrupted v1 screen, not a four-net closure")
receipt = dict(
    utc=subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
    reference="science32_rel_v4a_m15; known native row against its same-seed v4a control",
    status="SOURCE_BUILT_STATIC_AUDITED_NOT_BOUNDARY_TESTED",
    source_screen_sha256=SCREEN_SHA, source_equivalent_screen_setting={
        "_ASTRA_K31PAIR_ON": False, "_O20_NPSWAP_ON": False,
        "_O20_NPSWAP_ADD_AGE4": True},
    tail=dict(path=str(tail_path), sha256=sha(tail)),
    age_rows=age_rows, generic_tables_identical=True,
    dynamic_retention_length=True, exact_copy_age4_existing_tiny_checks=age4_checks,
    known_original_poly_native_observation=known, outputs=outputs,
    notes=[
        "No current complete four-net or32 original-polynomial age4 closure located.",
        "Historical32 vs v34y: raw -.142%, bill +.089%, adjusted -.053%; old lineage only.",
        "Reuse age4 conditional raw gain must not be transferred additively to original polynomial.",
        "COPY_ALL was exact on rank35 native and12 tiny folds including four age4 cases; no all32 copy-only parity claim.",
        "Full stack preserves the base KC schedule and adds no late-KC override.",
        "No estimator setup/prediction, heavy job, queue edit, or submission was run.",
    ],
)
freeze(D / "ORIGINAL_POLY_AGE4_BUILD_v1.json", json.dumps(receipt, indent=2) + "\n")
print(json.dumps(dict(known_original_poly_native_observation=known,
                     tail=receipt["tail"], outputs=outputs), indent=2))
