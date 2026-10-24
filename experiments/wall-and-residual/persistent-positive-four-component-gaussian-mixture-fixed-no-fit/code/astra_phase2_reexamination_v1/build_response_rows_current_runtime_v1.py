"""Inert builder for ONE frozen, no-refit response-row adjusted re-evaluation.

No model, data, estimator or numerical experiment runs here.  The opt-in build
reads the authenticated parameter NPZ offline, constructs a FlopScope-only
candidate in memory, verifies its source, then creates the NEW target with
apply_patch.  Existing files are never deliberately replaced or repaired.

The parent scalar-final Q payload is retained as dead source only.  The active
entry point uses the response-row P payload on every full-feature layer hook.
The existing final post-hook break remains, so there is no layer15 CP birth.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
TARGET = HERE / "candidate_response_rows_current_runtime_v1.py"
BASE_NAME = "candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py"
PAYLOAD_NAME = "payload_response_rows_v1.npz"
HELPER_NAME = "radial_response_rows_v1.py"
PINS = {
    BASE_NAME: "3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757",
    PAYLOAD_NAME: "6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c",
    HELPER_NAME: "bb496d4a205d4669d35050e1f56560fbd303a8125a215cb560a64ee4e2a785f5",
    "build_response_rows_cost_v1.py": "7bfc46031f081384ec8bc984f63121d1a98a91ddda46764064b5d53be9333390",
    "build_response_rows_cost_v2.py": "085919613a6a386074a7395b52307bfb5fc26a928ca8abebd35fff6c19c4773f",
}
PAYLOAD_KEYS = {
    "centers", "scales", "rms", "msg_w", "msg_b", "enc0_w", "enc0_b",
    "enc2_w", "enc2_b", "enc4_w", "enc4_b", "head", "head_b", "gain",
    "first",
}


def require(condition, message):
    """Keep checks active even when Python was invoked with optimization."""
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_pins():
    found = {name: sha256(HERE / name) for name in PINS}
    for name, expected in PINS.items():
        require(found[name] == expected, "Immutable input hash mismatch: " + name)
    return found


def node_text(source, node):
    """Whole physical lines, including indentation and the terminal newline."""
    return "".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def definitions(tree, name, cls=ast.FunctionDef):
    return [n for n in tree.body if isinstance(n, cls) and n.name == name]


def assignment(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id == name]
    require(len(matches) == 1, "Expected exactly one assignment: " + name)
    return matches[0]


def replace_once(source, old, new, label):
    require(source.count(old) == 1, "Exact edit count failed: " + label)
    return source.replace(old, new, 1)


def is_final_layer(test):
    expected = ast.parse("l == depth - 1", mode="eval").body
    return ast.dump(test, include_attributes=False) == ast.dump(expected, include_attributes=False)


def named_calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


def make_helper(source):
    tree = ast.parse(source)
    nodes = definitions(tree, "omitted_radial_source")
    require(len(nodes) == 1, "Expected one frozen omitted-source helper")
    helper = node_text(source, nodes[0])
    require(helper.count("np.") == 4, "Unexpected helper NumPy spelling count")
    helper = helper.replace("np.", "fnp.")
    helper = replace_once(
        helper,
        "    contraction -= fnp.sum(diagonal * a * beta)\n",
        "    contraction = contraction - fnp.sum(diagonal * a * beta)\n",
        "historical FlopScope immutable-scalar compatibility fix",
    )
    converted = ast.parse(helper).body[0]
    require(not any(isinstance(n, ast.AugAssign) for n in ast.walk(converted)),
            "Omitted-source helper must not mutate a FlopScope scalar")
    require(sum(isinstance(n, ast.MatMult) for n in ast.walk(converted)) == 2,
            "Frozen helper must retain both factor contractions")
    require(not any(isinstance(n, ast.Name) and n.id == "np" for n in ast.walk(converted)),
            "Unexpected ordinary NumPy reference in candidate helper")
    return helper


def make_predict(source):
    tree = ast.parse(source)
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef),
            "Expected one prediction function")
    function = tree.body[0]
    loops = [n for n in function.body if isinstance(n, ast.For)]
    require(len(loops) == 1, "Expected exactly one layer loop")
    final_branches = [n for n in loops[0].body if isinstance(n, ast.If) and is_final_layer(n.test)]
    require(len(final_branches) == 2, "Expected early scalar branch and final post-hook break")
    require(len(named_calls(final_branches[0], "nonlinear_scalar")) == 1,
            "Early final branch is not the pinned scalar consumer")
    require(len(final_branches[1].body) == 1 and isinstance(final_branches[1].body[0], ast.Break)
            and not final_branches[1].orelse, "Final post-hook break changed")
    source = replace_once(source, node_text(source, final_branches[0]), "", "remove scalar-final early branch")
    source = replace_once(
        source,
        "        live = alive[:, None] * alive[None, :]\n",
        "        live = alive[:, None] * alive[None, :]\n"
        "        response_gate = flops.stats.norm.cdf(mu / fnp.sqrt(fnp.maximum(vv, 1e-10)))\n"
        "        response_score = fnp.where(alive, fnp.minimum(response_gate, 1 - response_gate), -1.)\n"
        "        response_indices, response_mask = select(response_score, gate_floor)\n"
        "        omitted = alive & ~response_mask\n",
        "frozen response-row selection",
    )
    source = replace_once(
        source,
        "            sub = pp[live_indices]\n"
        "            fnp.put(cp_pre, live_indices[:, None] * n + live_indices[None, :], plane(sub, lam))\n",
        "            left = pp[response_indices]\n"
        "            right = pp[live_indices]\n"
        "            response_plane = _mm(left * left, (right * lam).T).astype(fnp.float64)\n"
        "            fnp.put(cp_pre, response_indices[:, None] * n + live_indices[None, :], response_plane)\n",
        "rectangular response-row CP plane read",
    )
    source = replace_once(
        source,
        "        host = m.copy()\n",
        "        host = m.copy()\n"
        "        if len(lam):\n"
        "            source_sum = omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host)\n"
        "            denominator = fnp.trace(C) ** 2 + 2 * fnp.sum(C * C) if metric == 'covariance' else n * (n + 2)\n"
        "            radial_new = radial_new + source_sum / denominator\n",
        "frozen omitted r=0 radial source before learned feedback",
    )
    return source


def audit_predict(source):
    node = ast.parse(source).body[0]
    defaults = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):], node.args.defaults))
    require(ast.literal_eval(defaults["gate_floor"]) == 0.01, "Response threshold is not frozen .01")
    for forbidden in ("nonlinear_scalar", "_cov_compact_diagonal", "_star_compact_diagonal", "final_hook"):
        require(not named_calls(node, forbidden), "Active prediction reaches forbidden scalar-final code: " + forbidden)
    require(len(named_calls(node, "hook")) == 1, "Expected one unconditional-per-layer optional full-feature hook")
    require(len(named_calls(node, "omitted_radial_source")) == 1, "Expected one omitted radial-source call")
    require(len(named_calls(node, "_push_bank_shared_newborn")) == 1, "Bank push ownership changed")
    loops = [n for n in node.body if isinstance(n, ast.For)]
    require(len(loops) == 1, "Prediction layer-loop count changed")
    body = loops[0].body
    final = [i for i, n in enumerate(body) if isinstance(n, ast.If) and is_final_layer(n.test)]
    require(len(final) == 1, "Only the post-hook final break may remain")
    stop = final[0]
    require(len(body[stop].body) == 1 and isinstance(body[stop].body[0], ast.Break),
            "Final post-hook break missing")
    preceding = body[stop - 1]
    require(isinstance(preceding, ast.Expr) and isinstance(preceding.value, ast.Call)
            and isinstance(preceding.value.func, ast.Attribute)
            and isinstance(preceding.value.func.value, ast.Name)
            and preceding.value.func.value.id == "means" and preceding.value.func.attr == "append",
            "Final break must follow output append")
    hook_indices = [i for i, n in enumerate(body) if named_calls(n, "hook")]
    require(len(hook_indices) == 1 and hook_indices[0] < stop, "Full hook must run before final break")
    birth_indices = [i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "birth" for t in n.targets)]
    require(len(birth_indices) == 1 and birth_indices[0] > stop, "Layer15 birth must remain unreachable")
    require(source.count("fnp.fill_diagonal(cp_pre, diagonal)") == 1, "Full CP marginal diagonal was lost")
    require(source.count("previous, previous_alive = (live_indices, alive)") == 1,
            "Active-state carry must not use response-row indices")
    require(source.count("selected, births = select(fnp.minimum(g, 1 - g), gate_floor)") == 1,
            "Frozen birth selection changed")


def build():
    require(not os.path.lexists(TARGET), "Refusing any existing target: " + TARGET.name)
    initial_hashes = check_pins()
    builder_path = Path(__file__).resolve()
    builder_hash = sha256(builder_path)
    source = (HERE / BASE_NAME).read_text(encoding="utf-8")
    tree = ast.parse(source)
    predicts = definitions(tree, "_predict_binary")
    estimators = definitions(tree, "Estimator", ast.ClassDef)
    require(len(predicts) == 4, "Pinned parent prediction-definition count changed")
    require(len(estimators) == 3, "Pinned parent Estimator-definition count changed")
    require(len(definitions(tree, "_mm")) == 4, "Pinned D4 dispatch-definition count changed")
    require(not definitions(tree, "omitted_radial_source"), "Parent unexpectedly already has omitted-source helper")
    literals = assignment(tree, "_GNN_LITERAL")
    original_payload = ast.literal_eval(literals.value)
    require(set(original_payload) == PAYLOAD_KEYS, "Parent P schema differs from frozen response P")

    # Offline serialization only: no estimator, FlopScope, Torch or network import.
    import numpy as np
    with np.load(HERE / PAYLOAD_NAME, allow_pickle=False) as archive:
        require(len(archive.files) == len(PAYLOAD_KEYS) and set(archive.files) == PAYLOAD_KEYS,
                "Frozen response payload key mismatch")
        payload = {key: archive[key].copy() for key in archive.files}
    for key, value in payload.items():
        require(value.dtype.kind in "fiu" and bool(np.isfinite(value).all()),
                "Non-finite or non-numeric response parameter: " + key)
        require(value.shape == np.asarray(original_payload[key]).shape,
                "Response/parent P shape mismatch: " + key)
    require(payload["first"].size == 1 and payload["first"].reshape(-1)[0] == 1,
            "Response payload must preserve the exact first-layer bypass")
    require(payload["centers"].shape[0] == 16 and payload["head"].shape == (16, 64),
            "Expected all16 response heads with64 recurrent channels")
    serialized = {key: value.tolist() for key, value in payload.items()}
    payload_line = "_GNN_LITERAL = " + repr(serialized) + "\n"
    decoded = ast.literal_eval(ast.parse(payload_line).body[0].value)
    for key, value in payload.items():
        roundtrip = np.asarray(decoded[key], dtype=value.dtype)
        require(roundtrip.shape == value.shape and roundtrip.tobytes(order="C") == value.tobytes(order="C"),
                "Literal serialization lost response payload bits: " + key)

    helper = make_helper((HERE / HELPER_NAME).read_text(encoding="utf-8"))
    predict = make_predict(node_text(source, predicts[-1]))
    audit_predict(predict)
    estimator = replace_once(
        node_text(source, estimators[-1]),
        "        return _predict_compact_final(mlp)\n",
        "        return _predict_learned_compact(mlp)\n",
        "active entry point selects response P, never parent Q",
    )
    replacements = {
        id(literals): payload_line,
        id(predicts[-1]): helper + "\n\n" + predict,
        id(estimators[-1]): estimator,
    }
    # Substitute complete physical spans in reverse order.  No global source
    # rewrite can accidentally edit an older shadow definition or the runtime.
    lines = source.splitlines(keepends=True)
    changed = [n for n in tree.body if id(n) in replacements]
    require(len(changed) == 3, "Exactly three existing top-level nodes may change")
    for node in sorted(changed, key=lambda n: n.lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [replacements[id(node)]]
    generated = "".join(lines)
    generated += "\n# Frozen no-refit response-row adjusted re-evaluation; no accuracy claim.\n"
    generated += "# Builder SHA256: " + builder_hash + "\n"
    for name, digest in initial_hashes.items():
        generated += "# Input SHA256 " + name + ": " + digest + "\n"
    new_tree = ast.parse(generated)
    helper_nodes = definitions(new_tree, "omitted_radial_source")
    require(len(helper_nodes) == 1 and len(new_tree.body) == len(tree.body) + 1,
            "Unexpected generated top-level additions")
    comparable = [n for n in new_tree.body if n is not helper_nodes[0]]
    untouched = 0
    for old, new in zip(tree.body, comparable):
        if id(old) not in replacements:
            require(node_text(source, old) == node_text(generated, new),
                    "Unrelated top-level source changed near original line " + str(old.lineno))
            untouched += 1
    require(node_text(generated, helper_nodes[0]) == helper, "Generated helper changed")
    require(node_text(generated, definitions(new_tree, "_predict_binary")[-1]) == predict,
            "Last generated prediction differs from reviewed edit")
    learned = definitions(new_tree, "_predict_learned_compact")
    require(len(learned) == 1, "Expected exactly one learned full-feature wrapper")
    wrapper_calls = named_calls(learned[0], "_predict_binary")
    require(len(wrapper_calls) == 1 and len(wrapper_calls[0].args) == 1
            and len(wrapper_calls[0].keywords) == 1 and wrapper_calls[0].keywords[0].arg == "hook",
            "Full-feature wrapper must use the frozen default threshold and no scalar-final Q")
    require(len(named_calls(learned[0], "features")) == 1 and len(named_calls(learned[0], "gnn_step")) == 1,
            "Full-feature P consumer changed")
    last_estimator = definitions(new_tree, "Estimator", ast.ClassDef)[-1]
    require(len(named_calls(last_estimator, "_predict_learned_compact")) == 1
            and not named_calls(last_estimator, "_predict_compact_final"), "Q remains reachable from Estimator")
    for node in ast.walk(new_tree):
        if isinstance(node, ast.Import):
            require(all(a.name in {"flopscope", "flopscope.numpy"} for a in node.names),
                    "Candidate imports non-FlopScope numerical machinery")
        elif isinstance(node, ast.ImportFrom):
            require(node.module == "whestbench", "Unexpected candidate import-from module")
        elif isinstance(node, ast.Name):
            require(node.id not in {"np", "numpy", "torch"}, "Ordinary numerical backend leaked into candidate")
    compile(generated, TARGET.name, "exec")  # Syntax only: never execute candidate.
    require(check_pins() == initial_hashes and sha256(builder_path) == builder_hash,
            "Builder inputs changed during source generation")
    require(not os.path.lexists(TARGET), "Target appeared before apply_patch; refusing overwrite")
    patch_program = shutil.which("apply_patch")
    require(patch_program is not None, "apply_patch is unavailable; no source was created")
    patch = "*** Begin Patch\n*** Add File: " + TARGET.name + "\n"
    patch += "".join("+" + line + "\n" for line in generated.splitlines())
    patch += "*** End Patch\n"
    process = subprocess.run([patch_program], input=patch, text=True, encoding="utf-8",
                             cwd=HERE, capture_output=True, check=False)
    require(process.returncode == 0,
            "apply_patch failed; preserve any artifact and inspect: " + (process.stdout + process.stderr)[-2000:])
    require(TARGET.is_file() and not TARGET.is_symlink(), "Created target is not an ordinary source file")
    expected_hash = hashlib.sha256(generated.encode("utf-8")).hexdigest()
    require(sha256(TARGET) == expected_hash, "Generated source bytes differ; preserve artifact, do not overwrite")
    require(check_pins() == initial_hashes and sha256(builder_path) == builder_hash,
            "Input changed during output creation; preserve target as unqualified")
    return {
        "status": "BUILT_SOURCE_ONLY_NOT_EXECUTED",
        "target": str(TARGET), "candidate_sha256": expected_hash,
        "builder_sha256": builder_hash, "input_sha256": initial_hashes,
        "changed_existing_top_level_nodes": 3,
        "added_helper_functions": 1, "byte_identical_other_top_level_nodes": untouched,
        "payload_keys": sorted(payload),
        "payload_shapes": {k: list(v.shape) for k, v in payload.items()},
        "response_threshold": 0.01,
        "full_feature_hook_layers": 16,
        "layer0_gnn_bypass_unchanged": True,
        "post_hook_final_break_preserved": True,
        "parent_Q_source_retained_but_unreachable": True,
        "D4_bulk_copy_runtime_unchanged": True,
        "omitted_source_scope": "frozen r=0 marginal-pair source only",
        "numerical_parity_meter_runtime_accuracy": "NOT_TESTED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Create the new source only after independent review; never run it")
    args = parser.parse_args()
    if not args.write:
        print(json.dumps({"status": "INERT", "target": str(TARGET),
                          "required_flag": "--write",
                          "inputs_read": False, "source_created": False}, sort_keys=True))
        return
    try:
        result = build()
    except Exception as exc:
        print(json.dumps({"status": "BUILD_FAILED_PRESERVE_ARTIFACTS", "target": str(TARGET),
                          "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), flush=True)
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
