"""Inert exclusive mechanical derivation of the fixed exact cleanup."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "resident_gram512_v1/estimator.py"
PARENT_SHA = "0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def segment(text, name):
    node = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
    lines = text.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1]))
    end = sum(map(len, lines[:node.end_lineno]))
    return start, end, text[start:end]


def changed(old, changes):
    text = old
    for before, after, count in changes:
        if text.count(before) != count:
            raise RuntimeError("Unexpected fixed source fragment count")
        text = text.replace(before, after)
    inverse = text
    for before, after, count in reversed(changes):
        if inverse.count(after) != count:
            raise RuntimeError("Ambiguous inverse fragment")
        inverse = inverse.replace(after, before)
    if inverse != old:
        raise RuntimeError("Inverse source mismatch")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.print_help(); return
    raw = PARENT.read_bytes()
    if sha(raw) != PARENT_SHA:
        raise RuntimeError("Frozen parent changed")
    source = raw.decode()
    feature_changes = [
        ("    postscale = fnp.sqrt", "    skplane_square = skplane * skplane\n    postscale = fnp.sqrt", 1),
        ("fnp.sum(skplane * skplane,", "fnp.sum(skplane_square,", 2),
    ]
    body_changes = [
        ("        live_indices, alive = select(flops.stats.norm.cdf(mu / fnp.sqrt(fnp.maximum(vv, 1e-10))), gate_floor)\n        live = alive[:, None] * alive[None, :]\n        response_gate = flops.stats.norm.cdf(mu / fnp.sqrt(fnp.maximum(vv, 1e-10)))",
         "        response_gate = flops.stats.norm.cdf(mu / fnp.sqrt(fnp.maximum(vv, 1e-10)))\n        live_indices, alive = select(response_gate, gate_floor)\n        live = alive[:, None] * alive[None, :]", 1),
        ("        residual = target - inherited - trace_post\n", "        # Innovation owns only the diagonal of the old residual temporary.\n", 1),
        ("        innovation = fnp.diag(residual).copy()", "        innovation = (fnp.diag(target) - fnp.diag(inherited) - fnp.diag(trace_post)).copy()", 1),
        ("        p1, p2 = (bp.copy(), bp.copy())\n        fnp.fill_diagonal(p1, t)\n        fnp.fill_diagonal(p2, s)\n", "        # Newest numerical tails are owned by the shared-newborn tuple.\n", 1),
        ("        p = fnp.concatenate([g.astype(fnp.float32)[:, None] * pp, p1[:, selected].astype(fnp.float32), p2[:, selected].astype(fnp.float32)], axis=1)", "        p = g.astype(fnp.float32)[:, None] * pp", 1),
        ("columns=p.shape[1]", "columns=len(lam)", 1),
    ]
    edits = []
    for name, changes in (("features", feature_changes), ("_predict_binary", body_changes)):
        start, end, old = segment(source, name)
        edits.append((start, end, old, changed(old, changes), name))
    candidate = source
    for start, end, old, new, name in sorted(edits, reverse=True):
        candidate = candidate[:start] + new + candidate[end:]
    ast.parse(candidate)
    inverse = candidate
    for start, end, old, new, name in edits:
        if inverse.count(new) != 1:
            raise RuntimeError("Whole-function inverse ambiguous")
        inverse = inverse.replace(new, old)
    if inverse != source:
        raise RuntimeError("Other source/payloads changed")
    data = candidate.encode()
    record = dict(passed=True, executed=False, parent=str(PARENT), parent_sha256=PARENT_SHA,
                  candidate_sha256=sha(data), builder_sha256=sha(Path(__file__).read_bytes()),
                  protocol_sha256=sha((HERE / "PROTOCOL.md").read_bytes()),
                  edited_functions=[e[4] for e in edits], inverse_source_byte_exact=True,
                  all_other_source_byte_exact=True, imports_and_payloads_unchanged=True,
                  candidate=str(HERE / "estimator.py"))
    with (HERE / "estimator.py").open("xb") as stream:
        stream.write(data)
    with (HERE / "build.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
