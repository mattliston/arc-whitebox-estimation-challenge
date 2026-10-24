"""Inert, exclusive-create mechanical source splice; no candidate execution."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TARGET = HERE / "candidate_response_pair_polynomial_f32_v1.py"
RECEIPT = HERE / "pair_polynomial_f32_build_v1.json"
HELPER = HERE / "pair_polynomial_f32_source_v1.py"
HELPER_SHA = "fb758f0e3e36a0ee29cb7f69490af83ece4c7e12424cb82c5122ee4e00113198"
PREFLIGHT = HERE / "pair_polynomial_f32_preflight_v1/result.json"
PREFLIGHT_SHA = "e5441d15e13ce38c186b7d15e943191a689349cb52628e6668c5a018a9567908"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def construction():
    require(digest(HELPER) == HELPER_SHA, "Frozen splice helper changed")
    require(digest(PREFLIGHT) == PREFLIGHT_SHA, "Synthetic result changed")
    preflight = json.loads(PREFLIGHT.read_text())
    require(preflight["decision"] == "SYNTHETIC_COMPONENT_GATE_PASS", "No passing preflight")
    from pair_polynomial_f32_source_v1 import sources, PARENT, PARENT_SHA
    original = PARENT.read_text()
    require(digest(PARENT) == PARENT_SHA, "Parent changed")
    code, transformed = sources()
    new = transformed.replace("def _nonlinear_pair_batch_fp32poly(", "def _nonlinear_pair_batch(", 1)
    old = code["_nonlinear_pair_batch"]
    require(original.count(old) == 1, "Ambiguous physical target")
    candidate = original.replace(old, new, 1)
    require(candidate.count(new) == 1 and candidate.replace(new, old, 1) == original,
            "Exact inverse failed")
    before, after = ast.parse(original).body, ast.parse(candidate).body
    require(len(before) == len(after), "AST node count changed")
    differences = []
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        if ast.dump(a) != ast.dump(b):
            require(isinstance(a, ast.FunctionDef) and a.name == "_nonlinear_pair_batch"
                    and isinstance(b, ast.FunctionDef) and b.name == a.name,
                    "A non-target AST node changed")
            differences.append(i)
    require(len(differences) == 1, "Exactly one changed definition required")
    compile(candidate, str(TARGET), "exec")
    receipt = dict(parent=PARENT.name, parent_sha256=PARENT_SHA,
                   candidate=TARGET.name, candidate_sha256=sha256(candidate.encode()).hexdigest(),
                   helper_sha256=HELPER_SHA, synthetic_result_sha256=PREFLIGHT_SHA,
                   changed_definition="_nonlinear_pair_batch", changed_ast_nodes=differences,
                   inverse_byte_exact=True, all_other_source_byte_exact=True,
                   same_imports_and_payloads=True, candidate_executed=False)
    return candidate, receipt


def main():
    if sys.argv[1:] != ["--build"]:
        print("Inert. Explicit --build required.")
        return
    require(not TARGET.exists() and not RECEIPT.exists(), "Refusing any existing build output")
    candidate, receipt = construction()
    # Generated, mechanical copy of the parent with one preflighted splice.
    with TARGET.open("x") as stream:
        stream.write(candidate)
    require(digest(TARGET) == receipt["candidate_sha256"], "Written candidate differs")
    receipt["builder_sha256"] = digest(__file__)
    with RECEIPT.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
