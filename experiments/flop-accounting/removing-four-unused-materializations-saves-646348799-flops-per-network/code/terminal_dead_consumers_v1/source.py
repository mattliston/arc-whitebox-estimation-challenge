"""Inert mechanical terminal-only transformation of a pinned Phase2 source."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "candidate_response_pair_polynomial_f32_v1.py"
PARENT_SHA = "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new):
    require(text.count(old) == 1, "Ambiguous source splice")
    return text.replace(old, new, 1)


def sources():
    require(digest(PARENT) == PARENT_SHA, "Frozen parent changed")
    source = PARENT.read_text()
    nodes = ast.parse(source).body
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in nodes if isinstance(node, ast.FunctionDef)}
    pair = functions["_nonlinear_pair_batch"]
    terminal = replace_once(pair, "def _nonlinear_pair_batch(", "def _nonlinear_terminal_live(")
    terminal = replace_once(terminal,
        "aa=fnp.stack(A,axis=0);bb=fnp.stack(B,axis=0)\n"
        "    F=fnp.stack([aa,bb,bb],axis=0);G=fnp.stack([aa,aa,bb],axis=0)\n"
        "    ans=fnp.zeros((3,n,n),dtype=fnp.float32)",
        "aa=fnp.stack(A,axis=0)\n"
        "    F=fnp.stack([aa],axis=0);G=fnp.stack([aa],axis=0)\n"
        "    ans=fnp.zeros((1,n,n),dtype=fnp.float32)")
    start = terminal.index("    pk21=ans[1]*fnp.outer(v,s)")
    stop = terminal.index("    aux = dict(", start)
    terminal = terminal[:start] + "    fnp.fill_diagonal(cov, var)\n" + terminal[stop:]
    terminal = replace_once(terminal, "return (m, cov, post21, radial, aux)",
                            "return (m, cov, None, 0.0, aux)")
    old_predictor = functions["_predict_binary"]  # Last/effective definition only.
    predictor = replace_once(old_predictor,
        "pair_map = _nonlinear_first_gaussian if l == 0 else nonlinear",
        "pair_map = _nonlinear_first_gaussian if l == 0 else (_nonlinear_terminal_live if l == depth - 1 else nonlinear)")
    predictor = replace_once(predictor,
        "        if len(lam):\n            source_sum = omitted_radial_source(",
        "        if len(lam) and l < depth - 1:\n            source_sum = omitted_radial_source(")
    require(source.count(old_predictor) == 1, "Ambiguous effective predictor")
    candidate = replace_once(source, old_predictor, terminal + "\n\n\n" + predictor)
    require(candidate.replace(terminal + "\n\n\n" + predictor, old_predictor, 1) == source,
            "Inverse source recovery failed")
    after = ast.parse(candidate).body
    stripped = [node for node in after if not (isinstance(node, ast.FunctionDef)
                                              and node.name == "_nonlinear_terminal_live")]
    require(len(stripped) == len(nodes) and len(after) == len(nodes) + 1, "Unexpected AST structure")
    differences = [(a, b) for a, b in zip(nodes, stripped, strict=True) if ast.dump(a) != ast.dump(b)]
    require(len(differences) == 1 and all(isinstance(x, ast.FunctionDef)
            and x.name == "_predict_binary" for x in differences[0]), "Unexpected changed definition")
    compile(candidate, str(HERE / "estimator.py"), "exec")
    return functions, terminal, candidate


def main():
    if sys.argv[1:] != ["--build"]:
        print("Inert. --build requires a completed synthetic gate.")
        return
    target, receipt = HERE / "estimator.py", HERE / "build.json"
    require(not target.exists() and not receipt.exists(), "Refusing existing build outputs")
    gate_path = HERE / "synthetic/result.json"
    gate = json.loads(gate_path.read_text())
    require(gate["passed"] and gate["sources_unchanged"], "Synthetic gate did not pass")
    require(gate["source_sha256"] == digest(__file__) and gate["parent_sha256"] == PARENT_SHA
            and gate["protocol_sha256"] == digest(HERE / "PROTOCOL.md")
            and gate["checker_sha256"] == digest(HERE / "check_synthetic.py"), "Gate source mismatch")
    _, terminal, candidate = sources()
    require(gate["terminal_function_sha256"] == sha256(terminal.encode()).hexdigest(), "Different qualified function")
    with target.open("x") as stream:
        stream.write(candidate)
    record = dict(passed=True, candidate=str(target), candidate_sha256=digest(target),
                  parent=str(PARENT), parent_sha256=PARENT_SHA, source_sha256=digest(__file__),
                  protocol_sha256=digest(HERE / "PROTOCOL.md"),
                  synthetic_result_sha256=digest(gate_path),
                  terminal_function_sha256=sha256(terminal.encode()).hexdigest(),
                  inverse_source_byte_exact=True, all_other_definitions_byte_exact=True,
                  imports_and_learned_payloads_unchanged=True, executed=False)
    with receipt.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
