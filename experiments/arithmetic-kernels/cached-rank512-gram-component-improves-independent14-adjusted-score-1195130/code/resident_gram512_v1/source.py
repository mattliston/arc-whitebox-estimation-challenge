"""Mechanical, source-inverted resident-Gram candidate/null packaging."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "terminal_dead_consumers_v1/estimator.py"
PARENT_SHA = "07b71a03e13f8daed17e0ee529c4113e88cd89c2cceb29ea0e2f148084e74d30"


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
    helper_source = (HERE / "gram.py").read_text()
    helper_nodes = [node for node in ast.parse(helper_source).body if isinstance(node, ast.FunctionDef)]
    require(len(helper_nodes) == 1 and helper_nodes[0].name == "_resident_gram512", "One qualified helper")
    helper = ast.get_source_segment(helper_source, helper_nodes[0])
    changes = {
        "_cov_compact": ("_mm((wc * diagonal[:, None]).T, wc)",
                         "_resident_gram512((wc * diagonal[:, None]).T, wc)"),
        "_predict_binary": ("Cpre = _mm(w32.T, w32).astype(fnp.float64)",
                            "Cpre = _resident_gram512(w32.T, w32).astype(fnp.float64)"),
    }
    candidate = source
    changed_functions = {}
    for name, (old, new) in changes.items():
        changed_functions[name] = replace_once(functions[name], old, new)
        candidate = replace_once(candidate, functions[name], changed_functions[name])
    candidate += "\n\n" + helper + "\n"
    inverse = candidate[:-(len(helper)+3)]
    for name in changes:
        inverse = replace_once(inverse, changed_functions[name], functions[name])
    require(inverse == source, "Exact source inverse failed")
    after = ast.parse(candidate).body
    require(len(after) == len(nodes)+1 and after[-1].name == "_resident_gram512", "Unexpected AST addition")
    differences = [(a, b) for a, b in zip(nodes, after[:-1], strict=True)
                   if ast.dump(a) != ast.dump(b)]
    require(len(differences) == 2 and {a.name for a, _ in differences} == set(changes)
            and all(isinstance(a, ast.FunctionDef) and isinstance(b, ast.FunctionDef)
                    and a.name == b.name for a, b in differences), "Unexpected AST change")
    compile(candidate, str(HERE / "estimator.py"), "exec")
    return source, candidate, helper


def main():
    if sys.argv[1:] != ["--build"]:
        print("Inert. --build needs the fixed passing synthetic gate.")
        return
    paths = [HERE / name for name in ("estimator.py", "estimator_null.py", "build.json")]
    require(not any(path.exists() for path in paths), "Refusing existing build files")
    gate_path = HERE / "synthetic/result.json"
    gate = json.loads(gate_path.read_text())
    complete = json.loads((HERE / "synthetic/completion.json").read_text())
    require(gate["passed"] and gate["sources_unchanged"] and complete["passed"]
            and complete["result_sha256"] == digest(gate_path), "No passing fixed synthetic seal")
    for path, expected in gate["source_bindings"].items():
        require(digest(path) == expected, "Qualified source changed")
    source, candidate, helper = sources()
    for path, value in zip(paths[:2], (candidate, source), strict=True):
        with path.open("x") as stream:
            stream.write(value)
    record = dict(passed=True, executed=False, parent=str(PARENT), parent_sha256=PARENT_SHA,
                  candidate=str(paths[0]), candidate_sha256=digest(paths[0]),
                  null=str(paths[1]), null_sha256=digest(paths[1]),
                  source_sha256=digest(__file__), protocol_sha256=digest(HERE / "PROTOCOL.md"),
                  helper_sha256=digest(HERE / "gram.py"),
                  helper_function_sha256=sha256(helper.encode()).hexdigest(),
                  synthetic_result_sha256=digest(gate_path),
                  inverse_source_byte_exact=True, null_source_byte_exact=True,
                  all_other_definitions_byte_exact=True, imports_and_payloads_unchanged=True)
    require(record["null_sha256"] == PARENT_SHA, "Null source is not exact parent")
    with paths[2].open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
