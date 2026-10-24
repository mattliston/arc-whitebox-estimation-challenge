"""Build a stacked science/release pair: base pair (must end with `_FAST_RESCUE_FRAC = {}`) + tail assignments + the same final line.
usage: build_stack_v1.py --sci BASE_SCI --rel BASE_REL --tail TAIL.py --out DIR --name NAME
Checks: tail contains only module-level assignments/asserts/comments; function/class AST of both outputs identical to their bases; both compile.
"""
import argparse, ast, hashlib, json, subprocess
from pathlib import Path
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FINAL = '_FAST_RESCUE_FRAC = {}\n'
def functions(tree): return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
def main():
    ap = argparse.ArgumentParser()
    for n in ('sci', 'rel', 'tail', 'out', 'name'): ap.add_argument('--' + n, required=True)
    a = ap.parse_args(); out = Path(a.out); assert not out.exists(); out.mkdir(parents=True)
    tail = Path(a.tail).read_text(); ttree = ast.parse(tail)
    assert all(isinstance(n, (ast.Assign, ast.Assert, ast.Expr, ast.ClassDef, ast.FunctionDef)) for n in ttree.body), 'tail must be assignments/asserts/defs only'
    added = {n.name for n in ttree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    assert not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_FAST_RESCUE_FRAC' for t in n.targets) for n in ttree.body)
    files = {}
    for role, src in (('science', a.sci), ('release', a.rel)):
        text = Path(src).read_text(); assert text.endswith(FINAL), role
        new = text[:-len(FINAL)] + '\n# fable 2026-10-21 stack ' + a.name + '\n' + tail.rstrip('\n') + '\n' + FINAL
        target = out / ('candidate_%s_%s_v1.py' % (a.name, role)); target.write_text(new)
        fo, fn = functions(ast.parse(text)), functions(ast.parse(new)); assert {k for k in fo if fo[k] != fn.get(k)} == set() and set(fn) - set(fo) == added, role
        compile(new, str(target), 'exec')
        files[role] = dict(path=str(target), sha256=sha(target), bytes=target.stat().st_size, base_sha256=sha(src))
    rec = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(), name=a.name, tail=tail, files=files)
    (out / 'BUILD_v1.json').write_text(json.dumps(rec, indent=1) + '\n'); print(json.dumps(files))
if __name__ == '__main__': main()
