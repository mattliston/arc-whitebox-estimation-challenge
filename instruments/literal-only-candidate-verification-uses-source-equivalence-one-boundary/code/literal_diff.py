"""TAG opus-release: is the difference between two candidate sources LITERAL-ONLY? (the proof half of the fast path, fast_path.py)

Both files are parsed (comments and layout do not count) and their syntax trees are walked side by side. The difference is
literal-only when the trees are the same node for node except in constants, and every differing constant is one of
  number : a float literal with another value (also int <-> float, as 1 against 0.8), a sign change included;
  data   : a packed array: a base64 string (or a bytes constant) of at least DATA_LEN characters on both sides, shown by
           length and hash;
  text   : a docstring.
A number that one file writes in place and the other reads through a module-level name bound once to a number literal
(NAME = 6.2176 ... x / NAME against x / 6: how the trainer exposes a new dial) is compared by value and listed as `named`;
rebinding through globals() is not seen.
Anything else fails the proof and sends the file to the full chain (stages A, C, D):
  code    : a statement, a name, an operator, a call or the number of items of a list differs;
  integer : an int literal with another int value (sizes, ranks, counts, indices, list entries of the prebind list);
  switch  : True / False / None;
  string  : a short string (an einsum signature, a type name, a mode, a key);
  type    : a constant that changed its kind (a string against a number).
Why floats and packed arrays only: an integer, a switch or a string can keep every count of the census and still change the
cost (a number type, a kernel choice), so those are never taken on trust. A float's value cannot change a number type or a
shape by itself, but it CAN steer operations: in the S38 file a K4L strength of exactly 1.0 skips a multiply, two equal F4
strengths per age take the scalar path, thresholds and kernel prices are floats too. So this proof alone is not the verdict:
the census of the two sessions decides (fast_path.py).
Every differing constant is listed with its place: enclosing definition, assignment target, key or index inside the value.
usage: literal_diff.py <qualified.py> <new.py> [--json OUT]        exit 0 = literal-only, 1 = not"""
import ast
import collections
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys

DATA_LEN = 200                                   # a string constant at least this long is a packed array, not a name
DIALS = ('number', 'data', 'text')               # the kinds a literal-only difference may consist of
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
BLOCK = (ast.stmt, ast.excepthandler, ast.match_case)


class Differs(Exception):
    """Two nodes differ in something other than a constant's value."""


class Fold(ast.NodeTransformer):
    """-0.028 parses as minus applied to 0.028: fold a signed number into one constant, so a sign change is a literal change."""

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        v = node.operand
        if isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(v, ast.Constant) and type(v.value) in (int, float, complex):
            return ast.copy_location(ast.Constant(-v.value if isinstance(node.op, ast.USub) else v.value), node)
        return node


def short(node, n=60):
    text = ast.unparse(node).split('\n')[0]
    return text if len(text) <= n else text[:n - 3] + '...'


def head(s):
    """A statement's label in the listing: a definition's name, an assignment's target, else its first line."""
    if isinstance(s, DEFS):
        return f"{'class' if isinstance(s, ast.ClassDef) else 'def'} {s.name}"
    if isinstance(s, ast.Assign):
        return short(s.targets[0])
    if isinstance(s, (ast.AnnAssign, ast.AugAssign)):
        return short(s.target)
    return f'<{short(s)}>'


def key(s):
    """What two statement lists of different length are aligned on: labels, with the numbers of other statements masked."""
    if isinstance(s, DEFS + (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return head(s)
    return type(s).__name__ + ' ' + re.sub(r'\d+\.?\d*(e[-+]?\d+)?', '#', short(s, 80))


def is_block(x):
    return isinstance(x, list) and bool(x) and isinstance(x[0], BLOCK)


def named(tree):
    """{name: constant} for every module-level NAME = <number literal> whose name is bound nowhere else in the file."""
    bound = collections.Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound[node.id] += 1
        elif isinstance(node, ast.arg):
            bound[node.arg] += 1
        elif isinstance(node, DEFS) or isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            bound[node.name] += 1
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.alias):
            bound[(node.asname or node.name).split('.')[0]] += 1
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound[node.rest] += 1
    return {s.targets[0].id: s.value for s in tree.body
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
            and isinstance(s.value, ast.Constant) and type(s.value.value) in (int, float) and bound[s.targets[0].id] == 1}


def kind(x, y):
    both = lambda *t: isinstance(x, t) and isinstance(y, t)
    if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None:
        return 'switch'
    if both(int):
        return 'integer'
    if both(int, float) or both(complex):
        return 'number'
    if type(x) is type(y) and both(str, bytes):
        packed = all(isinstance(v, bytes) or re.fullmatch(r'[A-Za-z0-9+/=_-]+', v) for v in (x, y))
        return 'data' if packed and min(len(x), len(y)) >= DATA_LEN else 'string'
    return 'type'


def show(v):
    if isinstance(v, (str, bytes)) and len(v) >= DATA_LEN:
        raw = v.encode() if isinstance(v, str) else v
        return f'{len(v):,} characters, sha {hashlib.sha256(raw).hexdigest()[:10]}'
    text = repr(v)
    return text if len(text) <= 48 else text[:45] + '...'


class Compare:
    """Side-by-side walk of two module trees. literals: the differing constants; code: everything else that differs."""

    def __init__(self, names=({}, {})):
        self.literals, self.code, self.named, self.names = [], [], [], names

    def note(self, scope, label, a, b, what):
        self.code.append(dict(where=' '.join(scope + (label,)), line_q=getattr(a, 'lineno', None),
                              line_n=getattr(b, 'lineno', None), what=what))

    def literal(self, scope, trail, a, b, kind_):
        self.literals.append(dict(kind=kind_, where=' '.join(scope + (trail,)), line_q=a.lineno, line_n=b.lineno,
                                  old=show(a.value), new=show(b.value)))

    def block(self, xs, ys, scope):
        """Two statement lists: paired in order when they hold the same statement types, else aligned on their labels."""
        if len(xs) == len(ys) and all(type(x) is type(y) for x, y in zip(xs, ys)):
            pairs = list(zip(xs, ys))
        else:
            pairs = []
            match = difflib.SequenceMatcher(None, [key(s) for s in xs], [key(s) for s in ys], autojunk=False)
            for tag, i1, i2, j1, j2 in match.get_opcodes():
                if tag == 'equal':
                    pairs += zip(xs[i1:i2], ys[j1:j2])
                    continue
                for side, part, file in ((0, xs[i1:i2], 'qualified'), (1, ys[j1:j2], 'new')):
                    for s in part:
                        if not scope and isinstance(s, ast.Assign) and self.names[side].get(head(s)) is s.value:
                            self.named.append(dict(name=head(s), value=s.value.value, file=file, line=s.lineno))
                        else:
                            self.note(scope, head(s), *((s, None) if side == 0 else (None, s)), f'only in the {file} file')
        for x, y in pairs:
            self.statement(x, y, scope)

    def statement(self, a, b, scope):
        if type(a) is not type(b):
            return self.note(scope, head(a), a, b, f'{type(a).__name__} against {type(b).__name__}')
        doc = lambda s: isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)
        if doc(a) and doc(b):
            if a.value.value != b.value.value:
                self.literal(scope, 'docstring', a.value, b.value, 'text')
            return
        label, found = head(a), []
        try:                                           # the statement's own line(s) first, its nested blocks after
            for name, x in ast.iter_fields(a):
                y = getattr(b, name, None)
                if not is_block(x) and not is_block(y):
                    self.field(a, name, x, y, label, found)
        except Differs as e:
            return self.note(scope, label, a, b, str(e))
        for trail, x, y in found:
            self.literal(scope, trail, x, y, kind(x.value, y.value))
        inner = scope + (label,) if isinstance(a, DEFS) else scope
        for name, x in ast.iter_fields(a):
            y = getattr(b, name, None)
            if is_block(x) or is_block(y):
                self.block(x or [], y or [], inner)

    def field(self, a, name, x, y, trail, found):
        if isinstance(x, list):
            if not isinstance(y, list) or len(x) != len(y):
                raise Differs(f'{type(a).__name__}.{name}: {len(x)} against {len(y) if isinstance(y, list) else 0} items')
            for i, (p, q) in enumerate(zip(x, y)):
                if isinstance(p, ast.AST):
                    self.expr(p, q, self.step(a, name, i, p, trail), found)
                elif p != q:
                    raise Differs(f'{type(a).__name__}.{name}: {p!r} against {q!r}')
        elif isinstance(x, ast.AST):
            self.expr(x, y, trail, found)
        elif x != y and name not in ('kind', 'type_comment'):
            raise Differs(f'{type(a).__name__}.{name}: {x!r} against {y!r}')

    @staticmethod
    def step(a, name, i, child, trail):
        """The place of a list item inside its statement: index in a list or tuple, argument of a call."""
        if isinstance(a, (ast.List, ast.Tuple, ast.Set)) and name == 'elts':
            return f'{trail}[{i}]'
        if isinstance(a, ast.Call) and name == 'args':
            return f'{trail} {short(a.func, 30)}(argument {i + 1})'
        if isinstance(a, ast.Call) and name == 'keywords':
            return f'{trail} {short(a.func, 30)}({child.arg}=)'
        return trail

    def expr(self, a, b, trail, found):
        for side, (x, y) in enumerate(((a, b), (b, a))):          # a named number against a number written in place
            if isinstance(x, ast.Name) and isinstance(y, ast.Constant) and x.id in self.names[side]:
                trail = f'{trail} (read through {x.id})'
                a, b = (self.names[0][x.id], b) if side == 0 else (a, self.names[1][x.id])
        if type(a) is not type(b):
            raise Differs(f'{type(a).__name__} against {type(b).__name__}')
        if isinstance(a, ast.Constant):
            if type(a.value) is not type(b.value) or a.value != b.value:
                found.append((trail, a, b))
        elif isinstance(a, ast.Dict):
            if len(a.keys) != len(b.keys):
                raise Differs(f'dict of {len(a.keys)} against {len(b.keys)} entries')
            for k, kb, v, vb in zip(a.keys, b.keys, a.values, b.values):
                if k is None or kb is None:            # ** unpacking inside the dict
                    if k is not kb:
                        raise Differs('dict unpacking')
                else:
                    self.expr(k, kb, trail + ' key', found)
                self.expr(v, vb, f'{trail}[{short(k, 30)}]' if k is not None else trail, found)
        else:
            for name, x in ast.iter_fields(a):
                self.field(a, name, x, getattr(b, name, None), trail, found)


def compare(qualified, new):
    """dict(literal_only, same_file, counts by kind, literals, code) for two source paths."""
    texts = [Path(p).read_text() for p in (qualified, new)]
    trees = [Fold().visit(ast.parse(t)) for t in texts]
    c = Compare([named(t) for t in trees])
    c.block(trees[0].body, trees[1].body, ())
    counts = {}
    for e in c.literals:
        counts[e['kind']] = counts.get(e['kind'], 0) + 1
    return dict(qualified=str(qualified), new=str(new), sha256=[hashlib.sha256(t.encode()).hexdigest() for t in texts],
                same_file=texts[0] == texts[1], literal_only=not c.code and all(k in DIALS for k in counts),
                counts=counts, literals=c.literals, named=c.named, code=c.code)


def lines(r, most=40):
    """The verdict line and the listing."""
    lit, code = r['literals'], r['code']
    counted = ', '.join(f'{n} {k}' for k, n in sorted(r['counts'].items())) or 'none'
    if r['literal_only']:
        what = ('the two files are byte-identical' if r['same_file'] else 'only comments or layout differ' if not lit
                else f"{len(lit)} constant{'s' if len(lit) > 1 else ''} differ{'' if len(lit) > 1 else 's'} ({counted}); no code differs")
        out = [f"[literal diff] LITERAL-ONLY: {what}"]
    else:
        bad = sum(n for k, n in r['counts'].items() if k not in DIALS)
        out = [f"[literal diff] NOT LITERAL-ONLY: {len(code)} statements differ in code; {bad} constants are not dials "
               f"(constants that differ: {counted}) -> full chain"]
    def at(e):
        where = [str(x) for x in dict.fromkeys((e['line_q'], e['line_n'])) if x is not None]
        return ('lines ' if len(where) > 1 else 'line ') + ' / '.join(where)
    for e in code[:most]:
        out.append(f"  code     {e['where']}: {e['what']} ({at(e)})")
    if len(code) > most:
        out.append(f'  ... and {len(code) - most} more statements')
    shown = sorted(lit, key=lambda e: e['kind'] in DIALS)[:most] if not r['literal_only'] else lit[:most]
    for e in shown:
        out.append(f"  {e['kind']:8s} {e['where']}: {e['old']} -> {e['new']} ({at(e)})")
    for e in r['named']:
        out.append(f"  named    {e['name']} = {e['value']!r}: only in the {e['file']} file, read as its value where it is used "
                   f"(line {e['line']})")
    if len(lit) > most:
        out.append(f'  ... and {len(lit) - most} more constants')
    return out


if __name__ == '__main__':
    result = compare(sys.argv[1], sys.argv[2])
    print('\n'.join(lines(result)))
    if '--json' in sys.argv:
        with open(sys.argv[sys.argv.index('--json') + 1], 'x') as stream:
            json.dump(result, stream, indent=1)
    sys.exit(0 if result['literal_only'] else 1)
