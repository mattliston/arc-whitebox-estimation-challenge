HOLD_CLASS = """

class _E2Hold(list):
    \"\"\"E2: keeps flopscope temporaries alive until the list is dropped, so the client frees them in one hidden round trip.\"\"\"
    __slots__ = ()

    def __call__(self, x):
        self.append(x)
        return x
"""
"""TAG fable-engineer2 (E2-H2): source-preserving AST pass that makes every temporary-producing operand of a statement live until its
function returns (or its loop iteration ends), so the flopscope client releases a whole call's temporaries in ONE hidden `free`
round trip instead of one per statement. Only whole simple statements whose AST changes are re-emitted (ast.unparse, same
operations in the same order); every other line, including the porter's anchor lines, stays byte-identical."""
import ast

PY_BUILTINS = {'len', 'int', 'float', 'str', 'bool', 'min', 'max', 'range', 'isinstance', 'tuple', 'list', 'dict', 'zip', 'enumerate',
               'any', 'all', 'sorted', 'print', 'type', 'id', 'getattr', 'hasattr', 'repr', 'set', 'divmod', 'round'}
DICT_METHODS = {'get', 'append', 'update', 'pop', 'clear', 'items', 'keys', 'values', 'setdefault', 'extend', 'copy_',
                'startswith', 'endswith', 'join', 'format'}
HOLD = '_e2_H'


def _is_temp_producer(node):
    if isinstance(node, (ast.BinOp, ast.UnaryOp)):
        return True
    if isinstance(node, ast.Attribute):
        return node.attr in ('T',)
    if isinstance(node, ast.Subscript):
        s = node.slice
        return not (isinstance(s, ast.Constant) and isinstance(s.value, (str, int)) and isinstance(node.value, ast.Name)
                    and node.value.id in ('aux', 'st', 'L', 'P', 't', 'k', 'c', 'state', 'context', 'group', 'wp', 'pr', 'gk',
                                          '_E2_LAYER', 'scope', 'grant', 'cache', 'newborn', 'Q', 'ns'))
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id not in PY_BUILTINS and f.id != HOLD
        if isinstance(f, ast.Attribute):
            return f.attr not in DICT_METHODS and not (isinstance(f.value, ast.Name) and f.value.id in (HOLD, '_e2_hold'))
    return False


class _Wrap(ast.NodeTransformer):
    """Wrap temp-producing operands (never the statement's own result) in _e2_H(...)."""
    def __init__(self):
        self.changed = False

    def _w(self, node):
        node = self.visit(node)
        if _is_temp_producer(node):
            self.changed = True
            return ast.Call(func=ast.Name(id=HOLD, ctx=ast.Load()), args=[node], keywords=[])
        return node

    def visit_BinOp(self, node):
        node.left = self._w(node.left); node.right = self._w(node.right); return node

    def visit_UnaryOp(self, node):
        node.operand = self._w(node.operand); return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            node.func.value = self._w(node.func.value)
        node.args = [a if isinstance(a, ast.Starred) else self._w(a) for a in node.args]
        for kw in node.keywords:
            if kw.arg != 'out':
                kw.value = self._w(kw.value)
        return node

    def visit_Subscript(self, node):
        node.value = self._w(node.value); return node

    def visit_Attribute(self, node):
        node.value = self._w(node.value); return node

    def visit_Tuple(self, node):
        node.elts = [self._w(e) for e in node.elts]; return node

    def visit_List(self, node):
        node.elts = [self._w(e) for e in node.elts]; return node

    def visit_Compare(self, node):
        node.left = self._w(node.left); node.comparators = [self._w(c) for c in node.comparators]; return node

    def visit_IfExp(self, node):
        node.test = self.visit(node.test); node.body = self.visit(node.body); node.orelse = self.visit(node.orelse); return node

    def visit_Lambda(self, node):
        return node

    def visit_GeneratorExp(self, node):
        return node

    def visit_ListComp(self, node):
        return node


def _stmt_targets(fn):
    """Simple statements to rewrite (skip nested defs/classes, keep compound statement headers)."""
    out = []
    def walk(body):
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(st, (ast.Assign, ast.AugAssign, ast.Expr, ast.Return, ast.AnnAssign)):
                out.append(st)
            for field in ('body', 'orelse', 'finalbody'):
                sub = getattr(st, field, None)
                if isinstance(sub, list) and st is not fn:
                    walk(sub)
            if isinstance(st, ast.Try):
                for h in st.handlers:
                    walk(h.body)
    walk(fn.body)
    return out


def transform_function(src, fn_start, skip_lines=(), loop_holds=False):
    """Rewrite the LAST function whose source starts with fn_start. skip_lines: exact line texts left untouched."""
    i = src.rindex(fn_start)
    tree = ast.parse(src)
    line_no = src.count('\n', 0, i) + 1
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.lineno == line_no]
    assert len(fn) == 1, fn_start
    fn = fn[0]
    lines = src.split('\n')
    edits = []   # (first_line_idx, last_line_idx, new_text)
    for st in _stmt_targets(fn):
        seg = lines[st.lineno - 1:st.end_lineno]
        if any(s in skip_lines for s in seg) or any('_e2_hold.append(' in s for s in seg):
            continue
        if st.lineno == st.end_lineno and ';' in seg[0]:
            continue   # multi-statement lines are left alone
        w = _Wrap()
        if isinstance(st, ast.Assign):
            st.value = w.visit(st.value)
            if isinstance(st.value, ast.Tuple):
                pass
        elif isinstance(st, ast.AugAssign):
            st.value = w._w(st.value)
        elif isinstance(st, ast.Expr):
            st.value = w.visit(st.value)
        elif isinstance(st, ast.Return):
            if st.value is not None:
                st.value = w.visit(st.value)
        if not w.changed:
            continue
        indent = seg[0][:len(seg[0]) - len(seg[0].lstrip())]
        edits.append((st.lineno - 1, st.end_lineno - 1, indent + ast.unparse(st)))
    body_indent = None
    for idx in range(fn.body[0].lineno - 1, len(lines)):
        if lines[idx].strip():
            body_indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip())]; break
    # hold list creation: at the top of the function (after a docstring), and (optionally) at the top of every for-loop body
    k = 0
    while k < len(fn.body) and ((isinstance(fn.body[k], ast.Expr) and isinstance(fn.body[k].value, ast.Constant)
                                 and isinstance(fn.body[k].value.value, str)) or isinstance(fn.body[k], (ast.Global, ast.Nonlocal))):
        k += 1
    top = fn.body[k - 1].end_lineno if k else fn.body[0].lineno - 1
    inserts = [(top, body_indent + f"{HOLD} = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)")]
    if loop_holds:
        for n in ast.walk(fn):
            if isinstance(n, ast.For):
                b0 = n.body[0]
                ind = lines[b0.lineno - 1][:len(lines[b0.lineno - 1]) - len(lines[b0.lineno - 1].lstrip())]
                inserts.append((b0.lineno - 1, ind + f"{HOLD} = _E2Hold()   # E2: this iteration's temporaries, released together at the next iteration"))
    # apply edits bottom-up
    for a, b, text in sorted(edits, reverse=True):
        lines[a:b + 1] = [text]
    # recompute insert positions: inserts were computed on original numbering; apply them after mapping
    # (simplest: apply inserts first on a fresh copy, then edits) -> redo in that order
    return lines, edits, inserts


def apply(src, fn_start, skip_lines=(), loop_holds=False):
    i = src.rindex(fn_start)
    tree = ast.parse(src)
    line_no = src.count('\n', 0, i) + 1
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.lineno == line_no]
    assert len(fn) == 1, fn_start
    _, edits, inserts = transform_function(src, fn_start, skip_lines, loop_holds)
    lines = src.split('\n')
    ops = [(a, b, 'edit', t) for a, b, t in edits] + [(p, p - 1, 'insert', t) for p, t in inserts]
    # apply from the bottom so earlier indices stay valid; an insert at p goes BEFORE line p
    for a, b, kind, t in sorted(ops, key=lambda o: (o[0], 1 if o[2] == 'edit' else 0), reverse=True):   # same index: edit first, then insert before it
        if kind == 'edit':
            lines[a:b + 1] = [t]
        else:
            lines[a:a] = [t]
    return '\n'.join(lines), len(edits)
