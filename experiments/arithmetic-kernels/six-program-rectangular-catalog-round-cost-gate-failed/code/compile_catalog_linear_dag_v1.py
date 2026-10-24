"""Compile whitelisted scalar AST data into three authored linear DAGs."""
import ast
from fractions import Fraction
from executable_linear_dag_v1 import Builder, Wire


def compile_program(program, shape, rank):
    m, k, n = shape
    builders = dict(A=Builder(m*k), B=Builder(k*n), C=Builder(rank))
    values, left, right = {}, [], []

    def product(a, sa, b, sb):
        assert (sa, sb) == ('A', 'B')
        left.append(a)
        right.append(b)
        assert len(left) <= rank
        return Wire(len(left)-1), 'C'

    def expr(node):
        if isinstance(node, ast.Constant):
            assert type(node.value) is int
            return Fraction(node.value), None
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Subscript):
            assert isinstance(node.value, ast.Name)
            stage = node.value.id
            assert stage in ('A', 'B', 'C')
            assert isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2
            assert all(isinstance(x, ast.Constant) and type(x.value) is int for x in node.slice.elts)
            i, j = [x.value-1 for x in node.slice.elts]
            if stage == 'C':
                return values[f'C[{i+1},{j+1}]']
            rows, cols = (m, k) if stage == 'A' else (k, n)
            assert 0 <= i < rows and 0 <= j < cols
            return Wire(i*cols+j), stage
        if isinstance(node, ast.UnaryOp):
            assert isinstance(node.op, ast.USub)
            value, stage = expr(node.operand)
            return (builders[stage].times(value, -1) if stage else -value), stage
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id == 'mm'
            assert len(node.args) == 2 and not node.keywords
            a, sa = expr(node.args[0])
            b, sb = expr(node.args[1])
            return product(a, sa, b, sb)
        if isinstance(node, ast.BinOp):
            a, sa = expr(node.left)
            b, sb = expr(node.right)
            if isinstance(node.op, ast.Mult) and (sa, sb) == ('A', 'B'):
                return product(a, sa, b, sb)
            stage = sa or sb
            assert sa is None or sb is None or sa == sb
            if isinstance(node.op, (ast.Add, ast.Sub)):
                assert sa == sb
                sign = 1 if isinstance(node.op, ast.Add) else -1
                return (builders[stage].add(a, builders[stage].times(b, sign))
                        if stage else a+sign*b), stage
            if isinstance(node.op, ast.Mult):
                assert sa is None or sb is None
                if sa:
                    return builders[sa].times(a, b), sa
                if sb:
                    return builders[sb].times(b, a), sb
                return a*b, None
            if isinstance(node.op, ast.Div):
                assert sb is None and b != 0
                return (builders[sa].times(a, 1/b) if sa else a/b), sa
        raise TypeError(type(node))

    for name, node in program:
        values[name] = expr(node)
    assert len(left) == len(right) == rank
    outputs = []
    for i in range(1, m+1):
        for j in range(1, n+1):
            wire, stage = values[f'C[{i},{j}]']
            assert stage == 'C'
            outputs.append(wire)
    return builders['A'].circuit(left), builders['B'].circuit(right), builders['C'].circuit(outputs)
