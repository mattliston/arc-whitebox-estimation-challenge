#!/usr/bin/env python3
"""Exact, target-free verifier for the B1B joint-K5 contraction audit."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent

BOUND = {
    P2 / "p2r0f3b1a_reverse_hermite_v1" / "RESULT.json":
        "785d789a9ce588989f6ab070d84c6dfdf8ec4faaacc3eabf1bbe791fb49b70e6",
    P2 / "p2r0f3b1a_reverse_hermite_v1" / "TERMINAL_RECEIPT.json":
        "cc5c5b094c2594297ff311aac5f366362c9d10a25833d53f54baf4f51910aa68",
    P2 / "p2r0f3b1a_reverse_hermite_independent_audit_v1" /
        "AUDIT_RECEIPT.json":
        "450378882b0fe75964cc1378fdd17b52d320ef5f80aba3ddd9bf1bb83cb761cb",
    P2 / "public_exact_ledger_v1" / "ledger_rig.py":
        "c74c92862a3afad7e44a077c80f87fdd09c66be4f02718db24a21ea7b5f4a581",
    P2 / "public_exact_ledger_v1" / "LEDGER_VERIFY_RESULTS.json":
        "f14b4964ab6d7da4917d4c4a77c08099477a840849819cc61ee46da0d842c938",
    P2 / "p2r0f3b2b_partition_complete_tt5_design_v1" /
        "TERMINAL_DESIGN_AUDIT.md":
        "ddfdfc64dec12efb5dbc7b96384e6087bc0eaad19a28a44e9d2eccd2839c16e6",
    P2 / "p2r0f3b2b2_independent_gaussian_linked_cluster_grade3_v1" /
        "CONSTRAINT_RECEIPT.md":
        "d6275c2ace74294c4d9c6fcc6b5126dbbb9725e4dac51aeca1affba0e98c7cba",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_rank(rows):
    a = [[F(x) for x in row] for row in rows]
    m, n = len(a), len(a[0])
    rank = 0
    for col in range(n):
        pivot = next((row for row in range(rank, m) if a[row][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][col]
        a[rank] = [value / scale for value in a[rank]]
        for row in range(m):
            if row != rank and a[row][col]:
                scale = a[row][col]
                a[row] = [left - scale * right
                          for left, right in zip(a[row], a[rank])]
        rank += 1
        if rank == m:
            break
    return rank


def make_sectors(n):
    diag = {i: F(2 * i + 1, 7) for i in range(n)}
    b41 = {(i, j): F(3 * i + 5 * j + 2, 11)
           for i in range(n) for j in range(n) if i != j}
    b32 = {(i, j): F(7 * i - 2 * j + 3, 13)
           for i in range(n) for j in range(n) if i != j}
    b311 = {}
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for j, k in itertools.combinations(others, 2):
            b311[(i, j, k)] = F(11 * i + 3 * j - 5 * k + 7, 17)
    b221 = {}
    for i, j in itertools.combinations(range(n), 2):
        for k in range(n):
            if k not in (i, j):
                b221[(i, j, k)] = F(2 * i - 7 * j + 13 * k + 5, 19)
    return diag, b41, b32, b311, b221


def dense_value(index, sectors):
    diag, b41, b32, b311, b221 = sectors
    counts = Counter(index)
    pattern = sorted(counts.values(), reverse=True)
    if pattern == [5]:
        return diag[index[0]]
    if pattern == [4, 1]:
        i = next(key for key, value in counts.items() if value == 4)
        j = next(key for key, value in counts.items() if value == 1)
        return b41[(i, j)]
    if pattern == [3, 2]:
        i = next(key for key, value in counts.items() if value == 3)
        j = next(key for key, value in counts.items() if value == 2)
        return b32[(i, j)]
    if pattern == [3, 1, 1]:
        i = next(key for key, value in counts.items() if value == 3)
        j, k = sorted(key for key, value in counts.items() if value == 1)
        return b311[(i, j, k)]
    if pattern == [2, 2, 1]:
        i, j = sorted(key for key, value in counts.items() if value == 2)
        k = next(key for key, value in counts.items() if value == 1)
        return b221[(i, j, k)]
    return F(0)


def dense_push_diagonal(n, sectors, w):
    out = []
    for a in range(n):
        value = F(0)
        for index in itertools.product(range(n), repeat=5):
            term = dense_value(index, sectors)
            for i in index:
                term *= w[i][a]
            value += term
        out.append(value)
    return out


def canonical_push_diagonal(n, sectors, w):
    diag, b41, b32, b311, b221 = sectors
    out = []
    for a in range(n):
        value = sum(diag[i] * w[i][a] ** 5 for i in range(n))
        value += 5 * sum(b * w[i][a] ** 4 * w[j][a]
                         for (i, j), b in b41.items())
        value += 10 * sum(b * w[i][a] ** 3 * w[j][a] ** 2
                          for (i, j), b in b32.items())
        value += 20 * sum(b * w[i][a] ** 3 * w[j][a] * w[k][a]
                          for (i, j, k), b in b311.items())
        value += 30 * sum(b * w[i][a] ** 2 * w[j][a] ** 2 * w[k][a]
                          for (i, j, k), b in b221.items())
        out.append(value)
    return out


def endpoint_from_diagonal(diagonal, h):
    outputs = len(h[0])
    return [sum(diagonal[a] * h[a][o] for a in range(len(diagonal)))
            for o in range(outputs)]


def direct_endpoint(n, sectors, w, h):
    diag, b41, b32, b311, b221 = sectors
    outputs = len(h[0])
    out = [F(0) for _ in range(outputs)]

    def add_sector(entries, multiplicity, monomial):
        for key, b in entries.items():
            for o in range(outputs):
                action = sum(h[a][o] * monomial(key, a)
                             for a in range(n))
                out[o] += multiplicity * b * action

    add_sector(diag, 1, lambda i, a: w[i][a] ** 5)
    add_sector(b41, 5,
               lambda ij, a: w[ij[0]][a] ** 4 * w[ij[1]][a])
    add_sector(b32, 10,
               lambda ij, a: w[ij[0]][a] ** 3 * w[ij[1]][a] ** 2)
    add_sector(b311, 20,
               lambda ijk, a: w[ijk[0]][a] ** 3 * w[ijk[1]][a]
               * w[ijk[2]][a])
    add_sector(b221, 30,
               lambda ijk, a: w[ijk[0]][a] ** 2 * w[ijk[1]][a] ** 2
               * w[ijk[2]][a])
    return out


def exact_duhamel_check():
    # A separate finite-state identity: forward source insertion equals the
    # reverse adjoint sum for three sources and two endpoint coordinates.
    dim, outputs, depth = 5, 2, 3
    js = [[[F((i + 2) * (j + 3) + 5 * layer + (i == j), 23)
            for j in range(dim)] for i in range(dim)]
          for layer in range(depth)]
    sources = [[F((layer + 2) * (i + 1) - 3, 29)
                for i in range(dim)] for layer in range(depth)]
    c = [[F((i + 1) * (o + 2) - 1, 31)
          for o in range(outputs)] for i in range(dim)]

    state = [F(0) for _ in range(dim)]
    for layer in range(depth):
        state = [sum(js[layer][i][j] * state[j] for j in range(dim))
                 + sources[layer][i] for i in range(dim)]
    forward = [sum(c[i][o] * state[i] for i in range(dim))
               for o in range(outputs)]

    reverse = [F(0) for _ in range(outputs)]
    lambdas = [[c[i][o] for i in range(dim)] for o in range(outputs)]
    for layer in reversed(range(depth)):
        for o in range(outputs):
            reverse[o] += sum(lambdas[o][i] * sources[layer][i]
                              for i in range(dim))
            lambdas[o] = [sum(js[layer][i][j] * lambdas[o][i]
                              for i in range(dim))
                          for j in range(dim)]
    assert forward == reverse


def elimination_treewidth(vertices, edges):
    best = len(vertices)
    for order in itertools.permutations(vertices):
        graph = {v: set() for v in vertices}
        for left, right in edges:
            graph[left].add(right)
            graph[right].add(left)
        width = 0
        for vertex in order:
            neighbors = list(graph[vertex])
            width = max(width, len(neighbors))
            for left, right in itertools.combinations(neighbors, 2):
                graph[left].add(right)
                graph[right].add(left)
            for neighbor in neighbors:
                graph[neighbor].discard(vertex)
            graph[vertex].clear()
        best = min(best, width)
    return best


def main():
    for path, expected in BOUND.items():
        assert sha256(path) == expected, path

    # The ledger explicitly carries only the support-at-most-two K5 shell.
    ledger = (P2 / "public_exact_ledger_v1" / "ledger_rig.py").read_text()
    for field in ("k5d", "K41", "K32"):
        assert field in ledger
    for absent in ("K311", "K221", "K2111", "K11111"):
        assert absent not in ledger

    n = 4
    sectors = make_sectors(n)
    w = [[F((i + 1) ** a + 2 * i + a + (i == a), 7)
          for a in range(n)] for i in range(n)]
    h = [[F((a + 2) * (o + 3) - 1, 11) for o in range(3)]
         for a in range(n)]
    dense = dense_push_diagonal(n, sectors, w)
    canonical = canonical_push_diagonal(n, sectors, w)
    assert dense == canonical
    forward_endpoint = endpoint_from_diagonal(dense, h)
    reverse_endpoint = direct_endpoint(n, sectors, w, h)
    assert forward_endpoint == reverse_endpoint

    # A pure [311] fixture is invisible to B1A's diagonal K5 and to K22, but
    # its joint fifth-order action is exactly nonzero.
    zero_diag = {i: F(0) for i in range(n)}
    empty = {}
    pure311 = {(0, 1, 2): F(5, 17)}
    joint_only = (zero_diag, empty, empty, pure311, empty)
    joint_endpoint = direct_endpoint(n, joint_only, w, h)
    assert any(joint_endpoint)
    assert not any(zero_diag.values())
    # Fifth-degree affine homogeneity distinguishes this formal source from
    # every order-four K22 contraction.
    w2 = [[2 * value for value in row] for row in w]
    joint_endpoint_2 = direct_endpoint(n, joint_only, w2, h)
    assert joint_endpoint_2 == [32 * value for value in joint_endpoint]

    exact_duhamel_check()

    # Exact generic-rank witness for the immediate [311] action tensor.
    vandermonde = [[F((i + 1) ** a) for a in range(n)] for i in range(n)]
    g = [F(a + 1) for a in range(n)]
    unfolding = [
        [sum(g[a] * vandermonde[i][a] ** 3 * vandermonde[j][a]
             * vandermonde[k][a] for a in range(n))
         for j in range(n) for k in range(n)]
        for i in range(n)
    ]
    assert matrix_rank(unfolding) == n

    # The dense trivariate/action factor graph is K4; bivariate is K3.
    tri_vertices = ("i", "j", "k", "a")
    tri_edges = tuple(itertools.combinations(tri_vertices, 2))
    assert elimination_treewidth(tri_vertices, tri_edges) == 3
    bi_vertices = ("i", "j", "a")
    bi_edges = tuple(itertools.combinations(bi_vertices, 2))
    assert elimination_treewidth(bi_vertices, bi_edges) == 2

    width, layers, rank = 1024, 15, 1024
    n311 = width * (width - 1) * (width - 2) // 2
    n221 = n311
    n3 = n311 + n221
    n2 = 2 * width * (width - 1)
    board = 2**41
    host = 1_752_673_252_727
    margin = board - host
    generic_triple_cost = 2 * layers * rank * n3
    bivariate_contraction_cost = 2 * layers * rank * n2
    assert n311 == n221 == 535_299_072
    assert n3 == 1_070_598_144
    assert generic_triple_cost == 32_888_774_983_680
    assert generic_triple_cost > board
    assert board // (2 * layers * n3) == 68
    assert margin // (2 * layers * n3) == 13
    assert bivariate_contraction_cost == 64_361_594_880
    assert 4 * n3 == 4_282_392_576
    assert 8 * n3 == 8_564_785_152
    assert 2**33 - 8 * n3 == 25_149_440

    receipt = json.loads((HERE / "AUDIT_RECEIPT.json").read_text())
    assert receipt["status"] == (
        "CLOSE_BACKWARD_SPELLING__EXTERNAL_FUSED_K5_3D_ACTION_ONLY"
    )
    assert receipt["exact_checks"]["dense_vs_canonical_k5"] == "PASS_EXACT"
    assert receipt["exact_checks"]["forward_vs_reverse"] == "PASS_EXACT"
    assert receipt["cost"]["generic_rank1024_15layer_flops"] == (
        generic_triple_cost
    )
    assert receipt["access"]["targets_opened"] == 0
    assert receipt["access"]["graph_edits"] == 0
    print("PASS_B1B_JOINT_K5_OUTPUT_CONTRACTION_AUDIT")


if __name__ == "__main__":
    main()
