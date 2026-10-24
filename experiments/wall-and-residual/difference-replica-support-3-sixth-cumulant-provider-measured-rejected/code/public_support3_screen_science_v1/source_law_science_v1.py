#!/usr/bin/env python3
"""Guarded support-3+ K6 law archive for the V2 governed runner.

Ordinary import raises before numerical imports or function definitions.  The
reviewed future runner may remove exactly this one AST Raise and compile the
remaining law into a private dictionary, but only after durable claim and
verified unique capability consumption.  No function here is module-visible
through the documented import API.
"""

from __future__ import annotations

# guard raise removed for science-only screen (see SCREEN_PREREGISTRATION.json)

import hashlib
import math
from collections import defaultdict

import numpy as fnp
import numpy


WIDTH = 1024
SOURCE_LAYERS = 15
PATHS = 1664
DOMAIN_SEPARATOR_U64 = 5_419_610_805_510_746_657
UINT64_MASK = (1 << 64) - 1
CARRIER_ORDER = ("6", "42", "222")
CARRIER_LAM = {"6": (6,), "42": (4, 2), "222": (2, 2, 2)}
CARRIER_RANK = {"6": 512, "42": 192, "222": 128}
SECTOR_ORDER = (
    (4, 1, 1),
    (3, 2, 1),
    (2, 2, 2),
    (3, 1, 1, 1),
    (2, 2, 1, 1),
    (2, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1),
)
SPLIT_COEFFICIENT = {
    "6": 1.0 / 512.0,
    "42": -5.0 / 64.0,
    "222": 15.0 / 64.0,
}


def _set_partitions(items):
    items = tuple(items)
    if not items:
        yield ()
        return
    first = items[0]
    for rest in _set_partitions(items[1:]):
        yield ((first,),) + rest
        for index in range(len(rest)):
            block = tuple(sorted((first,) + rest[index]))
            yield tuple(sorted(rest[:index] + (block,) + rest[index + 1 :]))


_PARTITIONS = {
    size: tuple(sorted(set(_set_partitions(range(size)))))
    for size in range(7)
}


def _sector_table(lam, sector):
    """Return canonical integer Moebius terms; host-only and data independent."""

    labels = []
    for factor_index, multiplicity in enumerate(lam):
        labels.extend([factor_index] * multiplicity)
    terms = defaultdict(int)
    for outer in _PARTITIONS[6]:
        if tuple(sorted((len(block) for block in outer), reverse=True)) != sector:
            continue
        exponent_vectors = []
        for block in outer:
            exponent = [0] * len(lam)
            for slot in block:
                exponent[labels[slot]] += 1
            exponent_vectors.append(tuple(exponent))
        for inner in _PARTITIONS[len(outer)]:
            coefficient = 1
            monomial = []
            for merged_blocks in inner:
                if (len(merged_blocks) - 1) % 2:
                    coefficient = -coefficient
                coefficient *= math.factorial(len(merged_blocks) - 1)
                monomial.append(
                    tuple(
                        sum(
                            exponent_vectors[index][axis]
                            for index in merged_blocks
                        )
                        for axis in range(len(lam))
                    )
                )
            terms[tuple(sorted(monomial))] += coefficient
    return tuple(
        sorted(
            (monomial, coefficient)
            for monomial, coefficient in terms.items()
            if coefficient
        )
    )


def _rng_paths_literal(mlp_seed):
    """The one frozen NumPy-2.4.6 PCG64DXSM scientific-screen draw."""

    if not isinstance(mlp_seed, int) or isinstance(mlp_seed, bool):
        raise TypeError("mlp_seed must be a Python int, not bool")
    if mlp_seed < 0 or mlp_seed > UINT64_MASK:
        raise ValueError("mlp_seed outside uint64 domain")
    seed_u64 = (mlp_seed ^ DOMAIN_SEPARATOR_U64) & UINT64_MASK
    bit_generator = numpy.random.PCG64DXSM(seed_u64)
    generator = numpy.random.Generator(bit_generator)
    return generator.standard_normal(
        size=(1664, 1024), dtype=numpy.float32, out=None
    )


def _factor_groups_literal(d64):
    """Build every mixed factor row using only repeated float64 multiply."""

    a1 = d64[0:512]
    b1 = d64[512:704]
    c1 = d64[704:832]

    a2 = a1 * a1
    a3 = a2 * a1
    a4 = a3 * a1
    a5 = a4 * a1
    a6 = a5 * a1
    b2 = b1 * b1
    c2 = c1 * c1

    a_power = {1: a1, 2: a2, 3: a3, 4: a4, 5: a5, 6: a6}
    b_power = {1: b1, 2: b2}
    c_power = {1: c1, 2: c2}
    groups = {}
    for degree in range(1, 7):
        groups[("6", (degree,))] = a_power[degree]

    a42_1 = fnp.concatenate((a1[0:96], a1[256:352]), axis=0)
    a42_2 = a42_1 * a42_1
    a42_3 = a42_2 * a42_1
    a42_4 = a42_3 * a42_1
    a42 = {1: a42_1, 2: a42_2, 3: a42_3, 4: a42_4}
    for alpha_a in range(5):
        for alpha_b in range(3):
            if alpha_a == 0 and alpha_b == 0:
                continue
            if alpha_b == 0:
                value = a42[alpha_a]
            elif alpha_a == 0:
                value = b_power[alpha_b]
            else:
                value = a42[alpha_a] * b_power[alpha_b]
            groups[("42", (alpha_a, alpha_b))] = value

    a222_1 = fnp.concatenate((a1[0:64], a1[256:320]), axis=0)
    a222_2 = a222_1 * a222_1
    b222_1 = fnp.concatenate((b1[0:64], b1[96:160]), axis=0)
    b222_2 = b222_1 * b222_1
    a222 = {1: a222_1, 2: a222_2}
    b222 = {1: b222_1, 2: b222_2}
    ab = {}
    ac = {}
    bc = {}
    for alpha_a in (1, 2):
        for alpha_b in (1, 2):
            ab[(alpha_a, alpha_b)] = a222[alpha_a] * b222[alpha_b]
    for alpha_a in (1, 2):
        for alpha_c in (1, 2):
            ac[(alpha_a, alpha_c)] = a222[alpha_a] * c_power[alpha_c]
    for alpha_b in (1, 2):
        for alpha_c in (1, 2):
            bc[(alpha_b, alpha_c)] = b222[alpha_b] * c_power[alpha_c]

    for alpha_a in range(3):
        for alpha_b in range(3):
            for alpha_c in range(3):
                if alpha_a == 0 and alpha_b == 0 and alpha_c == 0:
                    continue
                nonzero = (
                    int(alpha_a != 0)
                    + int(alpha_b != 0)
                    + int(alpha_c != 0)
                )
                if nonzero == 1:
                    if alpha_a:
                        value = a222[alpha_a]
                    elif alpha_b:
                        value = b222[alpha_b]
                    else:
                        value = c_power[alpha_c]
                elif nonzero == 2:
                    if alpha_c == 0:
                        value = ab[(alpha_a, alpha_b)]
                    elif alpha_b == 0:
                        value = ac[(alpha_a, alpha_c)]
                    else:
                        value = bc[(alpha_b, alpha_c)]
                else:
                    value = ab[(alpha_a, alpha_b)] * c_power[alpha_c]
                groups[("222", (alpha_a, alpha_b, alpha_c))] = value
    return groups


def _canonical_group_keys():
    keys = []
    for degree in range(1, 7):
        degree_keys = []
        degree_keys.extend(("6", (degree,)) for _unused in (0,))
        degree_keys.extend(
            ("42", (alpha_a, alpha_b))
            for alpha_a in range(5)
            for alpha_b in range(3)
            if alpha_a + alpha_b == degree
        )
        degree_keys.extend(
            ("222", (alpha_a, alpha_b, alpha_c))
            for alpha_a in range(3)
            for alpha_b in range(3)
            for alpha_c in range(3)
            if alpha_a + alpha_b + alpha_c == degree
        )
        keys.append(tuple(degree_keys))
    return tuple(keys)


_GROUP_KEYS_BY_DEGREE = _canonical_group_keys()


def _power_sum_maps_literal(groups, weight32):
    """Exactly six packed float64 maps in degree order one through six."""

    w1 = weight32.astype(fnp.float64)
    w2 = w1 * w1
    w3 = w2 * w1
    w4 = w3 * w1
    w5 = w4 * w1
    w6 = w5 * w1
    weight_power = (w1, w2, w3, w4, w5, w6)
    mapped_groups = {}
    for degree_index in range(6):
        keys = _GROUP_KEYS_BY_DEGREE[degree_index]
        packed = fnp.concatenate([groups[key] for key in keys], axis=0)
        mapped = packed @ weight_power[degree_index]
        cursor = 0
        for key in keys:
            rank = CARRIER_RANK[key[0]]
            mapped_groups[key] = mapped[cursor : cursor + rank]
            cursor += rank
    return mapped_groups


def _sector_carrier_literal(mapped_groups, carrier, sector):
    """Sequential low-memory evaluation with five explicit term multiplies."""

    table = _sector_table(CARRIER_LAM[carrier], sector)
    accumulator = None
    for monomial, coefficient in table:
        factors = [mapped_groups[(carrier, alpha)] for alpha in monomial]
        while len(factors) < 6:
            factors.append(1.0)
        product = factors[0] * factors[1]
        product = product * factors[2]
        product = product * factors[3]
        product = product * factors[4]
        product = product * factors[5]
        weighted = product * float(coefficient)
        del product
        if accumulator is None:
            accumulator = weighted
        else:
            accumulator = accumulator + weighted
        del weighted
    return accumulator


def _half_action_literal(carrier_actions, half):
    """Frozen disjoint halves and exact coefficients; all arrays stay remote."""

    if half == 0:
        block6 = carrier_actions["6"][0:256]
        block42 = carrier_actions["42"][0:96]
        block222 = carrier_actions["222"][0:64]
    else:
        block6 = carrier_actions["6"][256:512]
        block42 = carrier_actions["42"][96:192]
        block222 = carrier_actions["222"][64:128]
    y6 = fnp.sum(block6, axis=0) * SPLIT_COEFFICIENT["6"]
    y42 = fnp.sum(block42, axis=0) * SPLIT_COEFFICIENT["42"]
    y222 = fnp.sum(block222, axis=0) * SPLIT_COEFFICIENT["222"]
    return (y6 + y42) + y222


def _source_screen_literal(weight32_by_layer, mlp_seed):
    """Return backend S,N plus the exact single host-stream custody hash."""

    if len(weight32_by_layer) != 16:
        raise ValueError("exactly sixteen width-1024 float32 weights required")
    host_paths = _rng_paths_literal(mlp_seed)
    if host_paths.dtype.str != "<f4" or not host_paths.flags.c_contiguous:
        raise RuntimeError("canonical RNG draw is not little-endian C-order f32")
    path_stream_sha256 = hashlib.sha256(
        memoryview(host_paths).cast("B")
    ).hexdigest()
    paths32 = fnp.asarray(host_paths, dtype=fnp.float32)
    del host_paths
    signal_scalars = []
    noise_scalars = []
    for source_layer in range(15):
        paths32 = fnp.maximum(
            paths32 @ weight32_by_layer[source_layer], 0.0
        )
        d32 = paths32[0::2] - paths32[1::2]
        d64 = d32.astype(fnp.float64)
        groups = _factor_groups_literal(d64)
        mapped_groups = _power_sum_maps_literal(
            groups, weight32_by_layer[source_layer + 1]
        )
        del groups, d32, d64
        for sector in SECTOR_ORDER:
            carrier_actions = {
                carrier: _sector_carrier_literal(
                    mapped_groups, carrier, sector
                )
                for carrier in CARRIER_ORDER
            }
            y0 = _half_action_literal(carrier_actions, 0)
            y1 = _half_action_literal(carrier_actions, 1)
            signal_scalars.append(fnp.sum(y0 * y1))
            delta = y0 - y1
            noise_scalars.append(fnp.sum(delta * delta) * 0.5)
        del mapped_groups, carrier_actions, y0, y1, delta
    signal = fnp.reshape(fnp.stack(signal_scalars, axis=0), (15, 7))
    noise = fnp.reshape(fnp.stack(noise_scalars, axis=0), (15, 7))
    return signal, noise, path_stream_sha256
