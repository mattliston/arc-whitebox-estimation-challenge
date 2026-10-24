"""Exclusive source-only native builder. Inert without --build; no estimator import."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from common import HERE, OLD, PARENT, PARENT_SHA, bindings, module, sha, unchanged, write

MARKER = '\n\n# CURRENT409F E2E4096 PAIR-CAP: UNCHANGED NATIVE RETIREMENT LAW\n'


def build():
    destination, receipt = HERE/'estimator.py', HERE/'build.json'
    if destination.exists() or receipt.exists():
        raise RuntimeError('Exclusive source creation only')
    inputs = bindings((__file__, HERE/'common.py', HERE/'PROTOCOL.md', HERE/'IMPLEMENTATION.md'))
    old = module(OLD/'build.py', 'cap4096_frozen_source_builder')
    parent, core = PARENT.read_text(), (OLD/'core.py').read_text()
    original, derived = old.derived_predict(parent)
    source = parent + MARKER + core + '\n\n' + derived
    ast.parse(source)
    assert source.split(MARKER)[0].encode() == PARENT.read_bytes()
    assert old.last_function(source, '_predict_binary') == derived
    # The sole new effective function is produced by the frozen, inverse-proved
    # three-edit constructor. No P payload, runtime helper, or class is replaced.
    for name in ('features', '_nonlinear_pair_batch', '_nonlinear_terminal_live',
                 'omitted_radial_source', '_star_compact', '_cov_compact', '_mm'):
        assert old.last_function(source, name) == old.last_function(parent, name)
    unchanged(inputs)
    record = dict(passed=True, schema='current_paircap4096_e2e_native_source_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), source_bindings=inputs,
        parent_sha256=PARENT_SHA, candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate=str(destination), inverse_source_byte_exact=True,
        parent_source_prefix_byte_exact=True, predict_three_edit_inverse_byte_exact=True,
        signed_pair_cap=4096, physical_column_cap=8192, all16_P_unchanged=True,
        source_only=True, numerical_execution=False, full_cost_measured=False,
        generated_candidate_executed=False, training_admitted=False)
    with destination.open('xb') as stream:
        stream.write(source.encode())
    assert sha(destination) == record['candidate_sha256']
    write(receipt, record)
    print(record['candidate_sha256'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    if args.build:
        build()
    else:
        parser.print_help()
