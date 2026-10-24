"""EDBAND export v2 binds the manifest-aware worker2; originals remain intact.

Versioned, two-phase EDBAND export; no neural execution or native build.

prepare archives the full envelope/base/request/result/head and invokes the
existing pinned S46 base exporter. The bundle remains explicitly incomplete.
finalize requires the actual direct builder parent PATH and its explicit SHA,
emits the mandatory EDBAND sidecar, validates it with the engineer's text-only
builder, and pins every required artifact in COMPLETE.json. This is structural
export only, never a qualification or measured gain. Legacy tools must receive
the envelope (and fail); never hand them the extracted base as an EDBAND result.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import otr_edband_v1 as EB
import edband_requests_v1 as RQ
import worker_edband_v2 as WORKER

T = Path(__file__).resolve().parent
EXPORTER = T / 'export_dials_s46_v2.py'
EXPORTER_SHA = '2d6cb5f948487b9ecb1f6d940b6bb094e1b10bf5b95c27248de153481b4abdfa'
BUILDER = T.parent / 'engineer/a56_edge_band/edband_build_v3.py'
BUILDER_SHA = '175f9e4eec0a0eb0b1db2480c830248527270b51ddab3aa0dd01d875d42ee77d'
SCHEMA = 'trainer_edge_band_export_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def check_pin(item):
    require(EB.sha256(item['path']) == item['sha256'], 'Pinned file changed: ' + item['path'])


def save_bytes(path, raw):
    EB._publish(path, lambda stream: stream.write(raw), binary=True)


def archive(path, destination):
    raw = Path(path).read_bytes()
    save_bytes(destination, raw)
    require(EB.sha256(destination) == hashlib.sha256(raw).hexdigest(), 'Archive differs')
    return RQ.pin(destination)


def worker_capture_sha():
    source = (T / 'otr_engine.py').read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_network']
    require(len(nodes) == 1, 'Missing engine function')
    node = nodes[0]
    text = ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    return hashlib.sha256(WORKER.engine_source(text).encode()).hexdigest()


def verify_request_result(torch, checkpoint, checkpoint_sha, request, result, head, no_head):
    payload = EB.load_checkpoint(torch, checkpoint, checkpoint_sha)
    extra_pl = request.get('pl') or {}
    require(isinstance(extra_pl, dict), 'Request PL must be a dictionary')
    for family, values in extra_pl.items():
        require(isinstance(values, list) and len(values) == 16
                and all(type(x) in (int, float) and x == 0 for x in values),
                'Nonzero request PL is not materialized in the checkpoint; save a new envelope first: ' + family)
    spec = request.get('edge_band')
    require(isinstance(spec, dict), 'Export requires an enabled EDBAND request')
    require(Path(request['params']).resolve() == Path(checkpoint).resolve(), 'Request points at another checkpoint')
    session = EB.Session(torch, spec, request)
    require(EB._fingerprint(torch, session.payload) == EB._fingerprint(torch, payload), 'Request envelope differs')
    for key in ('id', 'rank', 'mode', 'params'):
        require(result.get(key) == request.get(key), 'Result identity mismatch: ' + key)
    require(result.get('cutsim') == request.get('cutsim'), 'Result cut simulator mismatch')
    identity = result.get('edge_band') or {}
    require(identity == session._identity, 'Result full EDBAND provenance mismatch')
    worker = result.get('edge_band_worker') or {}
    require(worker == dict(schema='trainer_edge_band_worker_v1', worker_sha256=EB.sha256(WORKER.__file__),
        sources=WORKER.PINS, patched_run_sha256=worker_capture_sha()), 'Worker source/capture provenance mismatch')
    if request['mode'] == 'grad':
        gradients = result.get('edge_band_grads') or {}
        require(gradients.get('schema') == 'trainer_edge_band_gradients_v1' and gradients.get('identity') == identity
                and gradients.get('scale') == request['scale'] == result.get('scale'), 'Band gradient provenance mismatch')
        values = gradients.get('values', {})
        require(set(values) == set(spec['train_families']), 'Missing/extra trained gradient family')
        require(all(tuple(v.shape) == (2, 16) and bool(torch.isfinite(v).all()) for v in values.values()), 'Invalid band gradients')
    if no_head:
        require(head is None and not request.get('hu') and not request.get('hf'), 'No-head contradicts request')
    else:
        require(head is not None and RQ.json_read(head) == request.get('hu'), 'Head JSON does not exactly match request HU')
    return payload


def source_receipts():
    WORKER.verify_sources()
    require(EB.sha256(EXPORTER) == EXPORTER_SHA, 'S46 exporter source changed')
    require(EB.sha256(BUILDER) == BUILDER_SHA, 'Native validator source changed')
    paths = [Path(__file__), Path(EB.__file__), Path(RQ.__file__), Path(WORKER.__file__),
             EXPORTER, T / 'export.py', T / 'otr_common.py', T / 'otr_engine.py',
             BUILDER, BUILDER.with_name('fragment_edband_v1.py')]
    paths.extend(T / name for name in WORKER.PINS)
    return [RQ.pin(p) for p in dict.fromkeys(paths)]


def prepare(args):
    torch = RQ.torch_module()
    sources = source_receipts()
    request = RQ.json_read(args.request)
    result = (RQ.load_pinned(torch, args.result, EB.sha256(args.result))
              if Path(args.result).suffix == '.pt' else RQ.json_read(args.result))
    payload = verify_request_result(torch, args.checkpoint, args.checkpoint_sha256, request, result, args.head, args.no_head)
    out = Path(args.out).resolve()
    require(not out.exists(), 'Export directory must be new')
    out.mkdir(parents=True, exist_ok=False)
    arc = out / 'archive'; arc.mkdir()
    originals = {name: RQ.pin(path) for name, path in dict(checkpoint=args.checkpoint, request=args.request, result=args.result).items()}
    archived = {name: archive(item['path'], arc / (name + Path(item['path']).suffix)) for name, item in originals.items()}
    if args.head:
        originals['head'] = RQ.pin(args.head)
        archived['head'] = archive(args.head, arc / 'head.json')
    base = arc / 'full_base.pt'
    EB._publish(base, lambda stream: torch.save(EB._clone(torch, payload['base']), stream), binary=True)
    roundtrip = RQ.load_pinned(torch, base, EB.sha256(base))
    require(EB._fingerprint(torch, roundtrip) == EB._fingerprint(torch, payload['base']), 'Complete base payload not preserved')
    archived['full_base'] = RQ.pin(base)
    # Write a persistent guard before exposing a base-only candidate directory.
    RQ.json_write(out / 'EDBAND_REQUIRED.json', dict(schema=SCHEMA, complete=False, qualified=False,
        warning='Base candidate alone omits EDGE offsets. EDBAND sidecar and COMPLETE.json are mandatory.',
        checkpoint=originals['checkpoint'], base_content_sha256=EB._fingerprint(torch, payload['base'])))
    env = dict(os.environ)
    env.update({name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    process = subprocess.run([sys.executable, '-B', str(EXPORTER), str(base), str(out / 'base_export'),
        'EDBAND incomplete base export; mandatory sidecar pending explicit direct-parent binding'],
        cwd=T, env=env, capture_output=True, text=True)
    save_bytes(out / 'base_export.stdout.txt', process.stdout.encode())
    save_bytes(out / 'base_export.stderr.txt', process.stderr.encode())
    require(process.returncode == 0, 'Base export failed; preserve incomplete directory and inspect stderr')
    base_outputs = {name: RQ.pin(out / 'base_export' / name) for name in ('candidate.py','physics_dials.json','per_layer_dials.json')}
    physics = RQ.json_read(base_outputs['physics_dials.json']['path'])
    require(physics['per_layer_sidecar']['sha256'] == base_outputs['per_layer_dials.json']['sha256'], 'Per-layer sidecar receipt differs')
    require(physics['per_layer_sidecar']['reconstruction_all16_exact'] and physics['per_layer_sidecar']['arrays_roundtrip_exact'], 'Base exporter roundtrip missing')
    for item in sources + list(originals.values()) + list(archived.values()):
        check_pin(item)
    receipt = dict(schema=SCHEMA, stage='prepared_requires_edband_sidecar', utc=RQ.utc(), complete=False, qualified=False,
        originals=originals, archived=archived, sources=sources, base_outputs=base_outputs,
        required_marker=RQ.pin(out / 'EDBAND_REQUIRED.json'),
        base_content_sha256=EB._fingerprint(torch, payload['base']), payload_sha256=EB._fingerprint(torch, payload),
        request_id=request['id'], request_sha256=RQ.canonical_request_sha(request),
        head_sha256=originals['head']['sha256'] if args.head else None,
        required_sidecars=['physics_dials.json','per_layer_dials.json','edge_band.json'])
    RQ.json_write(out / 'PREPARED.json', receipt)
    print(json.dumps(dict(prepared=str(out / 'PREPARED.json'), complete=False,
        intended_parent=base_outputs['candidate.py'], next='finalize with explicit actual parent path and SHA'), sort_keys=True))


def finalize(args):
    torch = RQ.torch_module()
    out = Path(args.bundle).resolve()
    prepared_path = out / 'PREPARED.json'
    receipt = RQ.json_read(prepared_path)
    require(receipt.get('schema') == SCHEMA and receipt.get('stage') == 'prepared_requires_edband_sidecar', 'Unexpected prepared bundle')
    require(not (out / 'COMPLETE.json').exists() and not (out / 'edge_band.json').exists(), 'Finalize outputs must be new')
    all_pins = receipt['sources'] + list(receipt['originals'].values()) + list(receipt['archived'].values()) + list(receipt['base_outputs'].values()) + [receipt['required_marker']]
    for item in all_pins:
        check_pin(item)
    require(EB.sha256(__file__) == next(p['sha256'] for p in receipt['sources'] if p['path'] == str(Path(__file__).resolve())), 'Export helper changed')
    parent = Path(args.parent_native).resolve()
    parent_pin = dict(path=str(parent), sha256=args.parent_native_sha256)
    check_pin(parent_pin)
    # This integration supports the exact exported base as builder PARENT. A
    # copied path is fine; a differently transformed parent needs a new adapter.
    require(args.parent_native_sha256 == receipt['base_outputs']['candidate.py']['sha256'], 'Actual parent differs from this exported base')
    raw_parent = parent.read_bytes()
    require(hashlib.sha256(raw_parent).hexdigest() == args.parent_native_sha256, 'Parent changed during read')
    require(b'\r' not in raw_parent, 'Parent must use LF bytes matching builder text hashing')
    payload = EB.load_checkpoint(torch, receipt['originals']['checkpoint']['path'], receipt['originals']['checkpoint']['sha256'])
    require(EB._fingerprint(torch, payload) == receipt['payload_sha256'], 'Payload fingerprint differs')
    source = dict(checkpoint_path=receipt['originals']['checkpoint']['path'],
        checkpoint_sha256=receipt['originals']['checkpoint']['sha256'], request_id=receipt['request_id'],
        request_sha256=receipt['request_sha256'], head_sha256=receipt['head_sha256'],
        prepared_receipt=RQ.pin(prepared_path), actual_direct_parent=parent_pin)
    values = EB.export_native(torch, payload, parent_native_sha256=args.parent_native_sha256, source=source)
    spec = importlib.util.spec_from_file_location('_edband_native_validator_v3', BUILDER)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    normalized = native.normalize(values)
    edits, fragment, metadata = native.prepare(raw_parent.decode('utf-8'), values)
    require(metadata['new_einsum_signatures'] == [] and len(edits) == 1, 'Unexpected native operation changes')
    sidecar = EB.save_native(out / 'edge_band.json', values)
    require(native.normalize(RQ.json_read(sidecar['path'])) == normalized, 'Native sidecar roundtrip differs')
    for item in all_pins + [parent_pin, sidecar]:
        check_pin(item)
    complete = dict(schema=SCHEMA, stage='structural_export_complete', utc=RQ.utc(), complete=True, qualified=False,
        prepared=RQ.pin(prepared_path), direct_parent=parent_pin, edge_band=sidecar,
        required_marker=receipt['required_marker'],
        required_base_outputs=receipt['base_outputs'], head=receipt['archived'].get('head'),
        required_request=receipt['archived']['request'],
        full_envelope=receipt['archived']['checkpoint'], full_base=receipt['archived']['full_base'],
        preserved_base_content_sha256=receipt['base_content_sha256'], payload_sha256=receipt['payload_sha256'],
        native_validator_metadata=metadata, fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),
        required_builder='engineer/stack/build_edband_v3.py', required_lever='EDBAND@<archived sidecar name>',
        legacy_driver_support=False, no_network_execution=True,
        remaining_gates='Enabled-zero/shared-gradient controls, nonzero torch/native parity, independent sizing/check/judge and all release gates.')
    RQ.json_write(out / 'COMPLETE.json', complete)
    print(json.dumps(dict(complete=str(out / 'COMPLETE.json'), edge_band=sidecar, qualified=False), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--request', required=True)
    p.add_argument('--result', required=True)
    head = p.add_mutually_exclusive_group(required=True)
    head.add_argument('--head')
    head.add_argument('--no-head', action='store_true')
    p.add_argument('--out', required=True)
    f = subs.add_parser('finalize')
    f.add_argument('--bundle', required=True)
    f.add_argument('--parent-native', required=True)
    f.add_argument('--parent-native-sha256', required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else finalize)(args)


if __name__ == '__main__':
    main()
