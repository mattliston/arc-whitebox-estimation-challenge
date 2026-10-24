"""Immutable EDBAND checkpoint/request preparation only; never enqueue or run.

checkpoint wraps a pinned complete legacy checkpoint, retaining every base field.
request copies all behavioral fields from a template and binds params to the new
envelope. --disabled writes an exact full-base control checkpoint and omits the
edge_band field. Only worker_edband_v1 supports enabled envelopes; old workers,
drivers, solvers and exporters must fail rather than extract/flatten the base.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess

import otr_edband_v1 as EB


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def canonical_request_sha(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=EB.sha256(path))


def json_read(path):
    return json.loads(Path(path).read_text())


def json_write(path, value):
    EB._publish(path, lambda stream: json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False))


def load_pinned(torch, path, expected):
    require(bool(re.fullmatch('[0-9a-f]{64}', expected)), 'Explicit input SHA256 required')
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Input checkpoint SHA mismatch')
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


def torch_module():
    import torch
    torch.set_num_threads(1)
    return torch


def make_checkpoint(args):
    torch = torch_module()
    base = load_pinned(torch, args.base, args.base_sha256)
    EB._base(torch, base)
    offsets = None
    if args.offsets:
        given = json_read(args.offsets)
        require(set(given) == set(EB.FAMILIES), 'Offsets JSON must contain bias and output')
        offsets = {family: torch.tensor(values, dtype=torch.float32) for family, values in given.items()}
    source = dict(schema='trainer_edge_band_preparation_v1', utc=utc(), base_checkpoint=pin(args.base),
                  offsets=pin(args.offsets) if args.offsets else None, helper=pin(__file__), adapter=pin(EB.__file__))
    if args.source_json:
        source['caller'] = json_read(args.source_json)
        source['caller_file'] = pin(args.source_json)
    payload = EB.make_checkpoint(torch, base, offsets=offsets, source=source)
    saved = EB.save_checkpoint(torch, args.out, payload)
    require(EB._fingerprint(torch, EB.load_checkpoint(torch, saved['path'], saved['sha256'])) ==
            EB._fingerprint(torch, payload), 'Envelope roundtrip differs')
    json_write(str(args.out) + '.receipt.json', dict(schema='trainer_edge_band_checkpoint_receipt_v1', utc=utc(),
        checkpoint=saved, source=source, base_content_sha256=EB._fingerprint(torch, base), qualified=False))
    print(json.dumps(saved, sort_keys=True))


def make_request(args):
    torch = torch_module()
    payload = EB.load_checkpoint(torch, args.checkpoint, args.checkpoint_sha256)
    request = json_read(args.template)
    require(isinstance(request, dict), 'Template must be a request object')
    require(re.fullmatch('[A-Za-z0-9_-]+', args.id) is not None, 'Unsafe/empty request id')
    require(type(request.get('rank')) is int and request['rank'] >= 0, 'Template rank missing')
    require(not request.get('dump15'), 'Remove side-effect dump15 from template')
    request.update(id=args.id, mode=args.mode)
    if args.mode == 'grad':
        scale = args.scale if args.scale is not None else request.get('scale', 1e8)
        require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0, 'Positive gradient scale required')
        request['scale'] = scale
    else:
        request.pop('scale', None)
    families = [name for name in args.train_families.split(',') if name]
    spec = EB.request_spec(args.checkpoint, args.checkpoint_sha256, families)
    disabled_base = None
    if args.disabled:
        require(args.disabled_base_out, '--disabled requires a new --disabled-base-out')
        base = EB._clone(torch, payload['base'])
        EB._publish(args.disabled_base_out, lambda stream: torch.save(base, stream), binary=True)
        disabled_base = pin(args.disabled_base_out)
        check = load_pinned(torch, disabled_base['path'], disabled_base['sha256'])
        require(EB._fingerprint(torch, check) == EB._fingerprint(torch, payload['base']), 'Disabled base roundtrip differs')
        request['params'] = disabled_base['path']
        request.pop('edge_band', None)
    else:
        require(not args.disabled_base_out, '--disabled-base-out is only valid with --disabled')
        request['params'] = spec['checkpoint']
        request['edge_band'] = spec
        # Validates the complete request/parameter/module binding, without a card.
        session = EB.Session(torch, spec, request)
        require(EB._fingerprint(torch, session.base()) == EB._fingerprint(torch, payload['base']), 'Request base differs')
    json_write(args.out, request)
    receipt = dict(schema='trainer_edge_band_request_receipt_v1', utc=utc(), request=pin(args.out),
        request_sha256=canonical_request_sha(request), template=pin(args.template),
        checkpoint=pin(args.checkpoint), disabled_base=disabled_base, enabled=not args.disabled,
        base_content_sha256=EB._fingerprint(torch, payload['base']), helper=pin(__file__),
        adapter=pin(EB.__file__), queued=False, qualified=False,
        supported_worker='worker_edband_v1.py', legacy_driver_support=False)
    json_write(str(args.out) + '.receipt.json', receipt)
    print(json.dumps(dict(request=str(Path(args.out).resolve()), enabled=not args.disabled, queued=False), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    cp = subs.add_parser('checkpoint')
    cp.add_argument('--base', required=True)
    cp.add_argument('--base-sha256', required=True)
    cp.add_argument('--offsets', help='JSON bias/output 2x16 offsets; omitted means zero')
    cp.add_argument('--source-json')
    cp.add_argument('--out', required=True)
    rq = subs.add_parser('request')
    rq.add_argument('--template', required=True)
    rq.add_argument('--checkpoint', required=True)
    rq.add_argument('--checkpoint-sha256', required=True)
    rq.add_argument('--id', required=True)
    rq.add_argument('--mode', choices=('grad', 'eval'), required=True)
    rq.add_argument('--scale', type=float)
    rq.add_argument('--train-families', default='bias,output')
    rq.add_argument('--disabled', action='store_true')
    rq.add_argument('--disabled-base-out')
    rq.add_argument('--out', required=True)
    args = parser.parse_args()
    (make_checkpoint if args.command == 'checkpoint' else make_request)(args)


if __name__ == '__main__':
    main()
