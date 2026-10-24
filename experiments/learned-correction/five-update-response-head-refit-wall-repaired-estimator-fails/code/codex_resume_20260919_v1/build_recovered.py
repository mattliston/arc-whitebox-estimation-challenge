"""Recover one completed payload; write only into this versioned experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
OLD = P2 / 'astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1'
BASE = P2 / 'fable_wall_repair_v1/cand_c2_wallrepair_I95.py'
PAYLOAD_SHA = '90b6077e08649d3242ec6dc3e3f5cbdfd9e9fb31c0498c9565e64d67fd45c501'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain(folder):
    r, c, f = [json.loads((folder / n).read_text()) for n in
               ('result.json', 'completion.json', 'finalization.json')]
    assert all(x['passed'] and x['completed'] for x in (r, c, f))
    assert c['result_sha256'] == f['result_sha256'] == sha(folder / 'result.json')
    assert f['completion_sha256'] == sha(folder / 'completion.json')
    return r


def main():
    spec = importlib.util.spec_from_file_location('old_literal_builder', OLD / 'build_native.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    # The unfinished historical builder accepted list literals only, but both
    # its own c2 parent and I95 store A/cstar as tuples. Normalize containers
    # for representation validation; never alter any original source file.
    original_pack = builder.pack_literal
    def pack_sequence(value):
        return original_pack(list(value) if isinstance(value, tuple) else value)
    builder.pack_literal = pack_sequence
    train = chain(OLD / 'training_v1')
    watch = chain(OLD / 'training_watch_v1')
    assert train['training_dose_completed'] and train['full_model_calls'] == 60
    assert train['optimizer_steps'] == 5 and train['backwards'] == 20
    assert watch['whole_owned_tree_retired'] and watch['retired_call_count'] == 60
    assert watch['child_result_sha256'] == sha(OLD / 'training_v1/result.json')
    for name, digest in builder.TRAIN_OWNERS.items():
        assert sha(OLD / name) == digest
    payload = OLD / 'training_v1/terminal_payload.npz'
    meta_path = OLD / 'training_v1/terminal_payload.json'
    meta = json.loads(meta_path.read_text())
    assert sha(payload) == meta['sha256'] == PAYLOAD_SHA
    assert train['artifacts'][str(payload)] == PAYLOAD_SHA
    assert sha(meta_path) == train['artifacts'][str(meta_path)]
    assert meta['validation_targets_accessed'] is False
    assert meta['parameter_updates'] == 5 and meta['trainable_keys'] == list(builder.PKEYS)
    last = json.loads((OLD / 'training_v1/update_04.json').read_text())
    assert sha(OLD / 'training_v1/update_04.json') == train['updates'][4]['sha256']
    assert last['parameter_hashes'] == meta['parameter_hashes']
    c2 = builder.PARENT.read_bytes()
    assert hashlib.sha256(c2).hexdigest() == builder.PARENT_SHA
    base = BASE.read_bytes()
    # The wall repair changed the execution schedule, not any learned literal.
    assert builder.surface(base)[0] == builder.surface(c2)[0]
    assert builder.surface(base)[2:] == builder.surface(c2)[2:]
    terminal = builder.decode_payload(payload, meta)
    candidate, proof = builder.emit(base, terminal)
    null, _ = builder.emit(base, builder.surface(base)[0])
    assert null == base
    assert proof['parameter_hashes'] == meta['parameter_hashes']
    # The reused emitter's report hard-codes its historical c2 parent. Record
    # the actual I95 parent here; the byte-inverse check itself used I95.
    proof['emitter_historical_parent_sha256'] = proof['parent_sha256']
    proof['parent_sha256'] = sha(BASE)
    proof['null_source_byte_exact'] = True
    proof['payload_sha256'] = PAYLOAD_SHA
    proof['training_result_sha256'] = sha(OLD / 'training_v1/result.json')
    proof['training_watch_result_sha256'] = sha(OLD / 'training_watch_v1/result.json')
    proof['build_helper_sha256'] = sha(OLD / 'build_native.py')
    proof['exporter_sha256'] = sha(__file__)
    out = HERE / 'recovered_p5_I95.py'
    with out.open('xb') as stream:
        stream.write(candidate)
    with (HERE / 'recovered_build.json').open('x') as stream:
        json.dump(proof, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: proof[k] for k in ('parent_sha256', 'candidate_sha256',
          'whole_source_inverse_verified', 'modified_keys', 'null_source_byte_exact')}))


if __name__ == '__main__':
    main()
