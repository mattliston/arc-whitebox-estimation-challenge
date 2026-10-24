"""Fixed common-cubic Q-only fit; source-inverse adapter, inert without --run."""
import ast
import hashlib
from pathlib import Path
import re

_HERE = Path(__file__).resolve().parent
_RD = _HERE.parent/'current_comp_receiver_directions_v1'
_RD_SHA = '7d31b34b8074fc338be5a74ef8ad50296565ab54c4f7b46c6393df5443c7de46'


def derived_source():
    path = _RD/'train.py'
    original = path.read_text()
    if hashlib.sha256(path.read_bytes()).hexdigest() != _RD_SHA:
        raise RuntimeError('Frozen RD trainer source changed')
    fixed = {
        _RD/'train.py': _RD_SHA,
        _RD/'TRAIN_PROTOCOL.md': '7c989ea707c9790d92793084569378983fad0861412d233153433e70ad012531',
        _RD/'watch_train.py': 'c7623198f8fb41ed1dbbe4a7ea4a0b9c7f5bd38bd0015f650d174fb4b4d87bce',
        _RD/'TRAIN_WATCH_SPEC.md': '68acc32338ef68085c68993a557ee67c64704d36acc8133126f95639e4bb592b',
        _HERE/'model.py': 'f068071bca67d052da101b47952eff15216d6e408a485148d2f250829ee8d37d',
        _HERE/'MODEL_PROTOCOL.md': '6eff6e23216dbc2f5df30c368cb0a3d39ead2f2dc736c13ef2a1567b61d0ccba',
        _HERE/'qualification.py': 'fd92241b5342d0e7b603b87418351fa08a8c56991da4f0ffa50aa5d1a54c36a0',
        _HERE/'QUALIFICATION_PROTOCOL.md': 'bce829f79260d9af7f7ea2e7ee4756283dc80df8a4fbd080d5e32454781e83b9',
        _HERE/'watch_qualification.py': '935eb9ffd0d040309b406121e800cabae62f6bf7fd5b2d4e0a85063c2f9a17bb',
        _HERE/'WATCH_SPEC.md': '1800774bd42608104fc9d99f9fc36f94f77a12993b3d5839ee6925ae54a8ecb2',
        _HERE/'qualification_v1/result.json': 'e15610a1c80b145a2e00804bc7afaf1732cf1c50430ab4f5408a1e62d1ae526d',
        _HERE/'qualification_v1/completion.json': '435578af00009e6a1aac758a6b330c5801006aaddb394b1f9de0cf8f1532e4b6',
        _HERE/'qualification_v1/finalization.json': 'f5f3553e5f07c2081a4ecaf9b01acaeca4c25655510c23a285bfaf80356464a9',
        _HERE/'qualification_watch_v1/result.json': '0b52cecdbc259b5bab2758686fab27839cfc1f19f447029ef5330d743e4762cc',
        _HERE/'qualification_watch_v1/completion.json': 'e11ee1ca408b98009395e54e28b9d53b7477bbb0acbeed6260a9cb38773e1250',
        _HERE/'qualification_watch_v1/finalization.json': '4c43bce531454040cda4c3e3f860548049315ff15f529cfe9521710c583c2c2b',
    }
    inserted_pins = ''.join('    Path('+repr(str(p))+'): '+repr(d)+',\n' for p, d in fixed.items())
    changes = (
        ('PINS = {\n', 'PINS = {\n'+inserted_pins),
        ('current_comp_receiver_directions_training_v1', 'current_comp_common_cubic_training_v1'),
        ('current_comp_receiver_directions_qualification_v1', 'current_comp_common_cubic_qualification_v1'),
        ('9309093, 600, 2**41', '9309101, 600, 2**41'),
        ('ad9b923d6c322585257013b293237a28b1d77826a86fb4a412a21a9bd9118b8c',
         'd90cec782ef7631acc227b1bb857f92d71284920521341764c3d182b87368d80'),
        ("HERE/'TRAIN_WATCH_SPEC.md': '68acc32338ef68085c68993a557ee67c64704d36acc8133126f95639e4bb592b'",
         "HERE/'TRAIN_WATCH_SPEC.md': 'b6435d051444b0d13443ca83d0692cb1c866b44526269d07136409555ea87503'"),
        ("set(forward['node_head_calls']) == set(forward['receiver_head_calls']) == {str(l) for l in range(1, 15)}",
         "set(forward['node_head_calls']) == {str(l) for l in range(1, 15)} and set(forward['common_head_calls']) == {str(l) for l in range(10, 15)}"),
        ('All14 original NODE and receiver birth calls', 'All14 original NODE and five late common-cubic birth calls'),
        ("and q['execution_integrity_passed'] and q['full_model_calls'] == q['retired_call_count'] == 5",
         "and q['execution_integrity_passed'] and q['full_model_calls'] == q['retired_call_count'] == 5\n"
         "        and q['null_forwards'] == 2 and q['common_backwards'] == 3\n"
         "        and q['Q_fixture']['shape'] == [65] and q['Q_fixture']['dtype'] == 'float32'"),
        ('av.shape == rv.shape == (64, 4)', 'av.shape == (64, 4) and rv.shape == (65,)'),
        ('value.shape == (64, 4)', 'value.shape == (65,)'),
        ('Finite all256 receiver gradient coordinates', 'Finite all65 common-cubic gradient coordinates'),
        ('R.numel() == 256', 'R.numel() == 65'),
        ('np.zeros((64, 4), dtype=np.float32)', 'np.zeros((65,), dtype=np.float32)'),
        ('trainable_parameter_count=256', 'trainable_parameter_count=65'),
        ('R_lr=[.01, .001]', 'R_lr=[.002, .0002]'),
        ('Adam([R], lr=.01,', 'Adam([R], lr=.002,'),
        ('lr = .001+(.01-.001)*cosine', 'lr = .0002+(.002-.0002)*cosine'),
        ("arm='receiver'", "arm='common_cubic'"),
        ("args.arm = 'receiver'", "args.arm = 'common_cubic'"),
        ("stage.record['trainer_source_proof'] = proof",
         "stage.record['trainer_source_proof'] = proof\n"
         "        stage.record['trainer_adapter_source_proof'] = TRAINER_ADAPTER_PROOF"),
        ('receiver_directions_training_model', 'common_cubic_training_model'),
    )
    derived, applied = original, []
    for old, new in changes:
        count = derived.count(old)
        if not count:
            raise RuntimeError('Missing exact CC trainer span: '+old)
        derived = derived.replace(old, new)
        applied.append((old, new, count))
    # Names inside the inherited run_card source strings must change too.
    # Use complete words, not substrings; destination names must be unused.
    names = {name: name.replace('R', 'Q') for name in (
        'R', 'original_R', 'initial_R', 'terminal_R', 'zero_R', 'before_R',
        'after_R', 'R_lr', 'R_before_sha256', 'trainable_R')}
    # The explicit qualification Q_fixture above is not an R parameter alias.
    pattern = r'\b(?:'+('|'.join(re.escape(k) for k in names))+r')\b'
    inverse_pattern = r'\b(?:'+('|'.join(re.escape(k) for k in names.values()))+r')\b'
    if re.search(inverse_pattern, derived):
        raise RuntimeError('CC parameter alias collision before renaming')
    renamed = re.sub(pattern, lambda m: names[m.group()], derived)
    reverse_names = {value: key for key, value in names.items()}
    restored = re.sub(inverse_pattern, lambda m: reverse_names[m.group()], renamed)
    if restored != derived:
        raise RuntimeError('Exact parameter-name inverse failed')
    # Remaining receiver words are diagnostic prose, not source paths/schemas.
    prose_count = len(re.findall(r'\breceiver\b', renamed))
    if 'common-cubic-proposal' in renamed:
        raise RuntimeError('Diagnostic rename collision')
    result = re.sub(r'\breceiver\b', 'common-cubic-proposal', renamed)
    inverse = result.replace('common-cubic-proposal', 'receiver')
    if inverse != renamed:
        raise RuntimeError('Diagnostic wording inverse failed')
    inverse = re.sub(inverse_pattern, lambda m: reverse_names[m.group()], inverse)
    for old, new, count in reversed(applied):
        if inverse.count(new) != count:
            raise RuntimeError('Unexpected exact CC trainer inverse: '+new)
        inverse = inverse.replace(new, old)
    if inverse != original:
        raise RuntimeError('Whole frozen RD trainer source inverse failed')
    ast.parse(result)
    return result, dict(owner=str(path), owner_sha256=_RD_SHA,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(result.encode()).hexdigest(),
        inverse_source_byte_exact=True, alias_source_byte_exact=True,
        original_loss_and_retirement_unchanged=True,
        diagnostic_word_replacements=prose_count, parameter_aliases=names)


_TRAIN_SOURCE, TRAINER_ADAPTER_PROOF = derived_source()
exec(compile(_TRAIN_SOURCE, str(_RD/'train.py')+':common-cubic-training-v1', 'exec'), globals())
