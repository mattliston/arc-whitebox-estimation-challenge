"""DRAFT: read-only poll of the exact accepted current gain-slice submission.

Whole-source adaptation of the frozen original poll. No upload, forum,
configuration write, automatic resubmission, or fabricated submission id.
Requires current positive submit result/completion/finalization and explicit id.
"""
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_shared_old384_v1/poll.py'
OWNER_SHA='ec56f8aa9eac446ee97c6174c767f3577a801f69a2e507710e08f6499c7b4f3d'


def derive_source():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Frozen original read-only poll source')
    original=raw.decode();source=original;history=[]
    for before,after in (
        ('4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3',
         'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'),
        ('current_comp_shared_old384_release_v1','current_comp_shared_gain_slice_release_v1'),
        ('"""Read-only status of the exact accepted SHARED Phase2 milestone submission.',
         '"""Read-only status of the exact accepted current gain-slice Phase2 submission.'),
        ("            candidate_sha256=CANDIDATE_SHA,per_mlp_counts_available=False,",
         "            candidate_sha256=CANDIDATE_SHA,per_mlp_counts_available=False,\n"
         "            current_poll_source_proof=_current_poll_proof,"),
    ):
        if source.count(before)!=1:
            raise RuntimeError('Exact current poll adaptation span')
        at=source.index(before);history.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(history):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact current poll inverse span')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Whole original read-only accepted-id and credential core restored')
    compile(source,'current-gain-poll-derived','exec')
    return source,dict(owner_sha256=OWNER_SHA,whole_source_inverse_verified=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_identity_retry_deadline_projection_and_credential_core_unchanged=True)


if __name__=='__main__':
    if '--poll' not in sys.argv:
        raise RuntimeError('Explicit read-only poll only')
    source,proof=derive_source()
    scope=dict(__name__='__main__',__file__=str(Path(__file__).resolve()),_current_poll_proof=proof)
    exec(compile(source,str(OWNER)+':current-accepted-id','exec'),scope)
