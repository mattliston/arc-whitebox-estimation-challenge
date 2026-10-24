#!/usr/bin/env python3
"""Opaque process-local token validator for callable bundle V2.

An ordinary import exposes no mint function, token class, or seal.  After the
runner has durably acquired its O_EXCL claim and irreversibly consumed the
one-use capability, its custom loader may inject an empty list named
``_Q24_POST_CONSUME_MINT_SINK`` into the module globals for this single module
execution.  The module deposits one closure in that transient list, deletes
the global, and never stores the closure in its namespace.  The runner clears
the list immediately after minting one token.

This is an API/process fail-closed boundary, not a claim of isolation from a
hostile interpreter allowed arbitrary frame/closure introspection or source
rewriting.  Normal direct import, attribute lookup, serialization, copying,
and ordinary fake objects cannot produce an accepted token.
"""

from __future__ import annotations


class RuntimeTokenRequired(RuntimeError):
    """Raised when a callable-bundle operation lacks the live token."""


def _is_lower_sha256(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _build_token_api(mint_sink):
    seal = object()
    accepted = [None]
    minted = [False]

    class RuntimeToken:
        __slots__ = (
            "_seal",
            "_claim_sha256",
            "_consumed_capability_sha256",
        )

        def __init__(self, construction_seal, claim_sha256, capability_sha256):
            if construction_seal is not seal:
                raise RuntimeTokenRequired("runtime token constructor is private")
            self._seal = construction_seal
            self._claim_sha256 = claim_sha256
            self._consumed_capability_sha256 = capability_sha256

        def __reduce__(self):
            raise TypeError("runtime token is process-local and non-serializable")

        def __reduce_ex__(self, _protocol):
            raise TypeError("runtime token is process-local and non-serializable")

        def __copy__(self):
            raise TypeError("runtime token may not be copied")

        def __deepcopy__(self, _memo):
            raise TypeError("runtime token may not be copied")

    def mint_once(*, claim_sha256, consumed_capability_sha256):
        if minted[0]:
            raise RuntimeTokenRequired("runtime token was already minted")
        if not _is_lower_sha256(claim_sha256):
            raise RuntimeTokenRequired("claim digest is not canonical SHA256")
        if not _is_lower_sha256(consumed_capability_sha256):
            raise RuntimeTokenRequired(
                "consumed capability digest is not canonical SHA256"
            )
        token = RuntimeToken(seal, claim_sha256, consumed_capability_sha256)
        accepted[0] = token
        minted[0] = True
        return token

    def require(token):
        if token is None or token is not accepted[0]:
            raise RuntimeTokenRequired("a live in-process runtime token is required")
        if type(token) is not RuntimeToken or token._seal is not seal:
            raise RuntimeTokenRequired("runtime token identity is invalid")
        if not _is_lower_sha256(token._claim_sha256):
            raise RuntimeTokenRequired("runtime token claim binding is malformed")
        if not _is_lower_sha256(token._consumed_capability_sha256):
            raise RuntimeTokenRequired("runtime token capability binding is malformed")

    def receipt_fields(token):
        require(token)
        return {
            "claim_sha256": token._claim_sha256,
            "consumed_capability_sha256": token._consumed_capability_sha256,
            "in_process_identity_validated": True,
            "serializable": False,
        }

    if mint_sink is not None:
        if type(mint_sink) is not list or mint_sink:
            raise RuntimeTokenRequired("post-consume mint sink is malformed")
        mint_sink.append(mint_once)
    return require, receipt_fields


_transient_mint_sink = globals().pop("_Q24_POST_CONSUME_MINT_SINK", None)
require_runtime_token, runtime_token_receipt_fields = _build_token_api(
    _transient_mint_sink
)
del _transient_mint_sink
del _build_token_api
