

# End-only retained storage cap. Live admission and transform arithmetic stay unchanged.
_O18_OWP_RETAIN_BYTES = 950_000_000
_O18_OWP_RETAIN_PARENT_END = _owp_end
_O18_OWP_RETAIN_STATS = dict(calls=0, trimmed_slots=0, trimmed_bytes=0, skipped_context=0)

def _owp_end():
    # _owp_predict invokes this only after its parent prediction and all inner
    # UBC/UBX/RC/RQX finally blocks have returned. Preserve original cleanup first.
    _O18_OWP_RETAIN_PARENT_END()
    _O18_OWP_RETAIN_STATS['calls'] += 1
    # Defensive no-op on an unexpected live consumer; never evict live storage.
    if (_OWP['active'] or _UBC_RUN[0] is not None or _UBC_CONTEXT[0] is not None
            or _UBX_CTX[0] is not None or _UBX_RC[0] is not None):
        _O18_OWP_RETAIN_STATS['skipped_context'] += 1
        return
    before = _OWP['bytes']
    for index in sorted(tuple(_OWP['slots']), reverse=True):
        if _OWP['bytes'] <= _O18_OWP_RETAIN_BYTES:
            break
        _owp_evict(index)
        _O18_OWP_RETAIN_STATS['trimmed_slots'] += 1
    _O18_OWP_RETAIN_STATS['trimmed_bytes'] += before - _OWP['bytes']
