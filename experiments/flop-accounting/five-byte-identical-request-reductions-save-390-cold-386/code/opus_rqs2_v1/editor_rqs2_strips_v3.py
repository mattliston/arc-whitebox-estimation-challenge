"""[opus-lowres] RQS round 2, piece 1 v3 (byte-identical; patches every def holding the anchor: request diet; editor: apply(text) -> text).
The strip-path temporaries of the LIVE _hzg_product and _hz7_product (o = fnp.empty((M, w)) holding one padded column packet, read
only by the copyto calls of the same iteration) become RQS-kept outputs: _rqs_get per (role, (M, w)) and _rqs_out, so _l3o_last takes
its cached quadrant views (no reshape / 2 unstack) and the packet skips its allocation. Safe by the same proof as RQS v1's tiles: every
element of o is written by the packet's own unwind before any read, and the only reads are that iteration's copyto (server order);
held views of an old packet are never read. Arithmetic, operands and order unchanged."""


def _sub_in_def(text, def_head, old, new):
    # the ONE definition whose body holds the anchor (the live wrapper later in the file calls it as its captured parent)
    hits = []
    i = text.find(def_head)
    while i >= 0:
        j = text.index('\ndef ', i + len(def_head))
        if text[i:j].count(old) == 1:
            hits.append((i, j))
        i = text.find(def_head, i + 1)
    assert 1 <= len(hits) <= 2, (def_head, len(hits))
    for i, j in reversed(hits):                    # every definition holding the anchor (live parent and its own HZD parent)
        text = text[:i] + text[i:j].replace(old, new) + text[j:]
    return text


def apply(text):
    text = _sub_in_def(text, 'def _hzg_product(a, b, m, k, n):',
                       "        o = fnp.empty((M, w), dtype=fnp.float32)\n",
                       "        o = _rqs_get('hzgo', (M, w), lambda: _rqs_out(fnp.empty((M, w), dtype=fnp.float32)))   # RQS2: kept strip packet\n")
    text = _sub_in_def(text, 'def _hz7_product(a, b, m, k, n, dest=None):',
                       "            o = fnp.empty((M, w), dtype=fnp.float32)\n",
                       "            o = _rqs_get('hz7o', (M, w), lambda: _rqs_out(fnp.empty((M, w), dtype=fnp.float32)))   # RQS2: kept strip packet\n")
    return text
