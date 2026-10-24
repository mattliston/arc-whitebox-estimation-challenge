"""Inert inverse adapter: actual served-vector JVP, unchanged scalar primal."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE/'qualification.py'
OLD_SHA = '1c15c70be04b79f797e02a652da6557bdf2b19d55f6ea355e5ccf28786d8fd4f'
SOURCE_PINS = {
    OLD: OLD_SHA,
    HERE/'model.py': 'af5ab0386bde2561543c631ace208f3649836d6581a04a69967705aec008c907',
    HERE/'PROTOCOL.md': 'c95df8ccb9aa800d79ae386760fd16ba0d489989180b442e7ce042fd873ec352',
    HERE/'watch_qualification.py': '4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb',
    HERE/'probe_ad_scalar.py': '7dac81919536000d0306d718072deb42cffd0c38b03f9d6048456142d70e426e',
    HERE/'AD_COMPATIBILITY_PROBE.md': 'cf4eecead9d64443d83e920893801bf1223476861f98642765f4d128ae44d06e',
}
INPUT_PINS = {
    HERE/'qualification_v1/result.json': '848f31f81456e16656c66f679af2d50fdc80f19373314e7b51f1d9d4575efd58',
    HERE/'qualification_v1/completion.json': '46de520189cc382dabdbdbf4cf035082a84e0c1c28eb5eb3e71810750d44b8be',
    HERE/'qualification_v1/finalization.json': 'b161710482bd7ad216a5973955d5f9268f3704361fe52b545d9e4b02d9913799',
    HERE/'qualification_watch_v1/result.json': '45749012206ade20b15feea664fac900d1c900e83f4d9e4eff3d38ad628d787a',
    HERE/'qualification_watch_v1/completion.json': '3564887d566b1e2c458162b41bce984d7c709e92c2ac890f61c626daa29c3558',
    HERE/'qualification_watch_v1/finalization.json': '1acb1250adb4b735840c125ce9363f34d35e8d0a5c12749d8145eead4c85180a',
    HERE/'ad_scalar_probe_v1/result.json': 'dc0a17fb4be9dea3f5984eebbffc6583c6a7c687c8cddddd965eaa474a0c67ea',
    HERE/'ad_scalar_probe_v1/completion.json': 'f4108462239802f7d84c65fe7e3af17c672e21b22c1cbcecd7c873272e84109a',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derived_source():
    for path, digest in {**SOURCE_PINS, **INPUT_PINS}.items():
        require(sha(path) == digest, 'Exact v1/primitive/source ownership: '+str(path))
    old_result = json.loads((HERE/'qualification_v1/result.json').read_text())
    require(old_result['passed'] is False and old_result['completed'] is False
            and old_result['calls'] == [], 'Original pre-wide failure remains failed')
    source = OLD.read_text()
    replacements = [
        ("OUT = HERE/'qualification_v1'", "OUT = HERE/'qualification_v2'"),
        ("schema='current_comp_receiver_birth_qualification_v1'", "schema='current_comp_receiver_birth_qualification_v2'"),
        ("HERE/'PROTOCOL.md': args.protocol_sha256", "HERE/'PROTOCOL_V2.md': args.protocol_sha256"),
        ("HERE/'watch_qualification.py': args.watcher_sha256", "HERE/'watch_qualification_v2.py': args.watcher_sha256"),
        ("    try:\n        source_pins = dict(SOURCE_PINS)\n",
         "    result['qualification_adapter'] = ADAPTER_PROOF\n"
         "    try:\n        source_pins = dict(SOURCE_PINS)\n"
         "        source_pins.update(ADAPTER_SOURCE_PINS)\n"
         "        require(args.model_sha256 == ADAPTER_MODEL_SHA, 'Frozen model unchanged')\n"),
        ("        for path, digest in INPUT_PINS.items():\n",
         "        INPUT_PINS.update(ADAPTER_INPUT_PINS)\n"
         "        for path, digest in INPUT_PINS.items():\n"),
        ("            tangent = gradient = None\n",
         "            tangent = gradient = served_jvp = None\n"),
        ("                    with torch.no_grad():\n"
         "                        raw, _ = card.run(training=False)\n"
         "                        value = scalar_value(raw)\n"
         "                    if mode == 'jvp':\n"
         "                        _, tv = torch.autograd.forward_ad.unpack_dual(value)\n"
         "                        require(tv is not None, 'MPS forward-AD scalar tangent is connected')\n"
         "                        tangent = float(tv.detach().cpu())\n",
         "                    with torch.no_grad():\n"
         "                        raw, _ = card.run(training=False)\n"
         "                        if mode == 'jvp':\n"
         "                            dual_served = m._CO.served(raw)\n"
         "                            yp, yt = torch.autograd.forward_ad.unpack_dual(dual_served)\n"
         "                            require(yt is not None, 'Connected actual served-vector forward AD')\n"
         "                            vector = array(yt)\n"
         "                            require(vector.shape == (card.depth, card.n) and vector.dtype == np.float32\n"
         "                                    and np.isfinite(vector).all(), 'Finite complete actual served JVP')\n"
         "                            served_jvp = save(f'jvp_{device}_{rank:04d}_depth{card.depth}', vector)\n"
         "                            j64, q64 = vector.astype(np.float64), np.asarray(cot, dtype=np.float64)\n"
         "                            tangent = float(np.sum(j64[-1]*q64[-1])/math.sqrt(card.n)\n"
         "                                      + .05*np.sum(j64[:-1]*q64[:-1])/math.sqrt((card.depth-1)*card.n))\n"
         "                            raw_primal = torch.autograd.forward_ad.unpack_dual(raw).primal\n"
         "                            require(array(yp).tobytes() == array(m._CO.served(raw_primal)).tobytes(),\n"
         "                                    'Actual dual served primal equals unchanged nondual serving')\n"
         "                            value = scalar_value(raw_primal)\n"
         "                            del dual_served, yp, yt, vector, j64, q64, raw_primal\n"
         "                        else:\n"
         "                            value = scalar_value(raw)\n"),
        ("                              diagnostics=d, frozen_hashes=before)\n",
         "                              diagnostics=d, frozen_hashes=before, served_jvp=served_jvp,\n"
         "                              scalar_raw_sha256=hashlib.sha256(array(\n"
         "                                  torch.autograd.forward_ad.unpack_dual(value).primal).tobytes()).hexdigest())\n"),
        ("                if mode == 'jvp':\n                    del tv, _\n", ""),
        ("        def identities(back, forward):\n",
         "        def identities(back, forward):\n"
         "            require(back['scalar_raw_sha256'] == forward['scalar_raw_sha256'],\n"
         "                    'Original scalar primal byte identity; no loss rewrite')\n"),
        ("            return dict(reverse_dot=reverse, forward_ad=tangent, absolute_error=error)\n",
         "            return dict(reverse_dot=reverse, forward_ad=tangent, absolute_error=error,\n"
         "                        actual_served_jvp=forward['served_jvp'])\n"),
    ]
    for old, new in replacements:
        require(source.count(old) == 1, 'Unique v2 adapter span: '+old[:80])
        source = source.replace(old, new)
    inverse = source
    for old, new in reversed(replacements):
        if new:
            require(inverse.count(new) == 1, 'Unique inverse span: '+new[:80])
            inverse = inverse.replace(new, old)
        else:
            anchor = ('                del scalar_value, card, raw, primal, served, value, theta, parameter, direction, P, A\n')
            require(inverse.count(anchor) == 1, 'Unique removed scalar-dual cleanup location')
            inverse = inverse.replace(anchor, anchor+old)
    require(inverse == OLD.read_text(), 'Exact whole-source inverse to frozen v1')
    return source, dict(original_source_sha256=OLD_SHA,
                       derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       inverse_source_byte_exact=True, frozen_model_unchanged=True,
                       original_scalar_and_reverse_unchanged=True, actual_served_vector_jvp=True,
                       no_custom_model_derivative=True, old_qualification_still_failed=True,
                       full_model_roster_unchanged=True, numerical_gates_unchanged=True)


def main():
    source, proof = derived_source()
    env = dict(__file__=str(Path(__file__).resolve()), __name__='receiver_qualification_v2_derived',
               ADAPTER_PROOF=proof, ADAPTER_SOURCE_PINS=SOURCE_PINS,
               ADAPTER_INPUT_PINS=INPUT_PINS, ADAPTER_MODEL_SHA=SOURCE_PINS[HERE/'model.py'])
    exec(compile(source, str(__file__)+':exact_adapter', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
