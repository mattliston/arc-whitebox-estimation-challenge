"""Inert full-card integration of the qualified native-forward rangefinder.

Only Y at the ten original shared joins is replaced. The aligned physical
pair/covariance/star functions, QR, F/core projection, push/read/commit, young
bank, selectors, P/NODE/cstar and served convention are otherwise unchanged.
There is no launcher, numerical import, target, parameter update or fit here.

Owner sequence (not executed by importing or preparing this source):
    m = model(REVIEWED_THIS_SHA, tick)
    bindings = m.authenticate(REVIEWED_SHARED_OPERATOR_SHA)  # still stdlib
    b, p = m.native_bridge, m.native_prepared
    b.select_backend(p, tick)  # BEFORE importing NumPy/Torch
    # Owner imports/pins NumPy2.4.6, Torch2.7.0, CPU threads1 and threadpoolctl.
    att = b.attest(p, np, torch, threadpool_info, tick)
    workspace = b.build_workspace(p, np, att)
    P, A = m.frozen('cpu', trainable_P=False)
    card = m.card(35, P=P, A=A, weights=owned_weights,
                  native_workspace=workspace, native_attestation=att,
                  numpy_api=np)

The caller keeps workspace open through every forward/checkpoint recomputation;
the per-card bridge holds only its weak reference. Retire card/graphs/parameters,
close and delete workspace, and verify weakrefs under the independently owned
resource scope. Preparation does not admit prior receipts or authorize any run.
Backward is the qualified real-algebra VJP at rounded product intermediates,
not a derivative of FP32 rounding. The unchanged all16 <=1e-5 fidelity gate,
full-causal gradient/resources and any fit remain separate prerequisites.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ALIGNMENT = HERE / 'physical_alignment.py'
OPERATOR = HERE / 'shared_operator.py'
BRIDGE = HERE / 'differentiable_rangefinder.py'
PINS = {
    ALIGNMENT: '9e1852e5b3a94e8d820b54003ba18a67dcce2776d0a763c474ffec24ef2d29b5',
    OPERATOR: '09136ff645869df9d8fae3e2e8c16ff65c777971c308d1ba8c124d35649d2ad2',
    HERE / 'model.py': '65805a9ffbdb310fb374b82279b79d2f0d0931b00f45191fbb3d377b10802311',
    BRIDGE: '584b468029c0de392afda7494ac0c1030e70f1c244b6490c5d912c4ea15fa572',
}

# The weight already present above the replaced Y block is retained verbatim:
# it still owns joined_core below it. The bridge recomputes the same lambda
# square for Y (one additional scalar multiply per joining signed column).
# No F/core/head coordinate is detached and this work is not credited away.
OPERATOR_EDITS = (
    ('         Q, F, core, cohort_stops, rank=384):\n',
     '         Q, F, core, cohort_stops, rank=384, *, rangefinder=None):\n'),
    ('        Y = J @ (weight[:, None]*(J.T @ Omega))\n'
     '        if old_columns:\n'
     '            Y = Y + Qp @ (core @ (Qp.T @ Omega))\n',
     '        _require(rangefinder is not None, "Explicit per-card native rangefinder")\n'
     '        Y = rangefinder(J, coefficient, Omega, Qp, core if old_columns else None)\n'),
)
STEP_EDIT = (
    '        previous_indices, Q, F, core, stops, rank=384)\n',
    '        previous_indices, Q, F, core, stops, rank=384,\n'
    '        rangefinder=self._native_rangefinder)\n',
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _load(path, name):
    require(sha(path) == PINS[path], 'Pinned inert integration dependency')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _certificate(reviewed_sha256, tick):
    require(not {'numpy', 'torch', 'flopscope', 'scipy', 'threadpoolctl'} & set(sys.modules),
            'Integration source admission before numerical imports')
    bindings = {str(p): h for p, h in PINS.items()}
    bindings[str(Path(__file__).resolve())] = reviewed_sha256
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Exact reviewed integration source: ' + path)
        if tick is not None:
            tick()
    return bindings


def model(reviewed_sha256, tick=None):
    """Return an inert, separately owned aligned-model API; no numerical state.

    native_bridge/native_prepared expose the frozen qualified backend API.
    card() and card_class() additionally require explicit native_workspace,
    native_attestation and numpy_api. They cannot silently allocate scratch or
    choose a backend. All cards use CPU FP32/rank384 at joins5..14; no join15.
    """
    direct = _certificate(reviewed_sha256, tick)
    alignment = _load(ALIGNMENT, 'native_Y_frozen_physical_alignment')
    bridge = _load(BRIDGE, 'native_Y_frozen_differentiable_bridge')
    prepared = bridge.prepare(PINS[BRIDGE], tick)
    result = alignment.model(PINS[ALIGNMENT])
    require(result._CLASS is None and result._OP is None,
            'Fresh private model before operator/class installation')

    operator_source, operator_proof = alignment._replace(OPERATOR.read_text(), OPERATOR_EDITS)
    operator = types.ModuleType('current_shared384_native_Y_operator')
    operator.__file__ = str(OPERATOR)
    exec(compile(operator_source, str(Path(__file__)) + ':derived_shared_operator', 'exec'),
         operator.__dict__)
    require(operator.PRODUCTION_RANK == 384 and operator.PRODUCTION_WIDTH == 1024
            and operator.FIRST_JOIN == 5 and operator.LAST_JOIN == 14,
            'Unchanged production geometry and original ten-join schedule')

    def align_step(source):
        # Existing physical alignment owns its own three-splice inverse first.
        aligned, physical_proof = alignment.align_step(source)
        text, step_proof = alignment._replace(aligned, (STEP_EDIT,))
        proof = dict(physical_proof)
        proof.update(native_rangefinder_call_wiring=step_proof,
            all_other_aligned_step_bytes_unchanged=True,
            per_card_rangefinder_not_global_carrier=True)
        return text, proof

    # This name is the existing physical adapter's source-only callback. Its
    # output changes only the explicit argument passed to shared_push.
    result._physical_align_source = align_step
    original_authenticate = result.authenticate

    def authenticate(shared_sha256):
        require(shared_sha256 == PINS[OPERATOR], 'Frozen original operator authority')
        fresh = _certificate(reviewed_sha256, tick)
        bindings = original_authenticate(shared_sha256)
        for inventory in (prepared['source_bindings'], fresh):
            for path, expected in inventory.items():
                require(path not in bindings or bindings[path] == expected, 'No source authority conflict')
                bindings[path] = expected
        result._OP = operator
        result._BINDINGS = dict(bindings)
        return dict(bindings)

    result.authenticate = authenticate
    original_card_class = result.card_class
    installed_class = None

    def card_class():
        nonlocal installed_class
        require(result._BINDINGS is not None and result._OP is operator,
                'Complete inert admission and exact derived operator before runtime')
        if installed_class is not None:
            return installed_class
        parent = original_card_class()

        class NativeRangefinderCard(parent):
            def __init__(self, rank, *, P, A, device='cpu', weights=None,
                         native_workspace, native_attestation, numpy_api):
                require(device == 'cpu' and weights is not None,
                        'Explicit owned CPU production fixture, never an implicit file load')
                torch = result.runtime().torch
                # This per-card bridge saves only original differentiable mm
                # operands and a WEAK workspace reference. No model-global
                # workspace, per-layer cache, state copy or detached Y exists.
                run = bridge.build(torch, numpy_api, prepared, native_attestation,
                                   native_workspace)
                require(run.rank == 384 and run.dtype == torch.float32 and not run.mathematical,
                        'Only the qualified production rangefinder branch')
                super().__init__(rank, P=P, A=A, device=device, weights=weights)
                self._native_rangefinder = run
                self.source_bindings.update(result._BINDINGS)

            def diagnostics(self):
                answer = super().diagnostics()
                answer['native_rangefinder_integration'] = dict(
                    configured_join_layers=list(range(5, 15)), rank=384,
                    caller_owned_workspace=True, per_card_bridge=True,
                    operator_inverse=operator_proof,
                    only_Y_arithmetic_changed=True, native_gradient_certified=False,
                    full_forward_fidelity_qualified=False)
                return answer

        installed_class = NativeRangefinderCard
        return installed_class

    def card(rank, *, P, A, device='cpu', weights=None,
             native_workspace, native_attestation, numpy_api):
        return card_class()(rank, P=P, A=A, device=device, weights=weights,
            native_workspace=native_workspace, native_attestation=native_attestation,
            numpy_api=numpy_api)

    result.card_class = card_class
    result.card = card
    result.native_bridge = bridge
    result.native_prepared = prepared
    result.rangefinder_operator_source = operator_source
    result.rangefinder_alignment_source_proof = dict(
        source_sha256=reviewed_sha256, direct_source_bindings=direct,
        operator_inverse=operator_proof, bridge_source_proof=prepared['source_proof'],
        original_model_operator_alignment_and_heads_unmodified=True,
        configured_join_layers=list(range(5, 15)), rank=384, CPU_FP32=True,
        no_global_mutable_carrier=True, no_state_detachment=True,
        QR_factor_core_push_read_commit_unchanged=True,
        native_Winograd_only_in_Y=True, all_other_GEMMs_still_Torch=True,
        numerical_imports=False, targets_accessed=False, fit_authorized=False)
    return result
