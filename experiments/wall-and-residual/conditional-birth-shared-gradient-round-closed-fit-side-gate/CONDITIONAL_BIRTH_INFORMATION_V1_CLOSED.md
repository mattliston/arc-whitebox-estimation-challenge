# Conditional birth shared-gradient round: closed at the fit-side gate

2026-09-05, 14:00 PDT (machine clock). Phase 2 only. No new upload.

The full reverse derivatives on 20 fit-side networks are finite. The
single pooled negative-gradient direction has norm-one coefficients;
its mean unnormalized gradient norm is .4883184 for MSE / 2e-8. The
forward-mode pooled response quadratic is

    a = 2.277363720257934e-8
    b = -4.883174927026877e-9
    c = 4.5102798930336855e-7
    predicted MSE(t) = a + 2*b*t + c*t*t.

The selected step is .01082676695, well inside the declared .25 radius.
Its predicted raw ratio is .99767850008: only .23215% improvement, below
the predeclared 10% fit-side gate. This is a linear-response diagnostic
against existing documented-1e9 local means, NOT an actual card score.

Stop this fixed shared-gradient round. The four new-update holdouts were
not evaluated or used to choose anything. No actual step, long fit,
direction/radius sweep, candidate build, panel, bill or submission.
The result does not bound the full nonlinear 192-parameter function
class or constitute a universal impossibility theorem. It supplies no
positive evidence for spending further fit compute on this proposal.

Elapsed399.43s; one nice-10, 3-thread CPU process, peakRSS11.079GB;
swap5966.50MiB unchanged. Immutable per-network derivative and response
receipts are in conditional_birth_information_progress_v1, with the
combined conditional_birth_information_v1.json. All worker jobs completed.
The usable official bank remains329926 at4.98223146018e-9 adjusted.
