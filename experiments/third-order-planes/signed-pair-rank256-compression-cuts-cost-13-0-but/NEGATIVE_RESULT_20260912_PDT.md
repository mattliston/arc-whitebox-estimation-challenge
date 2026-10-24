# Fixed PAIR256 current-head candidate: actual objective failure

Recorded September 12, 2026 PDT (September 13 UTC). This is the original
fixed PAIR256-only arm, not JOINT+D4 and not a refit. All earlier sources,
receipts and gates remain unchanged. Official bank #330390 remains
4.53886349950395e-9; failed #330840 is not a bank.

One reused development network, rank 37, ordinary final-layer MSE against
the same original 1-billion-sample truth, and each candidate's complete
cold native FLOPs including initialization:

| Candidate | Raw MSE | Complete FLOPs | Adjusted |
| --- | ---: | ---: | ---: |
| Current c2 | 2.1882384000672836e-8 | 406399683439 | 4.044065431464368e-9 |
| Fixed PAIR256 | 2.9752405624230613e-8 | 353467478085 | 4.78235405487738e-9 |

PAIR/current ratios: raw 1.3596510153242805, compute 0.8697533302534055,
adjusted 1.182560998560717. The prospective adjusted-ratio <=0.95 gate
fails. A 13.0% complete-cost reduction did not offset a 36.0% raw increase.
No panel, RPC expansion, submission, gate adjustment or automatic refit
follows. This is a one-network negative result, not final-score evidence.

Candidate d7e9cba805c286bc7e1f81d36d05e4bc8cfb9b992324e61037d3aa6bee5cf6ba
passed source/build checks, all 16 raw-domain checks, all 47 selectors,
first-five-layer equality to current c2, and its own exact cold/repeat
outputs. Cold/repeat output SHA256:
1e8941416ce9cf1c5ad8bbed2b101f76fea1085f6963dd591ae3a7a1de7832c6.

Cold native wall 59.4946772493422 s, repeat 58.75724762491882 s. The native
watch completed in 127.1141717499122 s, peak sampled tree RSS 6078414848 B,
maximum completed-sample interval 0.4123147074133158 s, no added swap and
complete process cleanup. These are native watched measurements, not
production RPC wall-time qualification.

The score watch completed in 2.851133083924651 s, peak 332611584 B,
maximum completed-sample interval 0.3573243748396635 s, no added swap and
complete cleanup. Truth was opened only after predictions and bills were
sealed. No numerical processes from these stages remain.

Score result/events/seal SHA256:

- 581d475d476ba9afa8da982a60beadc4c0d54c3ddb3ac0d1d7f1992186aa63cb
- 85dcbacc1c360437363485bd878547dff0e3a0bfe10e4b12e3a8fe001f8fd12f
- b5c1ef850c7aed37762d3012763662d023cbd39ead23926178c8e4beeb4765fe

Score-watch result/events/seal SHA256:

- b0cf42706e0f230d9cfc4f212daa010ebf244d695e07af976afbd877fe25006b
- 3b616c76ee7797bd04396dc5799e50d753df46782eac1a733fe9fcf31ddb5ede
- b0edf1d1855e180e8916d69082165d119038749dc3a52a190c213cb3068bedf3
