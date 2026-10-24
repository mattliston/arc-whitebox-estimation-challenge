# Query input completeness audit

The v1 shape-only q4/s4 price was incomplete. The current bank only fills
the active-active CP/star preplane, plus every scalar diagonal. The
certified full-width R@U query and its repeated-index restoration require
the FULL input K21; masked entries cannot silently stand in for those.

The v2 ledger charges full CP repeated-plane reads (including appended
source columns), and the extra compact-star bilinear/cold reads. The v1
receipts are preserved, not treated as a passed complete cost gate. No
width1024 accuracy or fitting work was run on that assumption.

This is a cost/integration correction, not proof that the all-active
restriction or a different separately certified diagonal-repair fusion is
impossible. Such variants would need an explicit mathematical contract
and a new certificate before any cost could be deducted.
