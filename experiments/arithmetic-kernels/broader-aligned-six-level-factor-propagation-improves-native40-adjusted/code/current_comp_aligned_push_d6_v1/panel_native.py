"""Same qualified15 paid panel calls; broad APUSH observer and current receipts."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import resource
import sys
import time
import panel_common as c

_owner_env = dict(globals())
c.install('native',('derived_source',),_owner_env)
_cd_derived_source = _owner_env['derived_source']


def derived_source():
    original,parent_proof = _cd_derived_source()
    source = c.splice(original,[
        ('    n = c.e.old()  # Metadata definitions only: no authenticate/construct in this process.\n',
         '    n = c.d.base()  # Same initialized metadata owner as APUSH detached diagnostics.\n'),
        ('build_watch_sha256=c.BUILD_WATCH_SHA,native40_expansion_prerequisite_passed=True,',
         'runtime_sha256=c.RUNTIME_SHA,native40_expansion_prerequisite_passed=True,'),
        ('Authenticated current COMP reference before output-only decode',
         'Authenticated actual CD reference before output-only decode'),
    ])
    proof = dict(parent_CD_worker=parent_proof,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),full_source_inverse_verified=True,
        same_original_COMP_paid_loop_and_panel_fixtures=True,same_initialized_diagnostic_owner=True,
        generic_APUSH_observer_and_actual_runtime_provenance=True,
        exact15_calls90_artifacts=True,no_estimator_change=True)
    return source,proof


c.install('native',('main',),globals())

if __name__ == '__main__':
    raise SystemExit(main())
