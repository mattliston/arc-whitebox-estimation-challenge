"""Original P2 numerical worker; new owner/roster schema and metadata hooks only."""
from pathlib import Path
import common as c

_changes = [
    ('current_p1_native_objective_p2_roster_v1',c.RESTART_ROSTER_SCHEMA,1),
    ("if __name__ == '__main__':\n    raise SystemExit(main())\n",
     '# Frozen entrypoint suppressed; this version owns the sole explicit entrypoint.\n',1),
]
_source, RESTART_WORKER_PROOF = c.splice_source(c.ORIGINAL/'worker.py',c.ORIGINAL_WORKER_SHA,_changes)
exec(compile(_source,str(Path(__file__).resolve())+':frozen_p2_worker','exec'),globals())
_original_roster = roster


def roster(stage,np):
    entries = _original_roster(stage,np)
    stage.record['restart_worker_inverse'] = RESTART_WORKER_PROOF
    if stage.args.stage == 'train':
        old = stage.restart_prefix
        c.require(stage.record['training_order'] == old['old_training_order']
                  and stage.record['training_orders'] == old['old_train_orders']
                  and list(stage.valid) == old['old_validation_order'], 'Same failed fixed trajectory, not new permutations')
    return entries


if __name__ == '__main__':
    raise SystemExit(main())
