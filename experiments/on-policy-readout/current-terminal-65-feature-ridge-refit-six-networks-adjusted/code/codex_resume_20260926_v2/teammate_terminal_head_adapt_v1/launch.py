"""Root-reviewed serial controller; preparation does not execute this file."""
import subprocess
from common import *


def main():
    # The controller itself need not run inside a numerical guard. Its child
    # scripts enforce thread/nice settings set by the existing root guard.
    for entry in read(HERE/'seal.json')['sources']:verify_pin(entry)
    once(HERE/'launch_seal.json',dict(prepared_seal=pin(HERE/'seal.json'),
         root_invoked=True,bank=pin(BANK),component=pin(COMPONENT)))
    light=ROOT/'.venv/bin/python'
    native=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
    guardian=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
    def run(name,script,args=(),runtime=native,seconds=300):
        with (HERE/f'{name}.log').open('x') as log:
            result=subprocess.run([str(light),str(guardian),'--record',str(HERE/f'{name}_guard.json'),
                 '--seconds',str(seconds),str(runtime),str(script),*args],stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0;guard(name)
        print(__import__('json').dumps(dict(event='retired',stage=name)),flush=True)
    run('parity',HERE/'collect.py',('parity',))
    run('capture',HERE/'collect.py',('capture',),seconds=600)
    run('fit',HERE/'fit.py',runtime=native,seconds=60)
    # Unmodified actual-score runner: the official S31 is always the bank.
    with (HERE/'score_controller.log').open('x') as log:
        result=subprocess.run([str(light),str(H2/'team_score_screen_v1.py'),str(HERE)],
                              stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0
    run('actual_invariants',HERE/'verify_actual.py',runtime=light,seconds=60)


if __name__=='__main__':main()
