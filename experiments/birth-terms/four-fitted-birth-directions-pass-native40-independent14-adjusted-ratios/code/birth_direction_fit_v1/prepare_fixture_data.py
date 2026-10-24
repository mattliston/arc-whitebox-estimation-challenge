"""Exclusive fixed zero/nonzero deployment data for implementation checks."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    if not args.write: parser.print_help(); return
    import numpy as np
    for label,theta in (('zero',np.zeros((15,4),np.float32)),
                        ('nonzero',(.1*np.sin(np.arange(60,dtype=float)+.3)).reshape(15,4).astype(np.float32))):
        with (HERE/f'fixture_{label}_theta.json').open('x') as f:
            json.dump(dict(theta=theta.tolist(),qualification_fixture=True,trained=False),f,indent=2)
            f.write('\n')


if __name__=='__main__':
    main()
