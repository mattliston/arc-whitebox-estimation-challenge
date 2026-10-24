"""Fixed v2 scalar-gated official-input native35/100/35; no truth access."""
import common as h
from check_scalar import verify_scalar


if __name__ == '__main__':
    runner = h.template('run_native.py')
    previous_gate = runner.small_gate

    def full_gate(bindings):
        result = previous_gate(bindings)
        result.update(verify_scalar(bindings))
        return result

    runner.small_gate = full_gate
    raise SystemExit(runner.main())
