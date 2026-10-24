"""Reuse the independent tiny arithmetic gate for the direct-write kernel."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "check_packed_deep_kernel.py").read_text()
source = source.replace("candidate_binary_learned_d6_v1.py", "candidate_binary_learned_d6_direct_v1.py")
source = source.replace("packed_deep_kernel_v1.json", "direct_transform_arithmetic_v1.json")
exec(compile(source, str(HERE / "check_packed_deep_kernel.py"), "exec"),
     dict(__file__=str(HERE / "check_packed_deep_kernel.py"), __name__="__main__"))
