"""Only closed-call retirement; reuse the authenticated allocator ABI helper."""
import hashlib
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent/'current_full_lower_p_allocator_v1/allocator.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '11b80a8e327915261f71b50cf24610df6c5f7483b5ceb9fdb435fa4cb513ec6e'
spec = importlib.util.spec_from_file_location('qualified_native_allocator_helper',SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
Allocator = helper.Allocator
rss = helper.rss
