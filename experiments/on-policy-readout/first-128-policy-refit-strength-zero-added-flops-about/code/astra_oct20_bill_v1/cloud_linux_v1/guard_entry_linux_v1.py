"""Wait until the guard has assigned the child group to its cgroup, then exec."""
import os
import sys

fd = int(sys.argv[1])
assert os.read(fd, 1) == b'1', 'guard did not admit child'
os.close(fd)
os.environ['ARC_GUARDED_PGID'] = str(os.getpgrp())
os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
