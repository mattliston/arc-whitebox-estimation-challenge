"""Process-local target-free digest memo; no global IO or hashlib replacement."""
import hashlib
import os
from pathlib import Path
import re
import stat


def require(value,message):
    if not value:
        raise RuntimeError(message)


class MetadataHashes:
    def __init__(self,root):
        self.root = Path(root).resolve()
        self.entries = {}
        self.guard = None

    @staticmethod
    def identity(value):
        require(stat.S_ISREG(value.st_mode),'Only regular frozen metadata/input files')
        return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)

    def path(self,value):
        path = Path(value).resolve(strict=True)
        require(path.is_relative_to(self.root) and 'truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),'Target arrays forbidden')
        return path

    def tick(self,tick):
        if self.guard is not None:
            self.guard()
        if tick is not None and tick is not self.guard:
            tick()

    def sha(self,value,tick=None,*,fresh=False):
        path = self.path(value)
        self.tick(tick)
        before = self.identity(path.stat())
        previous = self.entries.get(str(path))
        if previous is not None:
            require(previous[0] == before,'Frozen file metadata changed: '+str(path))
        if previous is not None and not fresh:
            self.tick(tick)
            require(self.identity(path.stat()) == before,'File changed during memoized digest check')
            return previous[1]
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            require(self.identity(os.fstat(stream.fileno())) == before,'Opened file identity matches frozen path')
            for block in iter(lambda:stream.read(1 << 20),b''):
                digest.update(block)
                self.tick(tick)
            require(self.identity(os.fstat(stream.fileno())) == before,'File changed while reading digest')
        after = self.identity(path.stat())
        require(after == before,'Path metadata changed while reading digest')
        value = digest.hexdigest()
        if previous is not None:
            require(value == previous[1],'Unconditional fresh digest differs from first checked bytes')
        self.entries[str(path)] = (before,value)
        self.tick(tick)
        return value

    def final_bind(self,mappings,tick=None):
        """Every unique named file is read again, irrespective of memo/stat hits."""
        require(isinstance(mappings,(tuple,list)) and mappings,'Explicit nonempty union of flat maps')
        union = {}
        for mapping in mappings:
            require(isinstance(mapping,dict),'Flat source/input/artifact map')
            for value,digest in mapping.items():
                require(Path(value).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest or ''),
                    'Every original expected SHA remains explicit')
                path = str(self.path(value))
                require(path not in union or union[path] == digest,'No conflicting expected digest in final union')
                union[path] = digest
                self.tick(tick)
        require(union,'No empty final verification')
        total = 0
        for path,expected in union.items():
            require(self.sha(path,tick,fresh=True) == expected,'Fresh final source/input/artifact SHA: '+path)
            total += self.entries[path][0][2]
        self.tick(tick)
        return dict(metadata_fresh_union_rehashed=True,
            metadata_unique_files_rehashed=len(union),metadata_unique_bytes_rehashed=total)
