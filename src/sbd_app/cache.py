from pathlib import Path
import hashlib
import json
import shutil
import time


def identity(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


class DiskCache:
    """Four input videos, at most 4 GiB/24h. Only the dedicated cache's direct children are evicted."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid cache key")
        path = self.root / key
        if path.is_symlink():
            raise ValueError("Cache directory must not be a symbolic link")
        path.mkdir(exist_ok=True)
        path.touch()
        return path

    def prune(self, active):
        entries = []
        for p in self.root.iterdir():
            if p.is_dir() and not p.is_symlink() and len(p.name) == 64 and all(c in "0123456789abcdef" for c in p.name):
                entries.append((p, sum(f.stat().st_size for f in p.rglob("*") if f.is_file()), p.stat().st_mtime))
        entries.sort(key=lambda v: v[2])
        size = sum(e[1] for e in entries)
        count = len(entries)
        for p, n, stamp in entries:
            if p == active:
                continue
            if count > 4 or size > 4 * 1024**3 or time.time() - stamp > 86400:
                if not p.resolve().is_relative_to(self.root):
                    raise ValueError("Refusing to delete outside cache")
                shutil.rmtree(p)
                count -= 1
                size -= n
