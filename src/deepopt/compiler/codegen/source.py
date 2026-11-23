"""Atomic source caching without importing a target's optional compiler."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading


_LOCK = threading.RLock()


def write_source(source, cache_dir=None):
    root = Path(cache_dir or os.environ.get("DEEPOPT_COMPILER_CACHE_DIR") or
                (Path(tempfile.gettempdir()) / "deepopt_compiler")).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()[:24]
    path = root / f"tilelang_{digest}.py"
    with _LOCK:
        if path.exists():
            if path.read_text() != source:
                raise RuntimeError(f"Generated kernel source cache mismatch: {path}")
            return path
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", dir=root)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(source)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
    return path


def load_source(path):
    name = f"deepopt_compiler_{path.stem}"
    with _LOCK:
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load generated source: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module
