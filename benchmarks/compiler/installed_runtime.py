"""Validate the installed DeepOpt distribution without using a checkout copy.

Importing this helper is standard-library only. Runtime identity is resolved
lazily; dry-run plans and exports require neither DeepOpt nor Torch.
"""

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy


ROOT = Path(__file__).resolve().parents[2]
IDENTITY_ENV = "DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY"
_SESSION = ContextVar("deepopt_installed_identity_session", default=None)


def prepare_imports(root=ROOT):
    root = Path(root).resolve()
    checkout = root / "src"
    cleaned = []
    for value in sys.path:
        path = Path(value or Path.cwd()).resolve()
        if path == checkout or checkout in path.parents:
            continue
        if path not in (root, root / "benchmarks"):
            cleaned.append(value)
    sys.path[:] = [str(root / "benchmarks"), str(root), *cleaned]
    importlib.invalidate_caches()


def installed_identity(root=ROOT, *, full=False):
    """Return and freeze the imported non-editable wheel/target installation.

    RECORD ownership distinguishes a pip-installed package from a source tree
    or editable installation. Package shadowing fails instead of falling back.
    """
    session = _SESSION.get()
    if session is not None and not full and Path(root).resolve() == session.root:
        return session.validate()
    prepare_imports(root)
    try:
        distribution = importlib.metadata.distribution("deep-optimization")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Install the current DeepOpt wheel in this Python environment before running benchmarks") from error
    direct = distribution.read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise RuntimeError("Editable DeepOpt installs use checkout source; install a non-editable wheel for qualification")
    records = distribution.files
    if not distribution.read_text("RECORD") or not records:
        raise RuntimeError("DeepOpt has no installed-wheel RECORD; a checkout/egg-info package cannot qualify")
    entry = next((entry for entry in records if entry.as_posix() == "deepopt/__init__.py"), None)
    if entry is None:
        raise RuntimeError("Installed distribution does not own deepopt/__init__.py; source/editable imports cannot qualify")
    expected = Path(distribution.locate_file(entry)).resolve()
    install_root = Path(distribution.locate_file("")).resolve()
    if not expected.is_file() or not expected.is_relative_to(install_root):
        raise RuntimeError("Installed DeepOpt package files are absent or resolve outside their installation")
    package = importlib.import_module("deepopt")
    actual = Path(package.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"DeepOpt import is shadowed: imported {actual}; installed distribution owns {expected}")
    directory = expected.parent
    for name, module in tuple(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if name.startswith("deepopt.") and location and Path(location).is_file():
            if not Path(location).resolve().is_relative_to(directory):
                raise RuntimeError(f"DeepOpt submodule {name} comes from outside the installed package: {location}")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if not path.resolve().is_relative_to(directory):
            raise RuntimeError(f"Installed DeepOpt file resolves to external source: {path}")
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
        count += 1
    identity = {"distribution_name": distribution.metadata["Name"], "version": distribution.version,
                "package_path": str(directory), "import_path": str(actual), "install_root": str(install_root),
                "source_sha256": digest.hexdigest(), "files_hashed": count, "editable": False}
    frozen = os.environ.get(IDENTITY_ENV)
    if frozen and json.loads(frozen) != identity:
        raise RuntimeError("Installed DeepOpt identity differs from the parent run; use the same unchanged installation")
    return identity


def _file_stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)


class InstalledIdentitySession:
    """Full content hashes at both boundaries; immutable attribution between.

    Every reuse checks tree membership, resolved paths, nanosecond mtime/ctime,
    inode/size/mode, wheel metadata and imported module ownership. Ctime detects
    changes even when the caller restores the original bytes and mtime. No
    cache crosses a process or group boundary.
    """

    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.identity = installed_identity(self.root, full=True)
        self.directory = Path(self.identity["package_path"])
        self.distribution = importlib.metadata.distribution("deep-optimization")
        self.metadata = tuple(self.distribution.read_text(name) for name in ("RECORD", "METADATA", "direct_url.json"))
        self.stamps = self._stamps()
        self.reuses = 0

    def _stamps(self):
        files = {}
        for path in self.directory.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if not path.resolve().is_relative_to(self.directory):
                raise RuntimeError(f"Installed DeepOpt file resolves to external source: {path}")
            files[str(path)] = _file_stamp(path)
        return files

    def validate(self):
        if (self._stamps() != self.stamps or self.metadata != tuple(self.distribution.read_text(name)
                for name in ("RECORD", "METADATA", "direct_url.json"))):
            raise RuntimeError("Installed DeepOpt files or wheel metadata changed inside the identity group")
        frozen = os.environ.get(IDENTITY_ENV)
        if frozen and json.loads(frozen) != self.identity:
            raise RuntimeError("Installed DeepOpt identity differs from the parent run")
        for name, module in tuple(sys.modules.items()):
            location = getattr(module, "__file__", None)
            if name == "deepopt" and location and Path(location).resolve() != Path(self.identity["import_path"]):
                raise RuntimeError("Installed DeepOpt import is shadowed inside the identity group")
            if name.startswith("deepopt.") and location and not Path(location).resolve().is_relative_to(self.directory):
                raise RuntimeError(f"Installed DeepOpt submodule is shadowed: {name}")
        self.reuses += 1
        return deepcopy(self.identity)

    def finish(self):
        self.validate()
        if installed_identity(self.root, full=True) != self.identity:
            raise RuntimeError("Installed DeepOpt content changed across the identity group")


@contextmanager
def identity_session(root=ROOT):
    """Reuse validated imports/hashes only within one immutable worker group."""
    active = _SESSION.get()
    if active is not None:
        if active.root != Path(root).resolve():
            raise RuntimeError("Nested identity groups must use the same benchmark root")
        active.validate()
        yield active
        active.validate()
        return
    session = InstalledIdentitySession(root)
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        try:
            session.finish()
        finally:
            _SESSION.reset(token)


def child_environment(root=ROOT, identity=None, base=None):
    root = Path(root).resolve()
    identity = installed_identity(root) if identity is None else identity
    environment = dict(os.environ if base is None else base)
    # Preserve the selected installed --target directory, not arbitrary source
    # entries inherited from PYTHONPATH. Remaining dependencies use site paths.
    environment["PYTHONPATH"] = os.pathsep.join((identity["install_root"], str(root / "benchmarks"), str(root)))
    environment[IDENTITY_ENV] = json.dumps(identity, sort_keys=True)
    return environment
