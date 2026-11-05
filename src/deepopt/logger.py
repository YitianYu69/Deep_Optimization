"""Small, scoped logging API with configuration-derived files and clean stdout.

No PyTorch dependency, root logger mutation, or stdout/stderr interception.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
from threading import RLock

from ._config_resolution import _UNSET, ResolvedConfig, _merge, resolve_config
from .config import ExperimentConfig


_active = ContextVar("deepopt_logging_session", default=None)
_lock = RLock()
_ansi = re.compile(r"\x1b\[[0-9;]*m")


def _main_process():
    # Keep config-only entry points independent of PyTorch imports.
    dist = sys.modules.get("torch.distributed")
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _config_dict(config):
    raw = asdict(config) if is_dataclass(config) and not isinstance(config, type) else config
    if not isinstance(raw, Mapping):
        raise TypeError("Logging configuration must be a dataclass instance or mapping")
    # Copy and validate without arbitrary repr(), tensor reads, or object addresses.
    encoded = json.dumps(dict(raw), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.loads(encoded), encoded


def _metadata_snapshot(config=None, *, base=_UNSET):
    # Logger metadata remains open-ended; only known complete configuration
    # sources go through the experiment schema. Extra recipe keys stay valid.
    if isinstance(config, (ResolvedConfig, ExperimentConfig, str, Path)):
        return resolve_config(config)
    inherited = resolve_config() if base is _UNSET else base
    supplied = {} if config is None else _config_dict(config)[0]
    return _merge(inherited, supplied)


def _filename(config) -> str:
    """Stable readable name plus a digest of ALL JSON-serializable settings."""
    raw, encoded = _config_dict(config)
    training, runtime = raw.get("training", {}), raw.get("runtime", {})
    parts = [str(raw.get("name", "training"))]
    for table, key, prefix in ((training, "batch_size", "bs"), (training, "learning_rate", "lr"),
                               (runtime, "precision", "")):
        if isinstance(table, dict) and key in table:
            parts.append(f"{prefix}{table[key]}")
    readable = re.sub(r"[^a-zA-Z0-9_.-]+", "-", "-".join(parts)).strip(".-")[:100] or "training"
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{readable}-{digest}.log"


def log_filename(config=None) -> str:
    """Name logs from an owned merge of current defaults and recipe metadata."""
    return _filename(_metadata_snapshot(config))


def _level(value):
    if isinstance(value, str):
        value = logging.getLevelNamesMapping().get(value.upper())
    if type(value) is not int or value < 0:
        raise ValueError("Logging level must be a level name or nonnegative integer")
    return value


class _Formatter(logging.Formatter):
    def format(self, record):
        text = _ansi.sub("", super().format(record))
        # Every traceback/multiline line remains identifiable in a shared terminal.
        prefix = f"{record.levelname} {self.formatTime(record, '%m-%d %H:%M:%S')} [{record.filename}:{record.lineno}] "
        return "\n".join(prefix + line for line in text.splitlines())


_formatter = _Formatter("%(message)s")


def _console(record, level):
    if getattr(record, "output_only", False) or record.levelno < level:
        return
    text = os.getenv("DEEPOPT_LOGGING_PREFIX", "") + _formatter.format(record)
    color = os.getenv("DEEPOPT_LOGGING_COLOR", "auto")
    if not os.getenv("NO_COLOR") and (color == "1" or (color == "auto" and sys.stderr.isatty())):
        code = {logging.DEBUG: 36, logging.INFO: 32, logging.WARNING: 33, logging.ERROR: 31,
                logging.CRITICAL: 31}.get(record.levelno, 0)
        text = f"\x1b[{code}m{text}\x1b[0m"
    # Avoid importing tqdm (or torch) for configuration-only entry points.
    tqdm_module = sys.modules.get("tqdm")
    tqdm_write = getattr(getattr(tqdm_module, "tqdm", None), "write", None)
    if callable(tqdm_write):
        tqdm_write(text, file=sys.stderr)
    else:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()


class _Router(logging.Handler):
    def emit(self, record):
        if not _main_process():
            return
        session = _active.get()
        if session is not None:
            session.emit(record)
        else:
            configured = resolve_config().get("logging", {}).get("level")
            _console(record, _level(configured if configured is not None else
                                   os.getenv("DEEPOPT_LOGGING_LEVEL", "INFO")))


def init_logger(name: str) -> logging.Logger:
    """Return a standard logger routed through DeepOpt's console/file handlers."""
    with _lock:
        package = logging.getLogger("deepopt")
        if not any(isinstance(handler, _Router) for handler in package.handlers):
            package.addHandler(_Router())
            package.setLevel(logging.DEBUG)
            package.propagate = False
    if name != "deepopt" and not name.startswith("deepopt."):
        name = f"deepopt.{name}"
    return logging.getLogger(name)


class _Session:
    def __init__(self, config, level, file_level):
        self.config, _ = _config_dict(config)
        options = self.config.get("logging", {})
        options = options if isinstance(options, Mapping) else {}
        if level is _UNSET:
            level = options.get("level")
        if file_level is _UNSET:
            file_level = options.get("file_level", "DEBUG")
        self.level = _level(level if level is not None else os.getenv("DEEPOPT_LOGGING_LEVEL", "INFO"))
        self.file_level = _level(file_level)
        self.path = None
        self._file = None
        self._pending = []

    def emit(self, record):
        _console(record, self.level)
        if record.levelno >= self.file_level:
            # Format immediately: do not retain tensors, exceptions, or mutable args.
            line = _formatter.format(record) + "\n"
            if self._file is None:
                self._pending.append(line)
            else:
                self._file.write(line)
                self._file.flush()

    def bind(self, directory, *, config=None):
        """Open once, after auto-sizing resolves the actual configuration."""
        if not _main_process():
            return None
        if self._file is not None:
            raise RuntimeError("Logging session already has a file")
        if config is not None:
            self.config, _ = _config_dict(_metadata_snapshot(config, base=self.config))
        path = Path(directory) / _filename(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self.path = path
        self._file.writelines(self._pending)
        self._pending.clear()
        self._file.flush()
        init_logger(__name__).info("Log file: %s", path)
        init_logger(__name__).debug("Resolved configuration: %s", json.dumps(self.config, sort_keys=True))
        return path

    def close(self):
        if self._file is not None:
            self._file.close()


@contextmanager
def logging_session(config=None, directory=None, *, level=_UNSET, file_level=_UNSET, failure_directory=None):
    """Snapshot logging policy; write a file only to an explicit destination.

    Omitted metadata/levels inherit current configuration without activating it.
    directory=None buffers until session.bind(). Configured runs use this to
    record preflight messages under the final, selected batch size. A failed
    preflight can flush to failure_directory without creating a run directory.
    Nested sessions are isolated and restore the outer session on exit.
    """
    session = _Session(_metadata_snapshot(config), level, file_level)
    token = _active.set(session)
    logger = init_logger(__name__)
    failure = None
    try:
        if directory is not None:
            session.bind(directory)
        logger.info("Logging session started")
        yield session
    except BaseException as exc:
        failure = exc
        try:
            logger.exception("Logging session failed")
            if session.path is None and failure_directory is not None:
                session.bind(failure_directory)
        except Exception as log_error:
            exc.add_note(f"Could not persist failure logs: {log_error}")
        raise
    else:
        logger.info("Logging session finished")
    finally:
        _active.reset(token)
        try:
            session.close()
        except OSError as close_error:
            if failure is None:
                raise
            failure.add_note(f"Could not close log file: {close_error}")


def log_output(*values, sep=" ", end="\n", file=None, flush=True):
    """Emit command/result text unchanged, also recording it in the active file.

    Use logger.info() for diagnostics; this print-compatible helper is for
    intentional stdout JSON and human-readable example results.
    """
    if not _main_process():
        return
    text = sep.join(str(value) for value in values)
    stream = sys.stdout if file is None else file
    stream.write(text + end)
    if flush:
        stream.flush()
    init_logger("deepopt.output").info(text, extra={"output_only": True}, stacklevel=2)
